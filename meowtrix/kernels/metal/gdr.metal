// Gated delta-rule kernels for the Qwen3.5 GatedDeltaNet linear-attention
// layers. Semantics mirror transformers' torch_recurrent_gated_delta_rule:
//   S      <- exp(g_t) * S
//   kv_t   = S^T k_t
//   delta  = (v_t - kv_t) * beta_t
//   S     += k_t (x) delta
//   out_t  = S'^T q_t
// where q,k are L2-normalized in fp32 (eps 1e-6) and q is scaled by dk^-0.5.
// The recurrent state S [dk, dv] stays in fp32 device memory; each output
// column v is owned by a single thread so the sequential scan needs no
// threadgroup barriers at all.

#include <metal_stdlib>
using namespace metal;

// Row-normalize pass for the sequential scan: computes fp32 L2-normalized
// qn,kn (qn additionally scaled by dk^-0.5) for every (b,t,h) row. Grid is
// (Hk, B*L) so the flat row index is gid.y; one thread per row.
template <typename T>
kernel void gdr_qknorm_rows(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    device float* qn [[buffer(2)]],
    device float* kn [[buffer(3)]],
    constant uint& Hk [[buffer(4)]],
    constant uint& dk [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint h = gid.x;
  size_t row = gid.y;  // b * L + t
  const device T* qr = q + (row * Hk + h) * dk;
  const device T* kr = k + (row * Hk + h) * dk;
  device float* qnr = qn + (row * Hk + h) * dk;
  device float* knr = kn + (row * Hk + h) * dk;

  float ss = 0.0f;
  for (uint d = 0; d < dk; ++d) {
    float x = float(qr[d]);
    qnr[d] = x;
    ss += x * x;
  }
  float inv = rsqrt(ss + 1e-6f) * rsqrt((float)dk);
  for (uint d = 0; d < dk; ++d) {
    qnr[d] *= inv;
  }
  ss = 0.0f;
  for (uint d = 0; d < dk; ++d) {
    float x = float(kr[d]);
    knr[d] = x;
    ss += x * x;
  }
  inv = rsqrt(ss + 1e-6f);
  for (uint d = 0; d < dk; ++d) {
    knr[d] *= inv;
  }
}

// Single-token decode step with in-kernel normalization. One threadgroup of
// `threads` lanes per (b, hv); lane v owns state column v end-to-end.
template <typename T>
kernel void gdr_step(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device T* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device T* out [[buffer(6)]],
    constant uint& dk [[buffer(7)]],
    constant uint& dv [[buffer(8)]],
    constant uint& Hk [[buffer(9)]],
    constant uint& Hv [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  // grid is (threads, Hv, B) with a full-width threadgroup, so the head and
  // batch indices live on the y/z threadgroup axes.
  uint h = tg.y;
  uint b = tg.z;
  uint hk = h / (Hv / Hk);

  const device T* qr = q + ((size_t)b * Hk + hk) * dk;
  const device T* kr = k + ((size_t)b * Hk + hk) * dk;
  const device T* vr = v + ((size_t)b * Hv + h) * dv;
  device float* S = state + ((size_t)b * Hv + h) * dk * dv;
  device T* orow = out + ((size_t)b * Hv + h) * dv;

  threadgroup float qn[256];
  threadgroup float kn[256];
  threadgroup float nrm[2];
  if (tid < dk) {
    qn[tid] = float(qr[tid]);
    kn[tid] = float(kr[tid]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float sq = 0.0f, sk = 0.0f;
    for (uint d = 0; d < dk; ++d) {
      sq += qn[d] * qn[d];
      sk += kn[d] * kn[d];
    }
    nrm[0] = rsqrt(sq + 1e-6f) * rsqrt((float)dk);
    nrm[1] = rsqrt(sk + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < dk) {
    qn[tid] *= nrm[0];
    kn[tid] *= nrm[1];
  }
  float dec = exp(g[(size_t)b * Hv + h]);
  float bt = float(beta[(size_t)b * Hv + h]);
  threadgroup float delta[256];
  if (tid < dv) {
    float acc = 0.0f;
    for (uint d = 0; d < dk; ++d) {
      float s = dec * S[(size_t)d * dv + tid];
      S[(size_t)d * dv + tid] = s;
      acc += s * kn[d];
    }
    delta[tid] = (float(vr[tid]) - acc) * bt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < dv) {
    float acc = 0.0f;
    for (uint d = 0; d < dk; ++d) {
      float s = S[(size_t)d * dv + tid] + kn[d] * delta[tid];
      S[(size_t)d * dv + tid] = s;
      acc += s * qn[d];
    }
    orow[tid] = T(acc);
  }
}

// Sequential scan over the sequence for inference/prefill. Consumes the
// pre-normalized fp32 qn,kn rows produced by gdr_qknorm_rows. One threadgroup
// per (b, hv, dv-block): the state block [DVB][dk] is staged TRANSPOSED in
// threadgroup memory (contiguous per-column rows), each lane owns one column
// for the whole sequence — zero barriers inside the scan.
#define GDR_DVB 32
template <typename T>
kernel void gdr_seq(
    const device float* qn [[buffer(0)]],
    const device float* kn [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device T* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device T* out [[buffer(6)]],
    constant uint& L [[buffer(7)]],
    constant uint& dk [[buffer(8)]],
    constant uint& dv [[buffer(9)]],
    constant uint& Hk [[buffer(10)]],
    constant uint& Hv [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  uint h = tg.y / (dv / GDR_DVB);
  uint vb = (tg.y % (dv / GDR_DVB)) * GDR_DVB;
  uint b = tg.z;
  uint hk = h / (Hv / Hk);
  uint vv = tid;
  if (vv >= GDR_DVB || vb + vv >= dv) {
    return;
  }

  device float* S = state + ((size_t)b * Hv + h) * dk * dv;
  // transposed private copy: Stg[col][d]
  threadgroup float Stg[GDR_DVB][128];
  for (uint d = 0; d < dk; ++d) {
    Stg[vv][d] = S[(size_t)d * dv + vb + vv];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint t = 0; t < L; ++t) {
    size_t row = (size_t)b * L + t;
    const device float* krow = kn + (row * Hk + hk) * dk;
    const device float* qrow = qn + (row * Hv + h) * dk;
    float dec = exp(g[row * Hv + h]);
    float bt = float(beta[row * Hv + h]);
    float vv_in = float(v[row * Hv * dv + (size_t)h * dv + vb + vv]);

    // pass 1: decay S column + accumulate kv = S'^T k — 4 independent
    // accumulator chains interleave to hide tg/L2 latency.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint d = 0;
    for (; d + 4 <= dk; d += 4) {
      float s0 = dec * Stg[vv][d];
      float s1 = dec * Stg[vv][d + 1];
      float s2 = dec * Stg[vv][d + 2];
      float s3 = dec * Stg[vv][d + 3];
      Stg[vv][d] = s0;
      Stg[vv][d + 1] = s1;
      Stg[vv][d + 2] = s2;
      Stg[vv][d + 3] = s3;
      a0 += s0 * krow[d];
      a1 += s1 * krow[d + 1];
      a2 += s2 * krow[d + 2];
      a3 += s3 * krow[d + 3];
    }
    for (; d < dk; ++d) {
      float s = dec * Stg[vv][d];
      Stg[vv][d] = s;
      a0 += s * krow[d];
    }
    float delta = (vv_in - (a0 + a1 + a2 + a3)) * bt;

    // pass 2: rank-1 update + out = S^T q on the same private column
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    d = 0;
    for (; d + 4 <= dk; d += 4) {
      float s0 = Stg[vv][d] + krow[d] * delta;
      float s1 = Stg[vv][d + 1] + krow[d + 1] * delta;
      float s2 = Stg[vv][d + 2] + krow[d + 2] * delta;
      float s3 = Stg[vv][d + 3] + krow[d + 3] * delta;
      Stg[vv][d] = s0;
      Stg[vv][d + 1] = s1;
      Stg[vv][d + 2] = s2;
      Stg[vv][d + 3] = s3;
      o0 += s0 * qrow[d];
      o1 += s1 * qrow[d + 1];
      o2 += s2 * qrow[d + 2];
      o3 += s3 * qrow[d + 3];
    }
    for (; d < dk; ++d) {
      float s = Stg[vv][d] + krow[d] * delta;
      Stg[vv][d] = s;
      o0 += s * qrow[d];
    }
    out[row * Hv * dv + (size_t)h * dv + vb + vv] = T(o0 + o1 + o2 + o3);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = 0; d < dk; ++d) {
    S[(size_t)d * dv + vb + vv] = Stg[vv][d];
  }
}

#define INSTANTIATE_GDR(T, suffix)                                          \
  template [[host_name("gdr_qknorm_rows_" #suffix)]]                        \
      kernel void gdr_qknorm_rows<T>(                                       \
          const device T*, const device T*, device float*, device float*,   \
          constant uint&, constant uint&, uint2);                           \
  template [[host_name("gdr_step_" #suffix)]]                               \
      kernel void gdr_step<T>(                                              \
          const device T*, const device T*, const device T*,                \
          const device float*, const device T*, device float*, device T*,   \
          constant uint&, constant uint&, constant uint&, constant uint&,   \
          uint, uint3);                                                     \
  template [[host_name("gdr_seq_" #suffix)]]                                \
      kernel void gdr_seq<T>(                                               \
          const device float*, const device float*, const device T*,        \
          const device float*, const device T*, device float*, device T*,   \
          constant uint&, constant uint&, constant uint&, constant uint&,   \
          constant uint&, uint, uint3);

INSTANTIATE_GDR(float, f32)
INSTANTIATE_GDR(half, f16)
INSTANTIATE_GDR(bfloat, bf16)
