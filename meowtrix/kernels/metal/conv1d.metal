// Fused causal depthwise conv1d kernels for the Qwen3.5 GatedDeltaNet layers.
//
// All tensors use the conv layout [B, D, L] (channels-first). The convolution
// is causal: output[t] = sum_k w[k] * x[t - (K-1) + k], with a left zero pad of
// K-1 taps. Activation is applied in fp32 for numerical parity with PyTorch.

#include <metal_stdlib>
using namespace metal;

enum ConvAct : uint {
  CONV_ACT_NONE = 0,
  CONV_ACT_SILU = 1,
};

static inline float apply_act(float v, uint act) {
  if (act == CONV_ACT_SILU) {
    return v / (1.0f + exp(-v));
  }
  return v;
}

static inline float silu_grad(float c) {
  float s = 1.0f / (1.0f + exp(-c));
  return s * (1.0f + c * (1.0f - s));
}

// Forward: out[b,d,t] = act(bias[d] + sum_k w[d,k] * x[b,d,t-(K-1)+k])
// Each thread produces TILE consecutive outputs along L so the sliding window
// loads amortize (K+TILE-1 loads for TILE outputs instead of K per output).
template <typename T>
kernel void causal_conv1d_fwd(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& L [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& has_bias [[buffer(7)]],
    constant uint& act [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]) {
  constexpr uint TILE = 4;
  uint t0 = gid.x * TILE;
  uint d = gid.y;
  uint b = gid.z;
  if (t0 >= L || d >= D) {
    return;
  }

  const device T* xr = x + (size_t)b * D * L + (size_t)d * L;
  const device T* wr = w + (size_t)d * K;
  device T* orow = out + (size_t)b * D * L + (size_t)d * L;
  float biasv = has_bias ? float(bias[d]) : 0.0f;

  float wv[8];
  for (uint k = 0; k < K; ++k) {
    wv[k] = float(wr[k]);
  }
  float acc[TILE];
  for (uint i = 0; i < TILE; ++i) {
    acc[i] = biasv;
  }
  // taps shared across the tile: xr[t0-(K-1)+k] for k in [0, K+TILE-1)
  for (uint k = 0; k < K + TILE - 1; ++k) {
    long idx = long(t0) + long(k) - long(K - 1);
    if (idx < 0 || idx >= long(L)) {
      continue;
    }
    float xv = float(xr[idx]);
    // tap k contributes w[j] * x to output t0+i where k = j + i
    for (uint i = 0; i < TILE; ++i) {
      if (k >= i && k - i < K) {
        acc[i] += wv[k - i] * xv;
      }
    }
  }
  for (uint i = 0; i < TILE; ++i) {
    if (t0 + i < L) {
      orow[t0 + i] = T(apply_act(acc[i], act));
    }
  }
}

// Backward w.r.t. the input. The pre-activation conv output is recomputed in
// registers (K^2 MACs per element) instead of stored, which halves the memory
// traffic of the combined fwd+bwd path:
//   dx[t] = sum_k w[k] * dy[u] * silu'(conv_pre[u])   with u = t + (K-1) - k
// i.e. a forward-looking window (right zero-pad) over gated grads.
template <typename T>
kernel void causal_conv1d_bwd_dx(
    const device T* dy [[buffer(0)]],
    const device T* x [[buffer(1)]],
    const device T* w [[buffer(2)]],
    const device T* bias [[buffer(3)]],
    device T* dx [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& L [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]) {
  uint t = gid.x;
  uint d = gid.y;
  uint b = gid.z;
  if (t >= L || d >= D) {
    return;
  }

  const device T* dr = dy + (size_t)b * D * L + (size_t)d * L;
  const device T* xr = x + (size_t)b * D * L + (size_t)d * L;
  const device T* wr = w + (size_t)d * K;
  float biasv = has_bias ? float(bias[d]) : 0.0f;

  float acc = 0.0f;
  for (uint k = 0; k < K; ++k) {
    uint u = t + (K - 1) - k;
    if (u >= L) {
      continue;
    }
    // recompute pre-activation at u: conv[u] = bias + sum_j w[j] * x[u-(K-1)+j]
    float pre = biasv;
    for (uint j = 0; j < K; ++j) {
      long xi = long(u) + long(j) - long(K - 1);
      if (xi >= 0) {
        pre += float(wr[j]) * float(xr[xi]);
      }
    }
    acc += float(wr[k]) * float(dr[u]) * silu_grad(pre);
  }
  dx[(size_t)b * D * L + (size_t)d * L + t] = T(acc);
}

// Single- or few-token decode update. conv_state [B,D,K] is updated in place
// to the last K entries of [state, x]. Output y[b,d,s] for s in [0,S):
//   y[s] = act(bias[d] + sum_k w[d,k] * xn[s+1+k])   with xn = [state, x]
// One thread per (b,d); sequential over s (S is tiny at decode).
template <typename T>
kernel void causal_conv1d_update(
    const device T* x [[buffer(0)]],
    device T* state [[buffer(1)]],
    const device T* w [[buffer(2)]],
    const device T* bias [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& S [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    constant uint& act [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint d = gid.x;
  uint b = gid.y;
  if (d >= D) {
    return;
  }

  const device T* xr = x + (size_t)b * D * S + (size_t)d * S;
  device T* sr = state + (size_t)b * D * K + (size_t)d * K;
  const device T* wr = w + (size_t)d * K;
  device T* orow = out + (size_t)b * D * S + (size_t)d * S;

  // combined window: xn[j] = sr[j] for j<K else xr[j-K], total K+S entries
  float xn[16];
  for (uint j = 0; j < K; ++j) {
    xn[j] = float(sr[j]);
  }
  for (uint j = 0; j < S; ++j) {
    xn[K + j] = float(xr[j]);
  }
  for (uint s = 0; s < S; ++s) {
    float acc = has_bias ? float(bias[d]) : 0.0f;
    for (uint k = 0; k < K; ++k) {
      acc += float(wr[k]) * xn[s + 1 + k];
    }
    orow[s] = T(apply_act(acc, act));
  }
  for (uint j = 0; j < K; ++j) {
    sr[j] = T(xn[S + j]);
  }
}

#define INSTANTIATE_CONV1D(T, suffix)                                       \
  template [[host_name("causal_conv1d_fwd_" #suffix)]]                      \
      kernel void causal_conv1d_fwd<T>(                                     \
          const device T*, const device T*, const device T*, device T*,     \
          constant uint&, constant uint&, constant uint&, constant uint&,   \
          constant uint&, uint3);                                           \
  template [[host_name("causal_conv1d_bwd_dx_" #suffix)]]                   \
      kernel void causal_conv1d_bwd_dx<T>(                                  \
          const device T*, const device T*, const device T*,                \
          const device T*, device T*, constant uint&, constant uint&,       \
          constant uint&, constant uint&, uint3);                           \
  template [[host_name("causal_conv1d_update_" #suffix)]]                   \
      kernel void causal_conv1d_update<T>(                                  \
          const device T*, device T*, const device T*, const device T*,     \
          device T*, constant uint&, constant uint&, constant uint&,        \
          constant uint&, constant uint&, uint2);

INSTANTIATE_CONV1D(float, f32)
INSTANTIATE_CONV1D(half, f16)
INSTANTIATE_CONV1D(bfloat, bf16)
