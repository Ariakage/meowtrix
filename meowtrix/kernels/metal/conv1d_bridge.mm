// Bridge between PyTorch MPS tensors and the fused conv1d Metal kernels.
//
// Tensors arrive as [B, D, L] conv-layout views (contiguous, enforced by the
// Python wrapper). Kernels are enqueued on the current MPSStream's shared
// command buffer via commandEncoder()/endKernelCoalescing(), matching how
// ATen's own Metal compute ops dispatch work.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <torch/extension.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef METAL_SRC_PATH
#error "METAL_SRC_PATH must point at conv1d.metal"
#endif

#ifndef GDR_SRC_PATH
#error "GDR_SRC_PATH must point at gdr.metal"
#endif

namespace {

using at::mps::MPSStream;

NSString* dtypeSuffix(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kFloat:
      return @"f32";
    case at::kHalf:
      return @"f16";
    case at::kBFloat16:
      return @"bf16";
    default:
      TORCH_CHECK(false, "meowtrix metal conv1d: unsupported dtype ", t.scalar_type());
  }
}

id<MTLComputePipelineState> pipeline(const std::string& name, const at::Tensor& ref) {
  static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cache = nil;
  static id<MTLLibrary> library = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    cache = [[NSMutableDictionary alloc] init];
    NSError* error = nil;
    NSString* src = [NSString stringWithContentsOfFile:@(METAL_SRC_PATH)
                                            encoding:NSUTF8StringEncoding
                                               error:&error];
    TORCH_CHECK(src != nil, "meowtrix metal conv1d: failed to read kernel source: ",
                error.localizedDescription.UTF8String);
    NSString* gsrc = [NSString stringWithContentsOfFile:@(GDR_SRC_PATH)
                                             encoding:NSUTF8StringEncoding
                                                error:&error];
    TORCH_CHECK(gsrc != nil, "meowtrix metal gdr: failed to read kernel source: ",
                error.localizedDescription.UTF8String);
    src = [src stringByAppendingString:gsrc];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 14.0, *)) {
      opts.languageVersion = MTLLanguageVersion3_1;
    }
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    library = [device newLibraryWithSource:src options:opts error:&error];
    [opts release];
    TORCH_CHECK(library != nil, "meowtrix metal kernels: metal compile failed: ",
                error.localizedDescription.UTF8String);
  });

  NSString* key = [NSString stringWithFormat:@"%s_%@", name.c_str(), dtypeSuffix(ref)];
  id<MTLComputePipelineState> pso = cache[key];
  if (pso == nil) {
    id<MTLFunction> fn = [library newFunctionWithName:key];
    TORCH_CHECK(fn != nil, "meowtrix metal conv1d: kernel not found: ", key.UTF8String);
    NSError* error = nil;
    pso = [library.device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    TORCH_CHECK(pso != nil, "meowtrix metal conv1d: pipeline creation failed: ",
                error.localizedDescription.UTF8String);
    cache[key] = pso;
    [pso release];
  }
  return pso;
}

inline void setBuffer(id<MTLComputeCommandEncoder> enc, const at::Tensor& t, NSUInteger idx) {
  [enc setBuffer:at::native::mps::getMTLBufferStorage(t)
          offset:t.storage_offset() * t.element_size()
         atIndex:idx];
}

inline void setOptionalBuffer(id<MTLComputeCommandEncoder> enc,
                              const std::optional<at::Tensor>& t,
                              const at::Tensor& fallback,
                              NSUInteger idx) {
  setBuffer(enc, t.has_value() ? *t : fallback, idx);
}

}  // namespace

namespace meowtrix::metal {

at::Tensor conv1d_fwd(const at::Tensor& x,
                      const at::Tensor& w,
                      const std::optional<at::Tensor>& bias,
                      bool silu) {
  TORCH_CHECK(x.dim() == 3, "x must be [B,D,L]");
  const auto B = x.size(0), D = x.size(1), L = x.size(2), K = w.size(-1);
  TORCH_CHECK(K <= 8, "meowtrix metal conv1d: kernel width > 8 unsupported");

  auto out = at::empty_like(x);

  uint32_t Db = (uint32_t)D, Lb = (uint32_t)L, Kb = (uint32_t)K;
  uint32_t hasBias = bias.has_value() ? 1u : 0u;
  uint32_t act = silu ? 1u : 0u;

  MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pipeline("causal_conv1d_fwd", x)];
    setBuffer(enc, x, 0);
    setBuffer(enc, w, 1);
    setOptionalBuffer(enc, bias, w, 2);
    setBuffer(enc, out, 3);
    [enc setBytes:&Db length:sizeof(Db) atIndex:4];
    [enc setBytes:&Lb length:sizeof(Lb) atIndex:5];
    [enc setBytes:&Kb length:sizeof(Kb) atIndex:6];
    [enc setBytes:&hasBias length:sizeof(hasBias) atIndex:7];
    [enc setBytes:&act length:sizeof(act) atIndex:8];
    MTLSize grid = MTLSizeMake((NSUInteger)(L + 3) / 4, (NSUInteger)D, (NSUInteger)B);
    MTLSize tg = MTLSizeMake(64, 4, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stream->endKernelCoalescing();
  });
  return out;
}

// Backward w.r.t. input only (conv weight is frozen under LoRA; dw/db use a
// torch fallback when required). The kernel recomputes the pre-activation in
// registers rather than reading a saved tensor.
at::Tensor conv1d_bwd_dx(const at::Tensor& dy,
                         const at::Tensor& x,
                         const at::Tensor& w,
                         const std::optional<at::Tensor>& bias) {
  const auto B = dy.size(0), D = dy.size(1), L = dy.size(2), K = w.size(-1);
  auto dx = at::empty_like(dy);

  uint32_t Db = (uint32_t)D, Lb = (uint32_t)L, Kb = (uint32_t)K;
  uint32_t hasBias = bias.has_value() ? 1u : 0u;

  MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pipeline("causal_conv1d_bwd_dx", dy)];
    setBuffer(enc, dy, 0);
    setBuffer(enc, x, 1);
    setBuffer(enc, w, 2);
    setOptionalBuffer(enc, bias, w, 3);
    setBuffer(enc, dx, 4);
    [enc setBytes:&Db length:sizeof(Db) atIndex:5];
    [enc setBytes:&Lb length:sizeof(Lb) atIndex:6];
    [enc setBytes:&Kb length:sizeof(Kb) atIndex:7];
    [enc setBytes:&hasBias length:sizeof(hasBias) atIndex:8];
    MTLSize grid = MTLSizeMake((NSUInteger)L, (NSUInteger)D, (NSUInteger)B);
    MTLSize tg = MTLSizeMake(64, 4, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stream->endKernelCoalescing();
  });
  return dx;
}

// Decode-time stateful update: mutates conv_state in place, returns y.
at::Tensor conv1d_update(const at::Tensor& x,
                         const at::Tensor& state,
                         const at::Tensor& w,
                         const std::optional<at::Tensor>& bias,
                         bool silu) {
  TORCH_CHECK(x.dim() == 3, "x must be [B,D,S]");
  const auto B = x.size(0), D = x.size(1), S = x.size(2), K = state.size(-1);
  TORCH_CHECK(K + S <= 16, "meowtrix metal conv1d update: K+S exceeds register window");

  auto out = at::empty_like(x);

  uint32_t Db = (uint32_t)D, Sb = (uint32_t)S, Kb = (uint32_t)K;
  uint32_t hasBias = bias.has_value() ? 1u : 0u;
  uint32_t act = silu ? 1u : 0u;

  MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pipeline("causal_conv1d_update", x)];
    setBuffer(enc, x, 0);
    setBuffer(enc, state, 1);
    setBuffer(enc, w, 2);
    setOptionalBuffer(enc, bias, w, 3);
    setBuffer(enc, out, 4);
    [enc setBytes:&Db length:sizeof(Db) atIndex:5];
    [enc setBytes:&Sb length:sizeof(Sb) atIndex:6];
    [enc setBytes:&Kb length:sizeof(Kb) atIndex:7];
    [enc setBytes:&hasBias length:sizeof(hasBias) atIndex:8];
    [enc setBytes:&act length:sizeof(act) atIndex:9];
    MTLSize grid = MTLSizeMake((NSUInteger)D, (NSUInteger)B, 1);
    MTLSize tg = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stream->endKernelCoalescing();
  });
  return out;
}

// ---- Gated delta rule -----------------------------------------------------

// fp32 L2-normalized q/k rows for every (b,t,h); qn additionally scaled by
// dk^-0.5. Inputs are [B, L, H, dk] contiguous.
std::vector<at::Tensor> gdr_norm(const at::Tensor& q, const at::Tensor& k) {
  const auto B = q.size(0), L = q.size(1), Hk = q.size(2), dk = q.size(3);
  auto opts = q.options().dtype(at::kFloat);
  auto qn = at::empty({B, L, Hk, dk}, opts);
  auto kn = at::empty({B, L, Hk, dk}, opts);

  uint32_t Hkb = (uint32_t)Hk, dkb = (uint32_t)dk;

  MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pipeline("gdr_qknorm_rows", q)];
    setBuffer(enc, q, 0);
    setBuffer(enc, k, 1);
    setBuffer(enc, qn, 2);
    setBuffer(enc, kn, 3);
    [enc setBytes:&Hkb length:sizeof(Hkb) atIndex:4];
    [enc setBytes:&dkb length:sizeof(dkb) atIndex:5];
    MTLSize grid = MTLSizeMake((NSUInteger)Hk, (NSUInteger)(B * L), 1);
    MTLSize tg = MTLSizeMake(32, 32, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stream->endKernelCoalescing();
  });
  return {qn, kn};
}

// Single-token decode step: mutates `state` in place, returns out [B,Hv,dv].
at::Tensor gdr_step(const at::Tensor& q,
                    const at::Tensor& k,
                    const at::Tensor& v,
                    const at::Tensor& g,
                    const at::Tensor& beta,
                    const at::Tensor& state) {
  const auto B = q.size(0), Hk = q.size(1), dk = q.size(2);
  const auto Hv = v.size(1), dv = v.size(2);
  TORCH_CHECK(dk <= 256 && dv <= 256, "gdr_step: head dims exceed 256");
  auto out = at::empty({B, Hv, dv}, v.options());

  uint32_t dkb = (uint32_t)dk, dvb = (uint32_t)dv;
  uint32_t Hkb = (uint32_t)Hk, Hvb = (uint32_t)Hv;

  MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pipeline("gdr_step", q)];
    setBuffer(enc, q, 0);
    setBuffer(enc, k, 1);
    setBuffer(enc, v, 2);
    setBuffer(enc, g, 3);
    setBuffer(enc, beta, 4);
    setBuffer(enc, state, 5);
    setBuffer(enc, out, 6);
    [enc setBytes:&dkb length:sizeof(dkb) atIndex:7];
    [enc setBytes:&dvb length:sizeof(dvb) atIndex:8];
    [enc setBytes:&Hkb length:sizeof(Hkb) atIndex:9];
    [enc setBytes:&Hvb length:sizeof(Hvb) atIndex:10];
    NSUInteger threads = std::max<NSUInteger>(dk, dv);
    MTLSize grid = MTLSizeMake(threads, (NSUInteger)Hv, (NSUInteger)B);
    MTLSize tg = MTLSizeMake(threads, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stream->endKernelCoalescing();
  });
  return out;
}

// Sequential scan for prefill/inference on pre-normalized fp32 qn,kn.
// Mutates `state` in place, returns out [B,L,Hv,dv].
at::Tensor gdr_seq(const at::Tensor& qn,
                   const at::Tensor& kn,
                   const at::Tensor& v,
                   const at::Tensor& g,
                   const at::Tensor& beta,
                   const at::Tensor& state) {
  const auto B = qn.size(0), L = qn.size(1), Hk = qn.size(2), dk = qn.size(3);
  const auto Hv = v.size(2), dv = v.size(3);
  TORCH_CHECK(dk <= 128, "gdr_seq: dk exceeds threadgroup row capacity");
  auto out = at::empty({B, L, Hv, dv}, v.options());

  uint32_t Lb = (uint32_t)L, dkb = (uint32_t)dk, dvb = (uint32_t)dv;
  uint32_t Hkb = (uint32_t)Hk, Hvb = (uint32_t)Hv;

  MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pipeline("gdr_seq", v)];
    setBuffer(enc, qn, 0);
    setBuffer(enc, kn, 1);
    setBuffer(enc, v, 2);
    setBuffer(enc, g, 3);
    setBuffer(enc, beta, 4);
    setBuffer(enc, state, 5);
    setBuffer(enc, out, 6);
    [enc setBytes:&Lb length:sizeof(Lb) atIndex:7];
    [enc setBytes:&dkb length:sizeof(dkb) atIndex:8];
    [enc setBytes:&dvb length:sizeof(dvb) atIndex:9];
    [enc setBytes:&Hkb length:sizeof(Hkb) atIndex:10];
    [enc setBytes:&Hvb length:sizeof(Hvb) atIndex:11];
    NSUInteger nblk = (NSUInteger)((dv + 31) / 32);
    MTLSize grid = MTLSizeMake(32, (NSUInteger)Hv * nblk, (NSUInteger)B);
    MTLSize tg = MTLSizeMake(32, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stream->endKernelCoalescing();
  });
  return out;
}

}  // namespace meowtrix::metal

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv1d_fwd", &meowtrix::metal::conv1d_fwd, "fused causal conv1d fwd (MPS)");
  m.def("conv1d_bwd_dx", &meowtrix::metal::conv1d_bwd_dx, "fused causal conv1d dx (MPS)");
  m.def("conv1d_update", &meowtrix::metal::conv1d_update, "fused causal conv1d update (MPS)");
  m.def("gdr_norm", &meowtrix::metal::gdr_norm, "gated delta rule qk norm (MPS)");
  m.def("gdr_step", &meowtrix::metal::gdr_step, "gated delta rule decode step (MPS)");
  m.def("gdr_seq", &meowtrix::metal::gdr_seq, "gated delta rule sequential scan (MPS)");
}
