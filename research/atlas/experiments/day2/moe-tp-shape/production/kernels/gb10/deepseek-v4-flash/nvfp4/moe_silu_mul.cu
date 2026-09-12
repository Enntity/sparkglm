// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE element-wise SiLU activation + multiply, WITH DeepSeek-V4's
// configured SwiGLU clamp. Shadows `common/moe_silu_mul.cu` for this model
// only.
//
//   output[i] = silu(clamp(gate[i])) * clamp(up[i])
//
// WHY THIS FILE EXISTS. The clamp used to live in `common/`, added there by the
// DeepSeek-V4 port (#186) as six lines on a kernel that was already shared. But
// `moe_silu_mul` is not a DeepSeek kernel: it is the SiLU activation for every
// dense model's decode and K-verify FFN (`DenseFfnLayer::act_mul`), for every
// MoE model's grouped prefill (`MoeLayer::moe_act_mul`), and for the MTP and
// DFlash draft heads. So one checkpoint's config value was applied to about
// twenty checkpoints, none of which declare a SwiGLU limit and none of whose
// reference implementations clamp at all (`Qwen3_5MLP.forward` and
// `Qwen3_5MoeExperts.forward` are a bare `act_fn(gate) * up`).
//
// `swiglu_limit` is genuinely model-specific — DeepSeek-V4 sets 10.0, GPT-OSS
// sets 7.0, Qwen sets nothing — so the value belongs with the model. Keeping it
// in a shadow is the smallest correct home for it until the limit is threaded
// from config into a kernel argument, which is what the checkpoint actually
// asks for and what Step-3.7's per-LAYER `swiglu_limits` array will require.
//
// The reference is `inference/model.py` in `deepseek-ai/DeepSeek-V4-Flash`:
//
//     if self.swiglu_limit > 0:
//         up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
//         gate = torch.clamp(gate, max=self.swiglu_limit)
//     x = F.silu(gate) * up
//
// Note the asymmetry — gate is bounded ABOVE only, up is bounded on both sides.
// The math below is byte-for-byte what `common/` computed before the move, so
// DeepSeek-V4's numerics are unchanged by relocating it.
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void moe_silu_mul(
    const __nv_bfloat16* __restrict__ gate,   // [total_expanded, inter_size]
    const __nv_bfloat16* __restrict__ up,     // [total_expanded, inter_size]
    __nv_bfloat16* __restrict__ output,        // [total_expanded, inter_size]
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    // swiglu_limit = 10.0, from the checkpoint's config.json. Hardcoded because
    // no Rust parser reads the field yet; `ModelConfig` has no home for it.
    const float SWIGLU_LIMIT = 10.0f;
    g = fminf(g, SWIGLU_LIMIT);
    u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}

// Fused DeepSeek-clamped SiLU·mul + standard token-major NVFP4 quantization.
//
// This is the native-FP4 analogue of common/moe_silu_mul.cu's fused FP8
// helper.  It emits exactly the packed `[M,K/2]` E2M1 + `[M,K/16]` E4M3
// layout consumed by moe_w4a4_grouped_gemm_prequant_t_k64.  The product is
// explicitly rounded through BF16 before the group maximum and FP4 encode,
// matching the old moe_silu_mul -> quantize_bf16_to_nvfp4 pair.
//
// `out_bf16` is nullable.  Base-model serving passes null and avoids the full
// BF16 intermediate; MoE down-projection LoRA may request it because its fold
// consumes the post-SiLU activations.

__device__ __forceinline__ unsigned char silu_nvfp4_float_to_e4m3(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int sign = (bits >> 31) & 1;
    if ((bits & 0x7FFFFFFF) == 0) return (unsigned char)(sign << 7);

    float absv = fabsf(v);
    if (absv > 448.0f) absv = 448.0f;
    bits = __float_as_uint(absv);
    int f32_exp = (int)((bits >> 23) & 0xFF) - 127;
    unsigned int f32_man = bits & 0x7FFFFF;

    if (f32_exp < -9) {
        return (unsigned char)(sign << 7);
    }
    if (f32_exp < -6) {
        int man = (int)(absv * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 0) man = 0;
        return (unsigned char)((sign << 7) | man);
    }

    int fp8_exp = f32_exp + 7;
    unsigned int fp8_man;
    if (fp8_exp < 1) fp8_exp = 1;
    if (fp8_exp > 15) {
        fp8_exp = 15;
        fp8_man = 6;
    } else {
        fp8_man = (f32_man + (1 << 19)) >> 20;
        if (fp8_man > 7) {
            fp8_man = 0;
            fp8_exp++;
            if (fp8_exp > 15) {
                fp8_exp = 15;
                fp8_man = 6;
            }
        }
    }
    return (unsigned char)((sign << 7) | (fp8_exp << 3) | fp8_man);
}

__device__ __forceinline__ unsigned int silu_nvfp4_quantize_e2m1(float v) {
    const float absv = fabsf(v);
    const unsigned int sign = (v < 0.0f) ? 8u : 0u;
    unsigned int idx;
    if      (absv <= 0.25f) idx = 0;
    else if (absv <= 0.75f) idx = 1;
    else if (absv <= 1.25f) idx = 2;
    else if (absv <= 1.75f) idx = 3;
    else if (absv <= 2.5f)  idx = 4;
    else if (absv <= 3.5f)  idx = 5;
    else if (absv <= 5.0f)  idx = 6;
    else                    idx = 7;
    return sign | idx;
}

extern "C" __global__ void silu_mul_quant_nvfp4(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    __nv_bfloat16* __restrict__ out_bf16,
    unsigned int M,
    unsigned int K
) {
    const unsigned int row = blockIdx.x;
    if (row >= M) return;

    const __nv_bfloat16* grow = gate + (unsigned long long)row * K;
    const __nv_bfloat16* urow = up + (unsigned long long)row * K;
    unsigned char* prow = packed_out + (unsigned long long)row * (K / 2);
    unsigned char* srow = scale_out + (unsigned long long)row * (K / 16);
    __nv_bfloat16* brow = out_bf16 ? out_bf16 + (unsigned long long)row * K : nullptr;
    const unsigned int groups = K / 16;
    const float SWIGLU_LIMIT = 10.0f;

    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        float vals[16];
        float group_max = 0.0f;
        const unsigned int base = group * 16;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            float g = __bfloat162float(grow[base + i]);
            float u = __bfloat162float(urow[base + i]);
            g = fminf(g, SWIGLU_LIMIT);
            u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
            const float sigmoid_g = 1.0f / (1.0f + __expf(-g));
            const __nv_bfloat16 r16 = __float2bfloat16(g * sigmoid_g * u);
            if (brow) brow[base + i] = r16;
            const float r = __bfloat162float(r16);
            vals[i] = r;
            group_max = fmaxf(group_max, fabsf(r));
        }

        const unsigned char fp8 = silu_nvfp4_float_to_e4m3(group_max / 6.0f);
        srow[group] = fp8;
        const unsigned int exp = (fp8 >> 3) & 0xF;
        const unsigned int man = fp8 & 0x7;
        float decoded;
        if (exp == 0) {
            decoded = (float)man * 0.001953125f;
        } else if (exp == 15 && man == 7) {
            decoded = 0.0f;
        } else {
            decoded = __uint_as_float((exp + 120u) << 23 | (man << 20));
        }
        const float inv = decoded > 0.0f ? 1.0f / decoded : 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i += 2) {
            const unsigned int q0 = silu_nvfp4_quantize_e2m1(vals[i] * inv);
            const unsigned int q1 = silu_nvfp4_quantize_e2m1(vals[i + 1] * inv);
            prow[group * 8 + i / 2] = (unsigned char)((q1 << 4) | q0);
        }
    }
}
