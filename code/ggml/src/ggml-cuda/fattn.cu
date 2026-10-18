#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#include <cmath>
#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>

template <int DKQ, int DV, int ncols2, ggml_type type_KV = GGML_TYPE_F16>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * V = dst->src[2];

    // ada: upstream #189 ("avoid short-query MMA FA crash with quantized V") says a quantized V cache with a
    // short query (2 < Q <= 4, i.e. a speculative verify batch) must not use the in-place q4_0/q8_0 kernel, and
    // sends it to mma_f16_case<.., 64/ncols2, ncols2> instead. That call drops type_KV, so launch_fattn sets
    // need_f16_KV: the whole K/V cache is dequantised into a scratch buffer on every pass, and the Q tile is 64
    // wide for 3 or 4 real rows.
    //
    // In this tree the guard is redundant and it takes the in-place path away exactly where it matters most: the
    // verify step, the hottest flash attention call of an MTP decode. ggml_cuda_flash_attn_ext_mma_f16 only enters
    // a quantized type_KV instantiation after ggml_cuda_fattn_mma_kv_native_supported() passed - matching K/V
    // types, q4_0/q8_0, D = 128/256, word-aligned rows - and routes everything else to the F16 instance anyway,
    // so there is nothing left for #189 to protect here.
    // Measured 2026-10-10, same engine and prompt, only this switch flipped: corrupt output at 24.9 tok/s with
    // #189, clean output at 27.1 tok/s without it.
    //
    // 0 (default) = ignore #189, keep the in-place kernel. 1 = upstream behaviour, for A/B only.
    static const int shortq_mma = [] {
        const char * e = getenv("GGML_CUDA_FA_SHORTQ_MMA");
        return e ? atoi(e) : 0;
    }();
    if (shortq_mma != 0 && turing_mma_available(cc) && ggml_is_quantized(V->type) && Q->ne[1] > 2 && Q->ne[1] <= 4) {
        // upstream #355: keep type_KV here too, otherwise the F16 instantiation converts K/V into scratch that
        // ggml_cuda_flash_attn_ext_get_alloc_size does not reserve on the in-place quantized path.
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2, type_KV>(ctx, dst);
        return;
    }

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2, type_KV>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2, type_KV>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2, type_KV>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2, type_KV>(ctx, dst);
}

template <int DKQ, int DV, ggml_type type_KV = GGML_TYPE_F16>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8, type_KV>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4, type_KV>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2, type_KV>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1, type_KV>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8, type_KV>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4, type_KV>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2, type_KV>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1, type_KV>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            if (ggml_cuda_fattn_mma_kv_native_supported(dst)) {
                if (K->type == GGML_TYPE_Q4_0) {
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128, GGML_TYPE_Q4_0>(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128, GGML_TYPE_Q8_0>(ctx, dst);
                }
                break;
            }
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            if (ggml_cuda_fattn_mma_kv_native_supported(dst)) {
                if (K->type == GGML_TYPE_Q4_0) {
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256, GGML_TYPE_Q4_0>(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256, GGML_TYPE_Q8_0>(ctx, dst);
                }
                break;
            }
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        // Quantized-KV decode with wide GQA (up to 8 queries: single-token decode and speculative verify):
        // the vector kernel runs one block per Q head, so every K/V row is fetched gqa_ratio times (6x for
        // Qwen3.5/Bonsai 2: 24 Q heads over 4 KV heads). The MMA kernel packs the GQA heads of one KV head
        // into one tile and reads q4_0/q8_0 K/V in place once. Same rule the F16 path already uses.
        // RTX 4070, Bonsai 2 27B, q8_0 K/V, MTP draft: 4k 78.1 -> 86.9 tok/s, 16k 77.7 -> 111.6, 32k 54.5 -> 103.6.
        // GGML_CUDA_FA_MMA_DECODE_MIN_KV = shortest KV length that takes this route (default 256, 0 = never).
        {
            static const int min_kv = [] {
                const char * e = getenv("GGML_CUDA_FA_MMA_DECODE_MIN_KV");
                return e ? atoi(e) : 256;
            }();
            if (min_kv > 0 && ggml_is_quantized(K->type) && K->type == V->type &&
                    ggml_cuda_fattn_mma_kv_native_supported(dst) &&
                    gqa_opt_applies && gqa_ratio > 4 && Q->ne[1] <= 8 && Q->ne[3] == 1 && K->ne[1] >= min_kv) {
                return BEST_FATTN_KERNEL_MMA_F16;
            }
        }
        if (can_use_vector_kernel) {
            // batch-invariant mode: the same (vector) kernel for 1 to 8 queries, so a token verified in a
            // speculative batch attends with the same arithmetic as a token decoded alone
            if (ggml_cuda_batch_invariant() && Q->ne[1] <= 8 && Q->ne[3] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_MMA_F16:
            if (ggml_cuda_fattn_mma_kv_native_supported(dst) || ggml_cuda_fattn_prefill_f16(dst)) {
                // In-place quantized K/V kernel, or f16 copies taken from the pool: nothing to reserve beyond dst.
                break;
            }
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_TILE:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

// ggml-cuda.cu: host-tail staging of tiered buffers (ggml_backend_cuda_tier_buffer_type), and its prefetch
void * ggml_cuda_tier_stage(const void * ptr, size_t nbytes, cudaStream_t stream);
bool   ggml_cuda_tier_fa_begin(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, void ** K_alias, void ** V_alias);
void   ggml_cuda_tier_fa_end(ggml_backend_cuda_context & ctx);

// ---------------------------------------------------------------------------------------------------------------------
// GGML_CUDA_FA_SPARSE=B[,P[,R]] (experiment: quality only). Attention ops whose K reaches the host tail of a tiered
// buffer keep every VRAM cell; in the host tail each query keeps the R cells before its own position and the B/P pages
// of P cells with the highest bound sum_d max(q_d*maxK_d, q_d*minK_d), the max over the layer's heads (Quest); the
// rest of the host tail is masked out for that query. It still reads every host row and syncs the stream once per op,
// so it is slow: it measures what reading only part of the host tail would cost in quality. Needs an f16 mask
// (LLAMA_ARG_KQ_MASK_PACKED=0) and CUDA graphs off (GGML_CUDA_DISABLE_GRAPHS=1). Unset or B = 0: off.
int64_t ggml_cuda_tier_first_host_cell(const ggml_tensor * t);
void *  ggml_cuda_tier_stage_pages(const void * ptr, size_t nbytes, size_t row_bytes, int64_t first_cell, int page, int n_pages,
            const uint8_t * keep, cudaStream_t stream);
const void * ggml_cuda_tier_unalias(const void * p);

struct fa_sparse_cfg {
    int64_t budget = 0, page = 64, recent = 1024;
    int     maxq = 64;   // GGML_CUDA_FA_SPARSE_MAXQ: fast mode takes ops of up to this many queries (decode, draft verify)
    int64_t sinks = -1;  // GGML_CUDA_FA_SPARSE_SINKS=N: select pages from cell N on, the VRAM part too (-1: host tail only)
    int64_t min_host = 0;   // GGML_CUDA_FA_SPARSE_MIN_HOST=N: dense while the host tail has fewer than N cells (sparse costs
                            // ~16 % on a short tail on 8 GB: 64k, ~18k host cells)
    bool    fast = false;   // GGML_CUDA_FA_SPARSE_FAST=1: decode-sized ops only (<= 8 queries), page bounds cached
    bool    check   = false;   // GGML_CUDA_FA_SPARSE_CHECK=1 (debug): fast mode also recomputes every page, logs stale ones
    bool    notrack = false;   // GGML_CUDA_FA_SPARSE_NOTRACK=1 (debug): writes do not invalidate cached bounds
    float   frac    = 0.0625f; // GGML_CUDA_FA_SPARSE_FRAC=f (default 1/16, 0: B only): keep at least f of the host tail (B is
                               // the floor), so the budget grows with depth (245k at B 8192 = 3.9 % of the tail lost an exact
                               // copy; 12288 = 5.8 % kept it)
};

static const fa_sparse_cfg & fa_sparse() {
    static const fa_sparse_cfg c = [] {
        fa_sparse_cfg r;
        if (const char * e = getenv("GGML_CUDA_FA_SPARSE")) {
            long long b = 0, p = 64, q = 1024;
            sscanf(e, "%lld,%lld,%lld", &b, &p, &q);
            r.budget = b;
            r.page   = p > 0 ? p : 64;
            r.recent = q >= 0 ? q : 0;
        }
        const char * f = getenv("GGML_CUDA_FA_SPARSE_FAST");
        r.fast = f && atoi(f) != 0;
        const char * c = getenv("GGML_CUDA_FA_SPARSE_CHECK");
        r.check = c && atoi(c) != 0;
        const char * n = getenv("GGML_CUDA_FA_SPARSE_NOTRACK");
        r.notrack = n && atoi(n) != 0;
        const char * mq = getenv("GGML_CUDA_FA_SPARSE_MAXQ");
        r.maxq = mq ? std::max(1, atoi(mq)) : 64;
        const char * fr = getenv("GGML_CUDA_FA_SPARSE_FRAC");
        r.frac = fr ? std::max(0.0f, std::min(1.0f, (float) atof(fr))) : 0.0625f;
        const char * sk = getenv("GGML_CUDA_FA_SPARSE_SINKS");
        r.sinks = sk ? std::max(0, atoi(sk)) : -1;
        const char * mh = getenv("GGML_CUDA_FA_SPARSE_MIN_HOST");
        r.min_host = mh ? std::max<long long>(0, atoll(mh)) : 0;
        return r;
    }();
    return c;
}

// pages a sparse op keeps from a host tail of n_sel cells: B, or f of the tail when that is more
static int64_t fa_sparse_keep_pages(const fa_sparse_cfg & cfg, int64_t n_sel) {
    const int64_t cells = std::max<int64_t>(cfg.budget, (int64_t) ceilf(cfg.frac*(float) n_sel));
    return std::max<int64_t>(1, cells/cfg.page);
}

// Fast mode: page-bound tables kept across ops, one per K cache tensor (key: the K data address the attention views start
// at). A write into a tracked K tensor lowers `valid` to the first host cell it can touch, and the next attention op
// recomputes the bounds from that page on: set_rows of new cells (exact rows, from the index tensor), any other op that
// writes into it, and buffer writes (state loads, copies, clears, frees) from their first byte on.
struct fa_sparse_table {
    // mn and sc point into the one allocation that starts at mx (fa_sparse_bnd_bytes)
    int8_t * mx = nullptr; int8_t * mn = nullptr; half * sc = nullptr; int cap = 0; int64_t first = -1; int64_t valid = 0;
    const char * end = nullptr; size_t row = 0;
};
static std::mutex                                fa_sparse_mtx;
static std::map<const char *, fa_sparse_table>   fa_sparse_tables;
static std::atomic<uintptr_t>                    fa_sparse_lo{UINTPTR_MAX}, fa_sparse_hi{0};

// the bytes [p, p + n) may have changed: in each table that holds some of them, the host cells from the first written
// one on. rows: the rows a set_rows wrote, relative to p, when its rows (row_bytes) match the table's (else every row of
// [p, p + n) counts as written). Writes that end before the host tail change nothing. Call with fa_sparse_mtx held.
static void fa_sparse_dirty_locked(const char * p, size_t n, const std::vector<int64_t> * rows = nullptr, size_t row_bytes = 0) {
    for (auto & [base, t] : fa_sparse_tables) {
        if (n == 0 || p + n <= base || p >= t.end || t.row == 0) {
            continue;
        }
        const int64_t r0 = p <= base ? 0 : (int64_t) ((size_t) (p - base)/t.row);
        int64_t lo = INT64_MAX;
        if (rows && p >= base && row_bytes == t.row) {
            for (const int64_t r : *rows) {
                if (r0 + r >= t.first) {
                    lo = std::min(lo, r0 + r);
                }
            }
        } else {
            const int64_t r1 = (int64_t) ((size_t) (std::min(p + n, t.end) - 1 - base)/t.row);
            if (r1 >= t.first) {
                lo = std::max(r0, t.first);
            }
        }
        if (lo != INT64_MAX) {
            t.valid = std::min<int64_t>(t.valid, lo - t.first);
        }
    }
}

static bool fa_sparse_maybe_tracked(const void * p, size_t n) {
    const uintptr_t a = (uintptr_t) p;
    return a < fa_sparse_hi.load(std::memory_order_relaxed) && a + n > fa_sparse_lo.load(std::memory_order_relaxed);
}

// graph setup (ggml-cuda.cu): an attention op the fast path takes stays out of the tier prefetch, which would copy the
// whole host tail ahead of it (same shape checks as ggml_cuda_fa_sparse_mask)
bool ggml_cuda_fa_sparse_eligible(const ggml_tensor * fa) {
    const fa_sparse_cfg & cfg = fa_sparse();
    const ggml_tensor * Q = fa->src[0], * K = fa->src[1], * M = fa->src[3];
    if (cfg.budget <= 0 || !cfg.fast || !K || !M || M->type != GGML_TYPE_F16 || Q->ne[1] > cfg.maxq) {
        return false;
    }
    // the selected region: the host tail, or with GGML_CUDA_FA_SPARSE_SINKS every cell from the sinks on (the op must
    // still reach the host tail, so short contexts stay dense)
    const int64_t first_host = ggml_cuda_tier_first_host_cell(K);
    const int64_t first = first_host >= 0 && cfg.sinks >= 0 ? std::min<int64_t>(cfg.sinks, first_host) : first_host;
    // MIN_HOST as in ggml_cuda_fa_sparse_mask: below it the op is the dense op, with the tier prefetch (an op left out
    // of the prefetch but not sparse read the whole host tail in place: 7.8 -> 22.8 ms per graph at 64k on 8 GB)
    if (first < 0 || first >= K->ne[1] || K->ne[1] - first_host < cfg.min_host) {
        return false;
    }
    const int64_t n_pages = (K->ne[1] - first + cfg.page - 1)/cfg.page;
    return n_pages > fa_sparse_keep_pages(cfg, K->ne[1] - first) && n_pages <= 12288;
}

// buffer-level writes (ggml-cuda.cu: set_tensor, memset, copies, clear, free): everything from p on
void ggml_cuda_fa_sparse_note_bytes(const void * p, size_t n) {
    if (!fa_sparse_maybe_tracked(p, n) || fa_sparse().notrack) {
        return;
    }
    std::lock_guard<std::mutex> lock(fa_sparse_mtx);
    fa_sparse_dirty_locked((const char *) p, n);
}

// after each op (ggml_cuda_compute_forward): an op whose result lands in a tracked K tensor
// host copies of small index tensors written from the host (buffer set_tensor, backend set_tensor_async), by device
// address: set_rows reads its rows from these without a device-to-host copy. The scheduler copies llama's cell
// indices to the device this way. Any other write to the same address (a non-index set, or a device op that writes
// an index tensor there) drops the entry, so a stale copy is never used.
static std::unordered_map<const void *, std::vector<int64_t>> fa_sparse_inputs;
static std::atomic<bool> fa_sparse_inputs_any{false};

void ggml_cuda_fa_sparse_note_input(const ggml_tensor * t, const void * data, size_t offset, size_t size) {
    if (fa_sparse().budget <= 0 || !fa_sparse().fast || !t->data) {
        return;
    }
    const bool index = (t->type == GGML_TYPE_I64 || t->type == GGML_TYPE_I32) && offset == 0 && size == ggml_nbytes(t) &&
        size <= (1u << 20);
    std::lock_guard<std::mutex> lock(fa_sparse_mtx);
    if (!index) {
        if (fa_sparse_inputs_any.load(std::memory_order_relaxed)) {
            fa_sparse_inputs.erase((const char *) t->data + offset);
        }
        return;
    }
    const int64_t n = ggml_nelements(t);
    std::vector<int64_t> & v = fa_sparse_inputs[t->data];
    v.resize((size_t) n);
    for (int64_t k = 0; k < n; ++k) {
        v[k] = t->type == GGML_TYPE_I64 ? ((const int64_t *) data)[k] : ((const int32_t *) data)[k];
    }
    fa_sparse_inputs_any.store(true, std::memory_order_relaxed);
}

void ggml_cuda_fa_sparse_note_op(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (fa_sparse_inputs_any.load(std::memory_order_relaxed) && dst->data &&
            (dst->type == GGML_TYPE_I64 || dst->type == GGML_TYPE_I32)) {
        std::lock_guard<std::mutex> lock(fa_sparse_mtx);
        fa_sparse_inputs.erase(dst->data);   // a device op wrote an index tensor here: the host copy is stale
    }
    // a set_rows redirected into the staging alias (the prefetch pipeline) writes the same cells as the cache itself
    const void * d = dst->data ? ggml_cuda_tier_unalias(dst->data) : nullptr;
    if (!d || !fa_sparse_maybe_tracked(d, ggml_nbytes(dst)) || fa_sparse().notrack) {
        return;
    }
    const char * p = (const char *) d;
    {
        // only writes into a tracked K tensor matter (the address range also covers the V tensors between them)
        std::lock_guard<std::mutex> lock(fa_sparse_mtx);
        bool hit = false;
        for (auto & [base, t] : fa_sparse_tables) {
            hit = hit || (p + ggml_nbytes(dst) > base && p < t.end);
        }
        if (!hit) {
            return;
        }
    }
    const ggml_tensor * idx = dst->src[1];
    std::vector<int64_t> rows;
    if (dst->op == GGML_OP_SET_ROWS && dst->ne[2] == 1 && dst->ne[3] == 1 && idx && ggml_is_contiguous(idx) &&
            (idx->type == GGML_TYPE_I64 || idx->type == GGML_TYPE_I32) && ggml_nelements(idx) > 0) {
        // exact: the rows this set_rows wrote, from the host copy of the index input when there is one, else one
        // small device-to-host copy and a sync
        const int64_t ni = ggml_nelements(idx);
        {
            std::lock_guard<std::mutex> lock(fa_sparse_mtx);
            auto it = fa_sparse_inputs.find(idx->data);
            if (it != fa_sparse_inputs.end() && (int64_t) it->second.size() == ni) {
                rows = it->second;
            }
        }
        if (rows.empty()) {
            static bool warned = false;
            if (!warned) {
                GGML_LOG_WARN("%s: GGML_CUDA_FA_SPARSE_FAST: set_rows indices were not written from the host: one sync per K write\n", __func__);
                warned = true;
            }
            rows.resize((size_t) ni);
            if (idx->type == GGML_TYPE_I64) {
                CUDA_CHECK(cudaMemcpyAsync(rows.data(), idx->data, ni*sizeof(int64_t), cudaMemcpyDefault, ctx.stream()));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            } else {
                std::vector<int32_t> h32((size_t) ni);
                CUDA_CHECK(cudaMemcpyAsync(h32.data(), idx->data, ni*sizeof(int32_t), cudaMemcpyDefault, ctx.stream()));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
                std::copy(h32.begin(), h32.end(), rows.begin());
            }
        }
    }
    std::lock_guard<std::mutex> lock(fa_sparse_mtx);
    fa_sparse_dirty_locked(p, ggml_nbytes(dst), rows.empty() ? nullptr : &rows, (size_t) dst->nb[1]);
}

// debug (GGML_CUDA_FA_SPARSE_CHECK): pages whose cached bounds differ from freshly computed ones
static __global__ void fa_sparse_diff(const int8_t * __restrict__ a_mx, const int8_t * __restrict__ a_mn,
        const half * __restrict__ a_sc, const int a_stride, const int8_t * __restrict__ b_mx, const int8_t * __restrict__ b_mn,
        const half * __restrict__ b_sc, const int b_stride, const int D, int * __restrict__ bad) {
    const int p = blockIdx.x, h = blockIdx.y;
    bool diff = __half2float(a_sc[(int64_t) h*a_stride + p]) != __half2float(b_sc[(int64_t) h*b_stride + p]);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const int64_t ia = ((int64_t) h*a_stride + p)*D + d, ib = ((int64_t) h*b_stride + p)*D + d;
        diff = diff || a_mx[ia] != b_mx[ib] || a_mn[ia] != b_mn[ib];
    }
    if (__syncthreads_or(diff) && threadIdx.x == 0) {
        atomicAdd(bad, 1);
        atomicMin(bad + 1, p);
    }
}

// per page and KV head: max and min over the page's cells of each K dimension (K as contiguous f16 [D, n_cells, H_kv]),
// stored as int8 with one f16 scale per page and head (sc >= max |bound| / 127, rounded up), rounded OUTWARD: max up,
// min down. The page score from them is still an upper bound of every q.k in the page (looser than exact bounds), at
// 2*D + 2 bytes per page and head: 514 B per host cell at page 64 for this model, half of f16 bounds.
// Exactness of the rounding check: |q| <= 127 and sc an f16 value, so q*sc is exact in f32.
static __global__ void fa_sparse_bounds(const half * __restrict__ K, int8_t * __restrict__ mx, int8_t * __restrict__ mn,
        half * __restrict__ sc, const int D, const int n_cells, const int page, const int p0, const int stride_pages) {
    extern __shared__ float fa_sparse_bounds_sh[];   // [2*D]: max, then min
    __shared__ float red[32];
    const int p = blockIdx.x, h = blockIdx.y;    // p: page within K; written at page p0 + p of a table stride_pages wide
    const int c0 = p*page, c1 = min(n_cells, c0 + page);
    float amax = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = -INFINITY, b = INFINITY;
        for (int c = c0; c < c1; ++c) {
            const float v = __half2float(K[((int64_t) h*n_cells + c)*D + d]);
            a = fmaxf(a, v);
            b = fminf(b, v);
        }
        fa_sparse_bounds_sh[d]     = a;
        fa_sparse_bounds_sh[D + d] = b;
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
    }
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    if ((threadIdx.x & 31) == 0) {
        red[threadIdx.x >> 5] = amax;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            t = fmaxf(t, __shfl_xor_sync(0xffffffff, t, o));
        }
        if (threadIdx.x == 0) {
            red[0] = t;
        }
    }
    __syncthreads();
    const half  s16 = __float2half_ru(red[0] / 127.0f);
    float s = __half2float(s16);
    while (127.0f*s < red[0]) {   // the division can round down: keep 127*s >= every |bound|
        s = __half2float(__float2half_ru(s*1.0009765625f));
    }
    const int64_t pg = (int64_t) h*stride_pages + p0 + p;
    if (threadIdx.x == 0) {
        sc[pg] = __float2half(s);
    }
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float a = fa_sparse_bounds_sh[d], b = fa_sparse_bounds_sh[D + d];
        int qa = 0, qb = 0;
        if (s > 0.0f) {
            qa = min(127, (int) ceilf(a/s));
            while (qa < 127 && (float) qa*s < a) { ++qa; }
            qb = max(-127, (int) floorf(b/s));
            while (qb > -127 && (float) qb*s > b) { --qb; }
        }
        mx[pg*D + d] = (int8_t) qa;
        mn[pg*D + d] = (int8_t) qb;
    }
}

// bytes of one bound table: int8 max and min [H_kv][pages][D], then the f16 scales [H_kv][pages]
static size_t fa_sparse_bnd_bytes(int H_kv, int pages, int D) {
    return (size_t) H_kv*pages*D*2 + (size_t) H_kv*pages*sizeof(half);
}

// ub[j][p] = max over Q heads of sum_d max(q*mx, q*mn), with the KV head of each Q head (GQA)
static __global__ void fa_sparse_scores(const float * __restrict__ Q, const int8_t * __restrict__ mx, const int8_t * __restrict__ mn,
        const half * __restrict__ sc,
        float * __restrict__ ub, const int D, const int n_pages, const int n_head, const int gqa,
        const int64_t q_s1, const int64_t q_s2, const int stride_pages) {
    const int p = blockIdx.x, j = blockIdx.y;
    __shared__ float red[32];
    float best = -INFINITY;
    for (int hq = 0; hq < n_head; ++hq) {
        const int h = hq / gqa;
        const float * q = Q + hq*q_s2 + (int64_t) j*q_s1;
        const int8_t * a = mx + ((int64_t) h*stride_pages + p)*D;
        const int8_t * b = mn + ((int64_t) h*stride_pages + p)*D;
        const float    f = __half2float(sc[(int64_t) h*stride_pages + p]);   // >= 0, so it factors out of the max
        float s = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            s += fmaxf(q[d]*(float) a[d], q[d]*(float) b[d]);
        }
        s *= f;
        for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_xor_sync(0xffffffff, s, o);
        }
        if ((threadIdx.x & 31) == 0) {
            red[threadIdx.x >> 5] = s;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.0f;
            for (int w = 0; w < (int) (blockDim.x >> 5); ++w) {
                t += red[w];
            }
            best = fmaxf(best, t);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ub[(int64_t) j*n_pages + p] = best;
    }
}

// last[j]: the last host-tail column (relative to first) that row j of the mask does not mask out, -1 if none
static __global__ void fa_sparse_last(const half * __restrict__ mask, int * __restrict__ last, const int64_t first,
        const int64_t n_host, const int64_t m_s1) {
    const int j = blockIdx.x;
    int best = -1;
    for (int64_t c = threadIdx.x; c < n_host; c += blockDim.x) {
        if (!isinf(__half2float(mask[(int64_t) j*m_s1 + first + c]))) {
            best = max(best, (int) c);
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        best = max(best, __shfl_xor_sync(0xffffffff, best, o));
    }
    __shared__ int red[32];
    if ((threadIdx.x & 31) == 0) {
        red[threadIdx.x >> 5] = best;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w) {
            best = max(best, red[w]);
        }
        last[j] = max(best, red[0]);
    }
}

// masks out the host-tail columns of row j whose page is not kept (keep[j][p] == 0)
// per query j: keep the pages from the one holding cell last+1-recent up to the one holding its last visible cell, and of
// the pages before those the keep_pages with the highest bound (ties: lower page first); drop the rest. One block per
// query, the scores in shared memory, the rank of each candidate by counting the candidates ahead of it.
static __global__ void fa_sparse_select(const float * __restrict__ ub, const int * __restrict__ last_cell,
        uint8_t * __restrict__ keep, const int n_pages, const int page, const int recent, const int keep_pages) {
    extern __shared__ float s_ub[];
    const int j    = blockIdx.x;
    const int last = last_cell[j];
    uint8_t * k = keep + (int64_t) j*n_pages;
    const int last_page   = last < 0 ? -1 : last/page;
    const int recent_page = last < 0 ? 0 : max(0, last + 1 - recent)/page;
    for (int p = threadIdx.x; p < recent_page; p += blockDim.x) {
        s_ub[p] = ub[(int64_t) j*n_pages + p];
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n_pages; p += blockDim.x) {
        uint8_t v = 0;
        if (p <= last_page) {
            if (p >= recent_page) {
                v = 1;
            } else {
                const float x = s_ub[p];
                int rank = 0;
                for (int q = 0; q < recent_page && rank < keep_pages; ++q) {
                    const float y = s_ub[q];
                    rank += (y > x) || (y == x && q < p);
                }
                v = rank < keep_pages;
            }
        }
        k[p] = v;
    }
}

static __global__ void fa_sparse_apply(half * __restrict__ mask, const uint8_t * __restrict__ keep, const int64_t first,
        const int64_t n_host, const int page, const int n_pages, const int64_t m_s1) {
    const int j = blockIdx.y;
    const int64_t c = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n_host) {
        return;
    }
    if (!keep[(int64_t) j*n_pages + c/page]) {
        mask[(int64_t) j*m_s1 + first + c] = __float2half(-INFINITY);
    }
}

// returns true when the op runs with a sparse copy of its mask in *mask_out (backed by mask_buf)
// the host-tail pages that at least one query of the op keeps (for staging only those, ggml_cuda_tier_stage_pages)
struct fa_sparse_pages {
    int64_t         first   = -1;   // first cell of the selected region (page 0)
    int64_t         first_host = -1;   // first host cell of K (V must match for page staging)
    int             page    = 0;
    int             n_pages = 0;
    const uint8_t * keep    = nullptr;   // [n_pages] on the device
};

static __global__ void fa_sparse_union(const uint8_t * __restrict__ keep, uint8_t * __restrict__ any, const int n_q, const int n_pages) {
    const int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n_pages) {
        return;
    }
    uint8_t v = 0;
    for (int j = 0; j < n_q; ++j) {
        v |= keep[(int64_t) j*n_pages + p];
    }
    any[p] = v;
}

static bool ggml_cuda_fa_sparse_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
        ggml_cuda_pool_alloc<half> & mask_buf, ggml_tensor * mask_out,
        ggml_cuda_pool_alloc<uint8_t> & union_buf, fa_sparse_pages * pages_out) {
    const fa_sparse_cfg & cfg = fa_sparse();
    if (cfg.budget <= 0) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];
    if (!Q || !K || !V || !M || Q->type != GGML_TYPE_F32 || Q->ne[3] != 1 || K->ne[3] != 1) {
        return false;
    }
    if (M->type != GGML_TYPE_F16) {
        // prefill-sized ops get the packed mask; fast mode does not take them anyway, so only decode-sized ops warn
        static bool warned = false;
        if (!warned && (!cfg.fast || Q->ne[1] <= cfg.maxq)) {
            GGML_LOG_WARN("%s: GGML_CUDA_FA_SPARSE needs an f16 mask (LLAMA_ARG_KQ_MASK_PACKED=0); off\n", __func__);
            warned = true;
        }
        return false;
    }
    // the selected region: the host tail, or with GGML_CUDA_FA_SPARSE_SINKS every cell from the sinks on (the op must
    // still reach the host tail, so short contexts stay dense)
    const int64_t first_host = ggml_cuda_tier_first_host_cell(K);
    const int64_t first = first_host >= 0 && cfg.sinks >= 0 ? std::min<int64_t>(cfg.sinks, first_host) : first_host;
    const int64_t n_kv  = K->ne[1];
    if (first < 0 || first >= n_kv || n_kv - first_host < cfg.min_host) {
        return false;
    }
    const int64_t n_host  = n_kv - first;
    const int     page    = (int) cfg.page;
    const int     n_pages = (int) ((n_host + page - 1)/page);
    const int64_t keep_pages = fa_sparse_keep_pages(cfg, n_host);
    if (n_pages <= keep_pages || n_pages > 12288) {
        return false;   // nothing to drop, or more page scores than fa_sparse_select holds in shared memory (48 KB)
    }
    const int D = (int) K->ne[0], H_kv = (int) K->ne[2], n_head = (int) Q->ne[2], n_q = (int) Q->ne[1];
    if (cfg.fast && n_q > cfg.maxq) {
        return false;   // prefill stays dense in fast mode
    }
    const int gqa = n_head / H_kv;
    cudaStream_t stream = ctx.stream();
    // the mask copy first: the pool frees in reverse order, and it outlives the temporaries below
    const int64_t m_s1   = M->nb[1]/sizeof(half);
    const int64_t m_rows = M->ne[1];
    mask_buf.alloc((size_t) m_s1*m_rows);
    union_buf.alloc((size_t) n_pages);   // also outlives this function: allocated before the temporaries below

    // page bounds: [H_kv][stride_pages][D] max and min. Emulation: all host pages, every op. Fast mode: a table per K tensor
    // kept across ops (fa_sparse_tables), filled once, then only the pages from the first written cell (writes lower
    // `valid`) or of the last `recent + page` cells, whichever is lower, are recomputed.
    ggml_cuda_pool_alloc<uint8_t> bnd_tmp(ctx.pool());
    const int8_t * mx = nullptr;
    const int8_t * mn = nullptr;
    const half   * sc = nullptr;
    int stride_pages = n_pages;
    int p_from = 0;   // first page to (re)compute
    int8_t * mx_w = nullptr;
    int8_t * mn_w = nullptr;
    half   * sc_w = nullptr;
    std::unique_lock<std::mutex> tbl_lock(fa_sparse_mtx, std::defer_lock);
    if (cfg.fast) {
        tbl_lock.lock();
        fa_sparse_table & t = fa_sparse_tables[(const char *) K->data];
        const ggml_tensor * Ks = K->view_src ? K->view_src : K;
        t.end = (const char *) Ks->data + ggml_nbytes(Ks);
        t.row = K->nb[1];
        fa_sparse_lo.store(std::min<uintptr_t>(fa_sparse_lo.load(), (uintptr_t) K->data));
        fa_sparse_hi.store(std::max<uintptr_t>(fa_sparse_hi.load(), (uintptr_t) t.end));
        if (t.first != first || t.cap < n_pages) {
            const int cap = std::max(n_pages, t.cap) + 64;
            if (t.mx) {
                CUDA_CHECK(cudaFree(t.mx));
            }
            CUDA_CHECK(cudaMalloc(&t.mx, fa_sparse_bnd_bytes(H_kv, cap, D)));
            t.mn = t.mx + (size_t) H_kv*cap*D;
            t.sc = (half *) (t.mx + (size_t) H_kv*cap*D*2);
            t.cap = cap;
            t.first = first;
            t.valid = 0;
        }
        p_from = (int) std::max<int64_t>(0, std::min<int64_t>(t.valid, n_host - cfg.recent - page)/page);
        t.valid = n_host;
        stride_pages = t.cap;
        mx_w = t.mx; mn_w = t.mn; sc_w = t.sc;
        tbl_lock.unlock();
    } else {
        bnd_tmp.alloc(fa_sparse_bnd_bytes(H_kv, n_pages, D));
        mx_w = (int8_t *) bnd_tmp.get();
        mn_w = mx_w + (size_t) H_kv*n_pages*D;
        sc_w = (half *) (mx_w + (size_t) H_kv*n_pages*D*2);
    }
    if (p_from < n_pages) {
        // K rows of pages [p_from, n_pages) of the host tail, as contiguous f16 [D, cells, H_kv], in chunks of at most
        // GGML_CUDA_FA_SPARSE_FILL_CELLS cells (default 16384): the first fill of a deep tail would otherwise take a
        // pool temporary of the whole tail (~380 MiB at 245k) at the first decode step, outside the VRAM budget
        static const int64_t fill_cells = [] {
            const char * e = getenv("GGML_CUDA_FA_SPARSE_FILL_CELLS");
            return e ? std::max<int64_t>(1, atoll(e)) : (int64_t) 16384;
        }();
        const int     chunk_pages = (int) std::max<int64_t>(1, fill_cells/page);
        const int64_t c_from      = (int64_t) p_from*page;
        const int64_t buf_cells   = std::min<int64_t>(n_host - c_from, (int64_t) chunk_pages*page);
        ggml_cuda_pool_alloc<half> Kh(ctx.pool(), (size_t) D*buf_cells*H_kv);
        const size_t ts = ggml_type_size(K->type);
        to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
        GGML_ASSERT(to_fp16 != nullptr);
        for (int p0 = p_from; p0 < n_pages; p0 += chunk_pages) {
            const int     np  = std::min(chunk_pages, n_pages - p0);
            const int64_t c0  = (int64_t) p0*page;
            const int64_t nc  = std::min<int64_t>(n_host - c0, (int64_t) np*page);
            const char *  kd  = (const char *) K->data + (first + c0)*K->nb[1];
            to_fp16(kd, Kh.get(), D, nc, H_kv, 1, K->nb[1]/ts, K->nb[2]/ts, K->nb[3]/ts, stream);
            fa_sparse_bounds<<<dim3(np, H_kv), 256, 2*D*sizeof(float), stream>>>(Kh.get(), mx_w, mn_w, sc_w, D, (int) nc, page, p0, stride_pages);
        }
    }
    mx = mx_w; mn = mn_w; sc = sc_w;
    if (cfg.fast && cfg.check) {
        // recompute every host page into temporaries and count the cached pages that differ
        ggml_cuda_pool_alloc<half>  Kf(ctx.pool(), (size_t) D*n_host*H_kv);
        ggml_cuda_pool_alloc<uint8_t> fbnd(ctx.pool(), fa_sparse_bnd_bytes(H_kv, n_pages, D));
        int8_t * fmx = (int8_t *) fbnd.get();
        int8_t * fmn = fmx + (size_t) H_kv*n_pages*D;
        half   * fsc = (half *) (fmx + (size_t) H_kv*n_pages*D*2);
        ggml_cuda_pool_alloc<int>   d_bad(ctx.pool(), 2);
        const size_t ts = ggml_type_size(K->type);
        ggml_get_to_fp16_nc_cuda(K->type)((const char *) K->data + first*K->nb[1], Kf.get(), D, n_host, H_kv, 1,
            K->nb[1]/ts, K->nb[2]/ts, K->nb[3]/ts, stream);
        fa_sparse_bounds<<<dim3(n_pages, H_kv), 256, 2*D*sizeof(float), stream>>>(Kf.get(), fmx, fmn, fsc, D, (int) n_host, page, 0, n_pages);
        const int init[2] = {0, INT_MAX};
        CUDA_CHECK(cudaMemcpyAsync(d_bad.get(), init, sizeof(init), cudaMemcpyHostToDevice, stream));
        fa_sparse_diff<<<dim3(n_pages, H_kv), 256, 0, stream>>>(mx, mn, sc, stride_pages, fmx, fmn, fsc, n_pages, D, d_bad.get());
        int bad[2];
        CUDA_CHECK(cudaMemcpyAsync(bad, d_bad.get(), sizeof(bad), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        static int64_t ck_ops = 0, ck_bad_ops = 0;
        ++ck_ops;
        if (bad[0] > 0) {
            ++ck_bad_ops;
            GGML_LOG_WARN("%s: CHECK stale bounds: K %p, %d page-heads of %d differ, first page %d (n_host %lld, pages %d, from %d)\n",
                __func__, K->data, bad[0], n_pages*H_kv, bad[1], (long long) n_host, n_pages, p_from);
        }
        if (ck_ops % 4096 == 0) {
            GGML_LOG_INFO("%s: CHECK %lld ops, %lld with stale bounds\n", __func__, (long long) ck_ops, (long long) ck_bad_ops);
        }
    }
    ggml_cuda_pool_alloc<float> ub(ctx.pool(), (size_t) n_q*n_pages);
    fa_sparse_scores<<<dim3(n_pages, n_q), 256, 0, stream>>>((const float *) Q->data, mx, mn, sc, ub.get(), D, n_pages,
        n_head, gqa, (int64_t) (Q->nb[1]/sizeof(float)), (int64_t) (Q->nb[2]/sizeof(float)), stride_pages);

    // a copy of the mask, the causal boundary of each row in the host tail, and the pages to keep (all on the device)
    CUDA_CHECK(cudaMemcpyAsync(mask_buf.get(), M->data, (size_t) m_s1*m_rows*sizeof(half), cudaMemcpyDeviceToDevice, stream));
    ggml_cuda_pool_alloc<int> d_last(ctx.pool(), (size_t) n_q);
    fa_sparse_last<<<n_q, 256, 0, stream>>>((const half *) M->data, d_last.get(), first, n_host, m_s1);
    ggml_cuda_pool_alloc<uint8_t> d_keep(ctx.pool(), (size_t) n_q*n_pages);
    fa_sparse_select<<<n_q, 256, n_pages*sizeof(float), stream>>>(ub.get(), d_last.get(), d_keep.get(), n_pages, page,
        (int) cfg.recent, (int) keep_pages);
    // debug: GGML_CUDA_FA_SPARSE_TRACE=a,b[,n] logs, for the first n ops (default 4096), the rank of the best page that
    // holds cells [a, b] (absolute positions) in query row 0's page scores, and whether the selection kept it. Rank 0 is
    // the top page; kept when rank < keep. "layer" is the order in which the K tensors were first seen. Syncs every op.
    {
        static int64_t tr[3] = { -1, -1, 4096 };
        static const bool tr_on = [] {
            const char * e = getenv("GGML_CUDA_FA_SPARSE_TRACE");
            if (!e) { return false; }
            long long a = -1, b = -1, n = 4096;
            sscanf(e, "%lld,%lld,%lld", &a, &b, &n);
            tr[0] = a; tr[1] = b; tr[2] = n;
            return a >= 0 && b >= a;
        }();
        static int64_t tr_ops = 0;
        static std::map<const void *, int> tr_layer;
        if (tr_on && tr_ops < tr[2] && tr[1] >= first && tr[0] < n_kv) {
            ++tr_ops;
            const int layer = tr_layer.emplace(K->data, (int) tr_layer.size()).first->second;
            std::vector<float> h_ub(n_pages);
            CUDA_CHECK(cudaMemcpyAsync(h_ub.data(), ub.get(), n_pages*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const int pa = (int) (std::max<int64_t>(tr[0], first) - first)/page;
            const int pb = (int) std::min<int64_t>((std::min<int64_t>(tr[1], n_kv - 1) - first)/page, n_pages - 1);
            int best_rank = INT_MAX, best_p = -1;
            for (int pt = pa; pt <= pb; ++pt) {
                int rank = 0;
                for (int q = 0; q < n_pages; ++q) {
                    rank += h_ub[q] > h_ub[pt] || (h_ub[q] == h_ub[pt] && q < pt);
                }
                if (rank < best_rank) { best_rank = rank; best_p = pt; }
            }
            float top = -INFINITY;
            for (int q = 0; q < n_pages; ++q) { top = fmaxf(top, h_ub[q]); }
            GGML_LOG_WARN("%s: TRACE layer %d op %lld: cells %lld..%lld = pages %d..%d of %d; best rank %d (page %d, score %.3f, top %.3f), keep %lld: %s\n",
                __func__, layer, (long long) tr_ops, (long long) tr[0], (long long) tr[1], pa, pb, n_pages, best_rank, best_p,
                best_p >= 0 ? h_ub[best_p] : 0.0f, top, (long long) keep_pages, best_rank < keep_pages ? "kept" : "DROPPED");
        }
    }
    if (cfg.check) {
        // the earlier host selection, same tie rule: count the pages where the two differ
        std::vector<float>   h_ub((size_t) n_q*n_pages);
        std::vector<int>     h_last((size_t) n_q);
        std::vector<uint8_t> h_dev((size_t) n_q*n_pages), keep((size_t) n_q*n_pages, 0);
        CUDA_CHECK(cudaMemcpyAsync(h_ub.data(), ub.get(), h_ub.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_last.data(), d_last.get(), h_last.size()*sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_dev.data(), d_keep.get(), h_dev.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::pair<float, int>> cand;
        for (int j = 0; j < n_q; ++j) {
            const int64_t last = h_last[j];
            if (last < 0) {
                continue;
            }
            const int last_page = (int) (last/page);
            uint8_t * k = keep.data() + (size_t) j*n_pages;
            const int64_t recent_from = std::max<int64_t>(0, last + 1 - cfg.recent);
            for (int p = (int) (recent_from/page); p <= last_page; ++p) {
                k[p] = 1;
            }
            cand.clear();
            for (int p = 0; p <= last_page; ++p) {
                if (!k[p]) {
                    cand.push_back({ h_ub[(size_t) j*n_pages + p], p });
                }
            }
            const size_t take = (size_t) std::min<int64_t>(keep_pages, (int64_t) cand.size());
            std::partial_sort(cand.begin(), cand.begin() + take, cand.end(),
                [](const std::pair<float, int> & x, const std::pair<float, int> & y) {
                    return x.first > y.first || (x.first == y.first && x.second < y.second); });
            for (size_t i = 0; i < take; ++i) {
                k[cand[i].second] = 1;
            }
        }
        int64_t diff = 0;
        for (size_t i = 0; i < keep.size(); ++i) {
            diff += keep[i] != h_dev[i];
        }
        static int64_t sel_ops = 0, sel_bad = 0;
        ++sel_ops;
        if (diff) {
            ++sel_bad;
            GGML_LOG_WARN("%s: CHECK selection differs from the host one in %lld of %lld pages\n", __func__,
                (long long) diff, (long long) keep.size());
        }
        if (sel_ops % 4096 == 0) {
            GGML_LOG_INFO("%s: CHECK selection: %lld ops, %lld differ\n", __func__, (long long) sel_ops, (long long) sel_bad);
        }
    }
    fa_sparse_apply<<<dim3((unsigned) ((n_host + 255)/256), n_q), 256, 0, stream>>>(mask_buf.get(), d_keep.get(), first, n_host,
        page, n_pages, m_s1);
    fa_sparse_union<<<(n_pages + 255)/256, 256, 0, stream>>>(d_keep.get(), union_buf.get(), n_q, n_pages);
    pages_out->first   = first;
    pages_out->first_host = first_host;
    pages_out->page    = page;
    pages_out->n_pages = n_pages;
    pages_out->keep    = union_buf.get();
    static int64_t st_ops = 0;
    if (++st_ops % 1024 == 1) {   // the first op and every 1024th: the path is engaged (printed while the server runs)
        GGML_LOG_WARN("%s: GGML_CUDA_FA_SPARSE B=%lld P=%lld R=%lld%s, bounds int8: %lld ops; this op %d queries, host tail %lld cells, keeps %lld of %d pages + recent\n",
            __func__, (long long) cfg.budget, (long long) cfg.page, (long long) cfg.recent, cfg.fast ? " fast" : "",
            (long long) st_ops, n_q, (long long) n_host, (long long) keep_pages, n_pages);
    }
    *mask_out = *M;
    mask_out->data      = mask_buf.get();
    mask_out->view_src  = nullptr;
    mask_out->view_offs = 0;
    return true;
}

// Packed KQ mask (GGML_TYPE_I32, 32 cells per word, bit set = attend) expanded to the f16 form the kernels read,
// into a pool buffer sized by this op's actual KV length. The reserved input tensor is 16x smaller than an f16
// mask; the f16 copy exists only for the duration of the op. (A native packed read in the tensor-core kernel is
// the next step; this keeps every kernel correct meanwhile.)
static __global__ void expand_kq_mask_bits(const uint32_t * __restrict__ bits, half * __restrict__ out, const int64_t n_words_total) {
    const int64_t w = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (w >= n_words_total) {
        return;
    }
    const uint32_t v = bits[w];
    half * o = out + w*32;
    const half keep = __float2half(0.0f), drop = __float2half(-INFINITY);
#pragma unroll
    for (int b = 0; b < 32; ++b) {
        o[b] = (v >> b) & 1u ? keep : drop;
    }
}

// GGML_CUDA_KV_TIER_STAGE_MIN_Q=q: attention ops with fewer than q queries (decode, small verify batches) that are not
// in the prefill prefetch read the host tail in place and copy nothing. Unset: 0 (every op that reaches the tail stages).
static int64_t ggml_cuda_tier_stage_min_q() {
    static const int64_t q = [] {
        const char * e = getenv("GGML_CUDA_KV_TIER_STAGE_MIN_Q");
        return e ? (int64_t) atoll(e) : (int64_t) 0;
    }();
    return q;
}

// ggml-cuda.cu: copy a range of a tiered buffer that reaches its host part into VRAM (dst == nullptr: only ask)
bool ggml_cuda_tier_copy_out(void * dst, const void * ptr, size_t nbytes, cudaStream_t stream);

// GGML_CUDA_FA_CHUNK=C: prefill-sized attention ops (at least GGML_CUDA_FA_PREFILL_F16_MIN_Q queries) over a quantized
// K/V cache longer than C cells run the KV dimension in chunks of C cells (rounded down to a multiple of 256). Per
// chunk: the chunk's K/V bytes are copied into VRAM when they reach the host part of a tiered buffer (copy engine,
// each host row read once), converted to f16 in pool memory sized by C, and the f16 tensor-core kernel runs on them
// with the matching mask columns. Its unnormalized output and (max, rowsum) per row are folded into a running result
// (launch_fattn, flash_attn_chunk_fold). Memory is bounded by C, not by the KV length, so the f16 prefill path works
// at any depth. Unset or 0: off, nothing changes. Anything this path does not handle takes the normal path.
static int64_t ggml_cuda_fattn_chunk_cells() {
    static const int64_t c = [] {
        const char * e = getenv("GGML_CUDA_FA_CHUNK");
        const int64_t v = e ? (int64_t) atoll(e) : 0;
        return v > 0 ? std::max<int64_t>(FATTN_KQ_STRIDE, v - v % FATTN_KQ_STRIDE) : (int64_t) 0;
    }();
    return c;
}

bool ggml_cuda_flash_attn_ext_chunked(const ggml_tensor * dst) {
    const int64_t C = ggml_cuda_fattn_chunk_cells();
    if (C == 0) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (!Q || !K || !V || K == V || sinks) {
        return false;
    }
    if (V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs))) {
        return false;
    }
    if (Q->ne[1] < ggml_cuda_fattn_prefill_min_q() || K->ne[1] <= C || K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || K->type != V->type || !ggml_is_quantized(K->type) ||
            ggml_get_to_fp16_nc_cuda(K->type) == nullptr) {
        return false;
    }
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }
    // head sizes of the plain f16 tensor-core kernel (ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2)
    switch (K->ne[0]) {
        case 64: case 80: case 96: case 112: case 128: case 256:
            break;
        default:
            return false;
    }
    if (V->ne[0] != K->ne[0]) {
        return false;
    }
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    // one sequence: a chunk of cells is then one contiguous byte range of K and of V
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || (mask && (mask->ne[2] != 1 || mask->ne[3] != 1))) {
        return false;
    }
    if (mask && mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_I32) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc)) {
        return false;
    }
    return ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst) == BEST_FATTN_KERNEL_MMA_F16;
}

static void ggml_cuda_flash_attn_ext_chunks(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_pool & pool   = ctx.pool();
    cudaStream_t     stream = ctx.stream();

    const int64_t C        = ggml_cuda_fattn_chunk_cells();
    const int64_t n_kv     = K->ne[1];
    const int64_t n_chunks = (n_kv + C - 1) / C;

    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("%s: GGML_CUDA_FA_CHUNK: chunks of %lld cells (first op: %lld cells, %lld chunks, %lld queries)\n",
            __func__, (long long) C, (long long) n_kv, (long long) n_chunks, (long long) dst->src[0]->ne[1]);
        logged = true;
    }

    // running (max, rowsum) per output row and Q head; the running output is dst itself
    ggml_cuda_pool_alloc<float2> meta(pool, ggml_nrows(dst));
    // unnormalized output of one chunk
    ggml_cuda_pool_alloc<float>  part(pool, ggml_nelements(dst));

    const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
    const size_t ts = ggml_type_size(K->type);

    for (int64_t ic = 0; ic < n_chunks; ++ic) {
        const int64_t kv0 = ic*C;
        const int64_t n   = std::min(C, n_kv - kv0);

        // quantized chunk views of K and V (cells [kv0, kv0 + n))
        ggml_tensor Kc = *K;
        ggml_tensor Vc = *V;
        Kc.ne[1] = n;
        Vc.ne[1] = n;
        Kc.data  = (char *) K->data + kv0*K->nb[1];
        Vc.data  = (char *) V->data + kv0*V->nb[1];

        // host-backed rows of a tiered cache: one DMA copy per chunk into VRAM; the kernels never read host pages
        ggml_cuda_pool_alloc<char> K_q(pool);
        ggml_cuda_pool_alloc<char> V_q(pool);
        const char * K_src = (const char *) Kc.data;
        const char * V_src = (const char *) Vc.data;
        const size_t K_span = ggml_nbytes(&Kc);
        const size_t V_span = ggml_nbytes(&Vc);
        if (ggml_cuda_tier_copy_out(nullptr, K_src, K_span, stream)) {
            ggml_cuda_tier_copy_out(K_q.alloc(K_span), K_src, K_span, stream);
            K_src = K_q.get();
        }
        if (ggml_cuda_tier_copy_out(nullptr, V_src, V_span, stream)) {
            ggml_cuda_tier_copy_out(V_q.alloc(V_span), V_src, V_span, stream);
            V_src = V_q.get();
        }

        // f16 copies of the chunk, contiguous [D, n, n_head_kv]
        ggml_cuda_pool_alloc<half> K_f16(pool, K->ne[0]*n*K->ne[2]);
        ggml_cuda_pool_alloc<half> V_f16(pool, V->ne[0]*n*V->ne[2]);
        to_fp16(K_src, K_f16.get(), K->ne[0], n, K->ne[2], 1, K->nb[1]/ts, K->nb[2]/ts, K->nb[3]/ts, stream);
        to_fp16(V_src, V_f16.get(), V->ne[0], n, V->ne[2], 1, V->nb[1]/ts, V->nb[2]/ts, V->nb[3]/ts, stream);

        ggml_tensor K_h = Kc;
        ggml_tensor V_h = Vc;
        for (ggml_tensor * t : {&K_h, &V_h}) {
            t->type      = GGML_TYPE_F16;
            t->nb[0]     = sizeof(half);
            t->nb[1]     = t->ne[0]*t->nb[0];
            t->nb[2]     = t->ne[1]*t->nb[1];
            t->nb[3]     = t->ne[2]*t->nb[2];
            t->view_src  = nullptr;
            t->view_offs = 0;
        }
        K_h.data = K_f16.get();
        V_h.data = V_f16.get();

        // mask columns of the chunk (packed: 32 cells per word; kv0 is a multiple of 256)
        ggml_tensor mask_c;
        if (mask) {
            const bool packed = mask->type == GGML_TYPE_I32;
            mask_c = *mask;
            mask_c.ne[0]    = packed ? n/32 : n;
            mask_c.data     = (char *) mask->data + (packed ? (kv0/32)*sizeof(uint32_t) : kv0*sizeof(half));
            mask_c.view_src = nullptr;
        }

        ggml_tensor dst_c = *dst;
        dst_c.data   = part.get();
        dst_c.src[1] = &K_h;
        dst_c.src[2] = &V_h;
        dst_c.src[3] = mask ? &mask_c : nullptr;

        const ggml_cuda_fattn_chunk state = { (float *) dst->data, meta.get(), ic == 0, ic == n_chunks - 1 };
        ctx.fattn_chunk = &state;
        ggml_cuda_flash_attn_ext_mma_f16(ctx, &dst_c);
        ctx.fattn_chunk = nullptr;
    }
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    // GGML_CUDA_FA_CHUNK: this op copies its own K/V chunks from the original addresses (no whole-op staging; the
    // tier prefetch leaves these ops out, see ggml_cuda_tier_graph_begin)
    if (ggml_cuda_flash_attn_ext_chunked(dst)) {
        ggml_cuda_flash_attn_ext_chunks(ctx, dst);
        return;
    }

    ggml_tensor * mask_in = dst->src[3];
    ggml_tensor mask_f16;
    ggml_cuda_pool_alloc<half> mask_buf(ctx.pool());
    // the tensor-core kernel reads packed bits natively; the vec and tile kernels get the expanded f16 copy
    const best_fattn_kernel best_for_mask = ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst);
    if (mask_in && mask_in->type == GGML_TYPE_I32 && best_for_mask != BEST_FATTN_KERNEL_MMA_F16) {
        const int64_t n_words = mask_in->ne[0];
        const int64_t rows    = mask_in->ne[1]*mask_in->ne[2]*mask_in->ne[3];
        mask_buf.alloc(n_words*32*rows);
        const int64_t total = n_words*rows;
        expand_kq_mask_bits<<<(unsigned) ((total + 255)/256), 256, 0, ctx.stream()>>>((const uint32_t *) mask_in->data, mask_buf.get(), total);
        mask_f16 = *mask_in;
        mask_f16.type  = GGML_TYPE_F16;
        mask_f16.ne[0] = n_words*32;
        mask_f16.nb[0] = sizeof(half);
        mask_f16.nb[1] = mask_f16.ne[0]*sizeof(half);
        mask_f16.nb[2] = mask_f16.nb[1]*mask_f16.ne[1];
        mask_f16.nb[3] = mask_f16.nb[2]*mask_f16.ne[2];
        mask_f16.data  = mask_buf.get();
        mask_f16.view_src = nullptr;
        dst->src[3] = &mask_f16;
    }
    struct restore_mask { ggml_tensor * dst; ggml_tensor * mask; ~restore_mask() { dst->src[3] = mask; } } restore{dst, mask_in};

    // GGML_CUDA_FA_SPARSE (experiment): this op's mask with the unselected pages of the host tail masked out
    ggml_tensor mask_sparse;
    ggml_cuda_pool_alloc<half>    sparse_buf(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> sparse_union(ctx.pool());
    fa_sparse_pages               sparse_pages;
    if (ggml_cuda_fa_sparse_mask(ctx, dst, sparse_buf, &mask_sparse, sparse_union, &sparse_pages)) {
        dst->src[3] = &mask_sparse;
    }
    struct sparse_flag { bool prev; sparse_flag(bool on) : prev(ggml_cuda_fa_sparse_active) { ggml_cuda_fa_sparse_active = on; }
        ~sparse_flag() { ggml_cuda_fa_sparse_active = prev; } } sparse_flag_guard(dst->src[3] == &mask_sparse);

    // K/V in a tiered buffer whose host tail this op reaches: copy the used host rows into the VRAM staging
    // buffer with the copy engine and point K/V at the all-VRAM alias of the same range for this op. Prefill
    // kernels read each K/V row once per query tile, which for host rows would be PCIe traffic every time;
    // decode reads each row once, but DMA moves it faster than SMs reading host memory from inside the
    // kernel (RTX 4070, 180k context, 86k cells in host memory: +28% decode). Nothing is copied for ops
    // that stay below the tier line. The staged bytes are the same bytes, so results are unchanged.
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    // With the prefetch (ggml_cuda_tier_graph_begin), the host rows are already in staging when this op starts.
    void * K_data = K ? K->data : nullptr;
    void * V_data = V ? V->data : nullptr;
    bool tier_piped = false;
    if (K && V && V != K) {
        void * K_alias = nullptr;
        void * V_alias = nullptr;
        tier_piped = ggml_cuda_tier_fa_begin(ctx, dst, &K_alias, &V_alias);
        if (tier_piped) {
            if (K_alias) {
                K->data = K_alias;
            }
            if (V_alias) {
                V->data = V_alias;
            }
        } else if (dst->src[0]->ne[1] >= ggml_cuda_tier_stage_min_q()) {
            // sparse op: stage only the host pages some query keeps (the others are masked out and keep whatever
            // finite bytes the shared staging buffer holds), so PCIe moves the kept pages only. V needs the same
            // host cells as K; otherwise everything is staged.
            const bool pages = sparse_pages.keep && ggml_cuda_tier_first_host_cell(V) == sparse_pages.first_host;
            for (ggml_tensor * t : { K, V }) {
                void * a = pages
                    ? ggml_cuda_tier_stage_pages(t->data, ggml_nbytes(t), t->nb[1], sparse_pages.first, sparse_pages.page,
                        sparse_pages.n_pages, sparse_pages.keep, ctx.stream())
                    : ggml_cuda_tier_stage(t->data, ggml_nbytes(t), ctx.stream());
                if (a) {
                    t->data = a;
                }
            }
        }
    }

    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }

    if (tier_piped) {
        ggml_cuda_tier_fa_end(ctx);
    }

    if (K) {
        K->data = K_data;
    }
    if (V) {
        V->data = V_data;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
