// q3_0: our 3-bit K/V cache type. Everything device-side about it lives here, so a
// merge with upstream touches only the dispatch cases, never the implementation.
//
// Layout: 32 weights per block, one fp16 scale plus a 12-byte sequential bitstream
// (weight j owns bits [3j, 3j+3)). Stored as (w + 4), so a decoded weight is
// (u - 4) * d with u in [0, 7] and d = amax / 4. The zero point is folded into the
// weight, which is why vec_dot needs no sum-of-q correction the way q4_0 does.
// 14 bytes per 32 values against q4_0's 18: 22% less K/V.
//
// Included at the end of common.cuh, so WARP_SIZE, half and block_q3_0 are already
// visible. Dispatch (GGML_TYPE_Q3_0 cases) stays in the files that own them:
//   convert.cu, cpy.cu, set-rows.cu, ggml-cuda.cu, fattn-common.cuh, fattn.cu, arg.cpp.
//
// inventory: docs/private-fork.md
#pragma once

#include "ggml-common.h"

// weight j owns bits [3j, 3j+3) of the 12-byte payload
static __device__ __forceinline__ int q3_0_unpack3(const uint8_t * __restrict__ qs, const int j) {
    const int bit = 3*j;
    const int by  = bit >> 3;
    const int sh  = bit & 7;
    uint32_t u = (uint32_t) qs[by] >> sh;
    if (sh > 5) {
        u |= (uint32_t) qs[by + 1] << (8 - sh);
    }
    return (int) (u & 7) - 4;
}

// ---- f32 -> q3_0 (the path a K/V write takes) ------------------------------

static __device__ void quantize_f32_q3_0_block(const float * __restrict__ x, block_q3_0 * __restrict__ y) {
    float amax = 0.0f;

    for (int j = 0; j < QK3_0; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }

    const float d  = amax / 4.0f;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint8_t * qs = y->qs;
#pragma unroll
    for (int j = 0; j < (QK3_0 * 3) / 8; ++j) {
        qs[j] = 0;
    }

    for (int j = 0; j < QK3_0; ++j) {
        int q = (int) (x[j]*id + 4.5f);
        q = min(7, max(0, q));

        const int bit = 3*j;
        qs[bit >> 3] |= (uint8_t) (q << (bit & 7));
        if ((bit & 7) > 5) {
            qs[(bit >> 3) + 1] |= (uint8_t) (q >> (8 - (bit & 7)));
        }
    }
}

static __device__ void cpy_blck_f32_q3_0(const char * cxi, char * cdsti) {
    quantize_f32_q3_0_block((const float *) cxi, (block_q3_0 *) cdsti);
}

// q3_0 is a sequential bitstream, so it cannot use the interleaved cpy_blck_q_f32
// (which reads element j and j + qk/2); unpack in order instead.
static __device__ void cpy_blck_q3_0_f32(const char * cxi, char * cdsti) {
    const block_q3_0 * x = (const block_q3_0 *) cxi;
    float * cdstf = (float *) cdsti;

    const float d = x->d;

    for (int j = 0; j < QK3_0; ++j) {
        cdstf[j] = q3_0_unpack3(x->qs, j) * d;
    }
}

// ---- q3_0 -> f32, the pair form the generic dequantize templates want -------

static __device__ __forceinline__ void dequantize_q3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q3_0 * x = (const block_q3_0 *) vx;
    const float d = x[ib].d;
    // qr == 1, so the caller passes an even element index and wants that
    // element and its neighbour.
    v.x = q3_0_unpack3(x[ib].qs, iqs + 0) * d;
    v.y = q3_0_unpack3(x[ib].qs, iqs + 1) * d;
}

// ---- flash attention -------------------------------------------------------
// No dp4a shortcut: a 3-bit weight does not fill a byte, so the four weights of one
// int group are unpacked and multiplied one by one. The zero point is folded into
// the weight (w in [-4, 3]), so unlike q4_0 there is no sum-of-q correction.

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q3_0 * K_q3_0 = (const block_q3_0 *) K_c;
    GGML_UNUSED(Q_v);

    const int8_t * q8 = (const int8_t *) Q_q8;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int i0 = 4*k_KQ;
        const int ib = i0 / QK3_0;

        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sumi += q3_0_unpack3(K_q3_0[ib].qs, (i0 % QK3_0) + l) * q8[i0 + l];
        }

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q3_0[ib].d) * Q_ds.x * sumi;
    }

    return sum;
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q3_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q3_0 * x = (const block_q3_0 *) vx;

    static_assert(ne == 2 || ne == 4, "bad ne");

    const float d = __half2float(x[i0 / QK3_0].d);
    const int   j0 = (int) (i0 % QK3_0);

    if constexpr (std::is_same_v<T, float>) {
        float * dst_f = (float *) dst;
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            dst_f[l] = q3_0_unpack3(x[i0 / QK3_0].qs, j0 + l) * d;
        }
    } else {
        half * dst_h = (half *) dst;
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            dst_h[l] = __float2half(q3_0_unpack3(x[i0 / QK3_0].qs, j0 + l) * d);
        }
    }
}
