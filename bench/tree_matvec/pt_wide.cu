// Standalone probe: can a PTQ1_0 matvec stay bandwidth-bound at 16-48 columns?
//
// The shipping kernel (mmvq-ptq1_0.cuh) cannot answer this. Its inner loop keeps
// u[ncols] / u2[ncols] int4 activation tiles in registers next to acc[ncols][rows],
// so registers grow ~4 per column on top of an already 224-reg footprint at 5
// columns (cuobjdump, quoted in that file) and 251 at 8. At 16 columns the
// formulation is out of registers, which is why rows/item collapses to 1 and the
// 6-column cliff (13 t/s) happens.
//
// This probe uses the opposite arrangement: activations live in shared memory,
// one K block at a time, and each thread holds ONLY the accumulators for its row
// (NCOLS ints) plus the trit words it is currently unpacking. Registers then grow
// ~1 per column, which is what makes 48 columns reachable.
//
// It is a bandwidth/scheduling probe, not a correctness test: the trit->weight
// mapping is the real base-3 stepping, but element ordering within the block is
// not checked against the ggml reference. What it measures is:
//   * is the weight stream still at ~272 GB/s (i.e. are we bandwidth-bound), and
//   * how many tokens per second a step at that column count would yield.
//
// build.bat compiles it; run.bat executes with the model-sized weight buffer.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define QK 128
#define K_DIM 5120            // hidden size of the 27B
#define ROWS_TOTAL 5271000    // 5271000 * 5120 * 1.75/8 = 5.79 GB of weights

struct __align__(4) block_pt {
    uint8_t qs[24];
    uint8_t qh[2];
    __half  d;
};
static_assert(sizeof(block_pt) == 28, "packing");

// same base-3 stepping as the shipping kernel: five trits per byte, carried in
// two 16-bit lanes, returned as four bytes
__device__ __forceinline__ uint32_t trit_step(uint32_t & vlo, uint32_t & vhi) {
    const uint32_t wlo = vlo * 3;
    const uint32_t whi = vhi * 3;
    vlo = wlo & 0x00FF00FF;
    vhi = whi & 0x00FF00FF;
    return __byte_perm(wlo, whi, 0x7531);
}

__device__ __forceinline__ int i4_at(const int4 & v, const int k) {
    switch (k & 3) {
        case 0:  return v.x;
        case 1:  return v.y;
        case 2:  return v.z;
        default: return v.w;
    }
}

// One thread owns one output row. The CTA stages one 128-weight slice of ALL
// NCOLS activation columns into shared memory, then every thread streams its own
// weight block against those same activation bytes -- the weight is read once per
// (row, block) no matter how many columns are in flight.
template <int NCOLS, int THREADS>
__global__ __launch_bounds__(THREADS) void mmv_wide(
        const block_pt * __restrict__ W, const int8_t * __restrict__ A,
        float * __restrict__ out, const int nblk) {
    __shared__ int8_t sA[NCOLS][QK];

    const int row = blockIdx.x * THREADS + threadIdx.x;
    if (row >= ROWS_TOTAL) {
        return;
    }

    int acc[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        acc[j] = 0;
    }

    const block_pt * wrow = W + (long long) row * nblk;

    for (int kb = 0; kb < nblk; ++kb) {
        // stage this K block's activations (NCOLS * 128 bytes)
        for (int i = threadIdx.x; i < NCOLS * QK; i += THREADS) {
            sA[i / QK][i % QK] = A[(long long) kb * QK + (i % QK)];
        }
        __syncthreads();

        const block_pt & b = wrow[kb];

        uint32_t vlo[4], vhi[4];
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint32_t packed = ((const uint32_t *) b.qs)[g];
            vlo[g] = __byte_perm(packed, 0, 0x4140);
            vhi[g] = __byte_perm(packed, 0, 0x4342);
        }

        // 80 trits: qs[0..15]
#pragma unroll
        for (int t = 0; t < 5; ++t) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t q = trit_step(vlo[g], vhi[g]);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const int4 a = ((const int4 *) sA[j])[g];
                    acc[j] = __dp4a((int) q, i4_at(a, g), acc[j]);
                }
            }
        }

        // 40 trits: qs[16..23]
        uint32_t vlo2[2], vhi2[2];
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            uint32_t packed = ((const uint32_t *) (b.qs + 16))[g];
            vlo2[g] = __byte_perm(packed, 0, 0x4140);
            vhi2[g] = __byte_perm(packed, 0, 0x4342);
        }
        int w = 20;
#pragma unroll
        for (int t = 0; t < 5; ++t) {
#pragma unroll
            for (int g = 0; g < 2; ++g, ++w) {
                const uint32_t q = trit_step(vlo2[g], vhi2[g]);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const int4 a = ((const int4 *) sA[j])[w & 3];
                    acc[j] = __dp4a((int) q, i4_at(a, w), acc[j]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        out[(long long) j * ROWS_TOTAL + row] = (float) acc[j];
    }
}

template <int NCOLS, int THREADS>
static float run(const block_pt * W, const int8_t * A, float * out, int nblk, int reps) {
    const int blocks = (ROWS_TOTAL + THREADS - 1) / THREADS;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    mmv_wide<NCOLS, THREADS><<<blocks, THREADS>>>(W, A, out, nblk);   // warm
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) {
        mmv_wide<NCOLS, THREADS><<<blocks, THREADS>>>(W, A, out, nblk);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / reps;
}

int main() {
    const int nblk = K_DIM / QK;
    const size_t wbytes = (size_t) ROWS_TOTAL * nblk * sizeof(block_pt);
    const size_t abytes = (size_t) K_DIM * 48;

    printf("weights: %.2f GB, rows=%d, K=%d, blocks/row=%d\n",
           wbytes / 1e9, ROWS_TOTAL, K_DIM, nblk);

    block_pt * W = nullptr;
    int8_t   * A = nullptr;
    float    * out = nullptr;
    if (cudaMalloc(&W, wbytes) != cudaSuccess) { printf("cudaMalloc W failed\n"); return 1; }
    cudaMalloc(&A, abytes);
    cudaMalloc(&out, (size_t) 48 * ROWS_TOTAL * sizeof(float));

    // synthetic contents: alternating trit patterns keep the unpack path honest
    std::vector<uint8_t> hostw(1 << 20);
    for (size_t i = 0; i < hostw.size(); ++i) hostw[i] = (uint8_t) (i * 37 + 11);
    for (size_t off = 0; off < wbytes; off += hostw.size()) {
        const size_t n = (wbytes - off) < hostw.size() ? (wbytes - off) : hostw.size();
        cudaMemcpy((char *) W + off, hostw.data(), n, cudaMemcpyHostToDevice);
    }
    cudaMemset(A, 1, abytes);

    struct { int ncols; float ms; } res[6];
    int n = 0;
    res[n++] = { 4,  run<4,  256>(W, A, out, nblk, 5) };
    res[n++] = { 8,  run<8,  256>(W, A, out, nblk, 5) };
    res[n++] = { 16, run<16, 256>(W, A, out, nblk, 5) };
    res[n++] = { 24, run<24, 256>(W, A, out, nblk, 5) };
    res[n++] = { 32, run<32, 256>(W, A, out, nblk, 5) };
    res[n++] = { 48, run<48, 256>(W, A, out, nblk, 5) };

    printf("\n%6s %10s %12s %12s %10s\n", "cols", "ms/step", "GB/s", "tok/s@1step", "vs 4cols");
    const float base = res[0].ms / (float) res[0].ncols;   // ms per column at 4 cols
    for (int i = 0; i < n; ++i) {
        const float gbs = (float) wbytes / (res[i].ms * 1e6f);
        const float tps = (float) res[i].ncols / (res[i].ms * 1e-3f);
        const float percol = res[i].ms / (float) res[i].ncols;
        printf("%6d %10.2f %12.1f %12.0f %9.2fx  (eff %.0f%% of 272 GB/s)\n",
               res[i].ncols, res[i].ms, gbs, tps, base / percol, gbs / 272.0f * 100.0f);
    }

    cudaFree(W);
    cudaFree(A);
    cudaFree(out);
    return 0;
}
