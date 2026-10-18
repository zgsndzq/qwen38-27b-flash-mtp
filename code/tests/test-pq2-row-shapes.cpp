// PQ2_0 rows are 128 wide, so a model may legitimately hold tensors whose row length is any
// multiple of 128 -- 128 and 384 as much as 256. The AVX2 kernels added for PQ2_0 consume Q8_K
// activations, which only exist in 256-element blocks, so routing every PQ2_0 matmul through
// Q8_K breaks exactly those rows. Guarding the quantizer is not enough: tensors already inside
// published models never pass through it.
//
// This exercises pre-existing PQ2_0 tensors directly (reference-encoded, as a model file holds
// them) for K in {128, 256, 384}, single-token and batched, and checks the numbers rather than
// just survival.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static std::vector<float> make_weights(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; i++) {
        v[i] = dist(rng);
    }
    return v;
}

// rms(computed - reference) / rms(reference), with the reference taken from the dequantized
// weights and exact float activations. Activation quantization (Q8_K or Q8_0) is the only
// remaining error term, so this stays far below the threshold when the path is correct and
// explodes when it is not.
static double rel_rms(const std::vector<float> & got, const std::vector<float> & ref) {
    double se = 0.0;
    double sr = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = (double) got[i] - (double) ref[i];
        se += d * d;
        sr += (double) ref[i] * (double) ref[i];
    }
    if (sr == 0.0) {
        return se == 0.0 ? 0.0 : INFINITY;
    }
    return std::sqrt(se / sr);
}

static bool run_shape(int64_t K, int64_t M, int64_t N) {
    const size_t mem_size = 64u * 1024 * 1024;
    struct ggml_init_params ip = { mem_size, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        printf("  K=%4lld M=%lld N=%lld : ggml_init failed\n", (long long) K, (long long) M, (long long) N);
        return false;
    }

    // A pre-existing PQ2_0 weight tensor: encoded by the reference row quantizer, which is what
    // a .gguf holds. It never sees llama-quant.cpp's type fallback.
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_0, K, M);
    const std::vector<float> w = make_weights(K * M, 1234 + (uint32_t) K);
    ggml_quantize_chunk(GGML_TYPE_PQ2_0, w.data(), a->data, 0, M, K, nullptr);

    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    const std::vector<float> x = make_weights(K * N, 99);
    memcpy(b->data, x.data(), x.size() * sizeof(float));

    struct ggml_tensor * c  = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // reference: dequantized weights x exact float activations
    std::vector<float> a_deq(K * M);
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_PQ2_0);
    tt->to_float(a->data, a_deq.data(), K * M);

    std::vector<float> ref(M * N);
    for (int64_t j = 0; j < N; j++) {
        for (int64_t i = 0; i < M; i++) {
            double s = 0.0;
            for (int64_t k = 0; k < K; k++) {
                s += (double) a_deq[i * K + k] * (double) x[j * K + k];
            }
            ref[j * M + i] = (float) s;
        }
    }

    std::vector<float> got(M * N);
    memcpy(got.data(), c->data, got.size() * sizeof(float));

    const double err = rel_rms(got, ref);
    const bool   ok  = err < 0.05;

    printf("  K=%4lld M=%lld N=%2lld : %-4s rel_rms=%.4f\n",
           (long long) K, (long long) M, (long long) N, ok ? "OK" : "FAIL", err);

    ggml_free(ctx);
    return ok;
}

int main() {
    printf("test-pq2-row-shapes: PQ2_0 matmul over every legal row width\n");

    bool all_ok = true;
    // 128 and 384 are legal PQ2_0 row widths that are NOT multiples of the Q8_K block (256).
    for (int64_t K : { (int64_t) 128, (int64_t) 256, (int64_t) 384, (int64_t) 512, (int64_t) 640 }) {
        for (int64_t N : { (int64_t) 1, (int64_t) 4, (int64_t) 16 }) {
            all_ok &= run_shape(K, 4, N);
        }
    }

    printf("%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
