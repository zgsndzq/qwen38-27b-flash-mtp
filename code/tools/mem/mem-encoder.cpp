#include "mem-internal.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>

mem_encoder::~mem_encoder() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_model_free(model);
    }
}

bool mem_encoder::load(const std::string & path, int32_t key_dim_in, std::string & err) {
    key_dim = key_dim_in;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;

    // pin to CPU so the encoder contributes 0 MiB VRAM
    static ggml_backend_dev_t cpu_devs[2] = { nullptr, nullptr };
    cpu_devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    mp.devices = cpu_devs;

    model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        err = "cannot load memory encoder: " + path;
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx          = n_ctx;
    cp.n_batch        = n_ctx;
    cp.n_ubatch       = n_ctx;
    cp.embeddings     = true;
    cp.pooling_type   = LLAMA_POOLING_TYPE_MEAN;
    ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        err = "cannot create encoder context";
        return false;
    }
    vocab = llama_model_get_vocab(model);
    return true;
}

bool mem_encoder::encode(const std::string & text, std::vector<float> & out) {
    std::vector<llama_token> ids(n_ctx);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                           ids.data(), (int32_t) ids.size(), true, true);
    if (n < 0) {
        ids.resize((size_t) -n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                           ids.data(), (int32_t) ids.size(), true, true);
    }
    if (n <= 0) {
        return false;
    }
    if (n > n_ctx) {
        n = n_ctx;
    }

    llama_batch b = llama_batch_get_one(ids.data(), n);
    if (llama_decode(ctx, b) != 0) {
        return false;
    }

    const float * emb = llama_get_embeddings_seq(ctx, 0);
    if (emb == nullptr) {
        return false;
    }

    // keep the first key_dim dims and L2-normalize (matches the ctypes tower)
    const int32_t n_embd = (int32_t) llama_model_n_embd(model);
    const int32_t dim = std::min(key_dim, n_embd);
    out.assign(key_dim, 0.0f);
    for (int32_t i = 0; i < dim; i++) {
        out[i] = emb[i];
    }
    double norm = 0.0;
    for (float v : out) {
        norm += (double) v * v;
    }
    norm = std::sqrt(norm);
    if (norm < 1e-8) {
        return false;
    }
    for (float & v : out) {
        v = (float) (v / norm);
    }
    return true;
}
