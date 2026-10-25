#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_add_rms_norm_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul);

bool ggml_cuda_add_rms_norm_fwht_supported(int ncols);

void ggml_cuda_op_add_rms_norm_fwht_fused(
        ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * rms_norm,
        const ggml_tensor * weight, const ggml_tensor * signs, ggml_tensor * normed, ggml_tensor * dst);

void ggml_cuda_op_add_rms_norm_scale_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, float * row_scale);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
