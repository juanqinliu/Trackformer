#pragma once
#include <torch/extension.h>
#include "cpu/ms_deform_attn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif

// 辅助函数：检查张量是否在 CUDA 设备上
inline bool is_cuda(const at::Tensor& tensor) {
    return tensor.device().type() == at::kCUDA;
}

at::Tensor ms_deform_attn_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step) 
{
    if (is_cuda(value))
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value, 
            spatial_shapes, 
            sampling_loc, 
            attn_weight, 
            im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> ms_deform_attn_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step) 
{
    if (is_cuda(value))
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(
            value, 
            spatial_shapes, 
            sampling_loc, 
            attn_weight, 
            grad_output, 
            im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}