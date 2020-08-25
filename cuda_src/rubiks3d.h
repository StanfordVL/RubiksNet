#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/core/Tensor.h>

#ifndef RUBIKS_SHIFT_H_
#define RUBIKS_SHIFT_H_

namespace rubiks {

template <typename T>
struct RubiksShift3DForward {
    
    void operator()(const c10::Device& d, \
            const int total_num_elements, \
            const int N_dim, const int input_T_dim, \
            const int output_T_dim, const int C_dim, \
            const int input_H_dim, const int output_H_dim, \
            const int input_W_dim, const int output_W_dim, \
            const T* shift_tensor_data_T_ptr, \
            const T* shift_tensor_data_H_ptr, \
            const T* shift_tensor_data_W_ptr, \
            const int pad_T, const int pad_H, \
            const int pad_W, const int stride_T, \
            const int stride_H, const int stride_W, \
            const T* input_tensor_data_ptr, T* output_tensor_data_ptr,
            bool quantize);
};


template <typename T>
struct RubiksShift3DBackward {
    void operator()(const c10::Device& d, \
            const int total_num_elements, \
            const int N_dim, const int input_T_dim, \
            const int output_T_dim, const int C_dim, \
            const int input_H_dim, const int output_H_dim, \
            const int input_W_dim, const int output_W_dim, \
            const T* shift_tensor_data_T_ptr, \
            const T* shift_tensor_data_H_ptr, \
            const T* shift_tensor_data_W_ptr, \
            const int pad_T, const int pad_H, \
            const int pad_W, const int stride_T, \
            const int stride_H, const int stride_W, \
            const T* input_tensor_data_ptr, \
            const T* output_grad_data_ptr, \
            T* shift_grad_buffer_T_start_ptr, \
            T* shift_grad_buffer_H_start_ptr, \
            T* shift_grad_buffer_W_start_ptr);
};
    

template <typename T>
struct RubiksShift3DBackwardInput {
    
    void operator()(const c10::Device& d, \
            const int total_num_elements, \
            const int N_dim, const int input_T_dim, \
            const int output_T_dim, const int C_dim, \
            const int input_H_dim, const int output_H_dim, \
            const int input_W_dim, const int output_W_dim, \
            const T* shift_tensor_data_T_ptr, \
            const T* shift_tensor_data_H_ptr, \
            const T* shift_tensor_data_W_ptr, \
            const int pad_T, const int pad_H, \
            const int pad_W, const int stride_T, \
            const int stride_H, const int stride_W, \
            const T* input_tensor_data_ptr, \
            const T* output_grad_data_ptr, \
            T* input_grad_data_ptr,
            bool quantize);
};
    
template <typename T>
struct NormalizeShiftGrad3D {

    void operator()(const c10::Device& d, \
                    const int C_dim, \
                    T* shift_grad_T_data_ptr, \
                    T* shift_grad_H_data_ptr, \
                    T* shift_grad_W_data_ptr,
                    const T normalize_t_factor);
};


}

#endif
