#include <algorithm>
#include <cstring>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <THC/THCAtomics.cuh>

namespace rubiks {

template <typename T>
__global__ void rubiks_shift_3d_forward_cuda(const int total_num_elements, \
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
            bool quantize) {
    
    const int output_HW_dim = output_H_dim * output_W_dim;
    const int input_HW_dim = input_H_dim * input_W_dim;
    
    // Organization: grid -> block -> threads.
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
            index < total_num_elements; \
            index += blockDim.x * gridDim.x) {
        
        // Batch number of output
		const int N_idx = index / (output_T_dim * C_dim * output_HW_dim);
		const int within_N_idx = index % (output_T_dim * C_dim * output_HW_dim);
        
        // Timestep number
		const int output_T_idx = within_N_idx / (C_dim * output_HW_dim);
		const int within_T_idx = within_N_idx % (C_dim * output_HW_dim);
        
        // Channel number
		const int C_idx = within_T_idx / output_HW_dim;
		const int within_C_idx = within_T_idx % output_HW_dim;
        
        // Spatial index (where in the H, W grid you are)
		const int output_H_idx = within_C_idx / output_W_dim;
		const int output_W_idx = within_C_idx % output_W_dim;

        const int strided_output_T_idx = output_T_idx * stride_T - pad_T;
		const int strided_output_H_idx = output_H_idx * stride_H - pad_H;
		const int strided_output_W_idx = output_W_idx * stride_W - pad_W;

        // Takes the offsets from the corresponding channel
        const T shift_T = shift_tensor_data_T_ptr[C_idx];
        const T shift_H = shift_tensor_data_H_ptr[C_idx];
        const T shift_W = shift_tensor_data_W_ptr[C_idx];

        int input_T_idx, input_H_idx, input_W_idx;

        const int small_shift_T = floorf(shift_T);
        const int large_shift_T = small_shift_T + 1;
        const int small_shift_H = floorf(shift_H);
        const int large_shift_H = small_shift_H + 1;
        const int small_shift_W = floorf(shift_W);
        const int large_shift_W = small_shift_W + 1;

        T remainder_T = shift_T - small_shift_T;
        T remainder_W = shift_W - small_shift_W;
        T remainder_H = shift_H - small_shift_H;

        if (quantize) {
            int quantize_T = (remainder_T < 0.5f) ? small_shift_T : large_shift_T;
            int quantize_H = (remainder_H < 0.5f) ? small_shift_H : large_shift_H;
            int quantize_W = (remainder_W < 0.5f) ? small_shift_W : large_shift_W;

            input_T_idx = strided_output_T_idx + quantize_T;
            input_H_idx = strided_output_H_idx + quantize_H;
            input_W_idx = strided_output_W_idx + quantize_W;

            T q_quantize = (input_H_idx >= 0 && input_W_idx >= 0 && input_T_idx >= 0 && \
            input_H_idx < input_H_dim && input_W_idx < input_W_dim && input_T_idx < input_T_dim) ? \
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + \
            input_T_idx * C_dim * input_HW_dim + C_idx * input_HW_dim + \
            input_H_idx * input_W_dim + input_W_idx] : 0;

            output_tensor_data_ptr[index] = q_quantize;
            continue;  // skip the rest to do the next grid-stride loop
        }

        // ------------------- SMALL T, SMALL H, SMALL W -------------------
        
        input_T_idx = strided_output_T_idx + small_shift_T;
        input_H_idx = strided_output_H_idx + small_shift_H;
        input_W_idx = strided_output_W_idx + small_shift_W;

        // If we're in bounds, grab the data at that location. Otherwise, simply give zero.
        T q111 = (input_H_idx >= 0 && input_W_idx >= 0 && input_T_idx >= 0 && \
            input_H_idx < input_H_dim && input_W_idx < input_W_dim && input_T_idx < input_T_dim) ? \
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + \
            input_T_idx * C_dim * input_HW_dim + C_idx * input_HW_dim + \
            input_H_idx * input_W_dim + input_W_idx] : 0;
            
        // ------------------- SMALL T, SMALL H, LARGE W -------------------
        
        input_T_idx = strided_output_T_idx + small_shift_T;
        input_H_idx = strided_output_H_idx + small_shift_H;
        input_W_idx = strided_output_W_idx + large_shift_W;
        
        T q112 = (input_H_idx >= 0 && input_W_idx >= 0 && input_T_idx >= 0 && \
            input_H_idx < input_H_dim && input_W_idx < input_W_dim && input_T_idx < input_T_dim) ? \
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + \
            input_T_idx * C_dim * input_HW_dim + C_idx * input_HW_dim + \
            input_H_idx * input_W_dim + input_W_idx] : 0;
            
        // ------------------- SMALL T, LARGE H, SMALL W -------------------
        
        input_T_idx = strided_output_T_idx + small_shift_T;
        input_H_idx = strided_output_H_idx + large_shift_H;
        input_W_idx = strided_output_W_idx + small_shift_W;
        
        T q121 = (input_H_idx >= 0 && input_W_idx >= 0 && input_T_idx >= 0 && \
            input_H_idx < input_H_dim && input_W_idx < input_W_dim && input_T_idx < input_T_dim) ? \
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + \
            input_T_idx * C_dim * input_HW_dim + C_idx * input_HW_dim + \
            input_H_idx * input_W_dim + input_W_idx] : 0;
            
        // ------------------- SMALL T, LARGE H, LARGE W -------------------
        
        input_T_idx = strided_output_T_idx + small_shift_T;
        input_H_idx = strided_output_H_idx + large_shift_H;
        input_W_idx = strided_output_W_idx + large_shift_W;
        
        T q122 = (input_H_idx >= 0 && input_W_idx >= 0 && input_T_idx >= 0 && \
            input_H_idx < input_H_dim && input_W_idx < input_W_dim && input_T_idx < input_T_dim) ? \
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + \
            input_T_idx * C_dim * input_HW_dim + C_idx * input_HW_dim + \
            input_H_idx * input_W_dim + input_W_idx] : 0;

        // ------------------- LARGE T, SMALL H, SMALL W -------------------
        
        input_T_idx = strided_output_T_idx + large_shift_T;
        input_H_idx = strided_output_H_idx + small_shift_H;
        input_W_idx = strided_output_W_idx + small_shift_W;
        
        // If we're in bounds, grab the data at that location. Otherwise, simply give zero.
        T q211 = (input_H_idx >= 0 && input_W_idx >= 0 && input_T_idx >= 0 && \
            input_H_idx < input_H_dim && input_W_idx < input_W_dim && input_T_idx < input_T_dim) ? \
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + \
            input_T_idx * C_dim * input_HW_dim + C_idx * input_HW_dim + \
            input_H_idx * input_W_dim + input_W_idx] : 0;
            
        // ------------------- LARGE T, SMALL H, LARGE W -------------------
        
        input_T_idx = strided_output_T_idx + large_shift_T;
        input_H_idx = strided_output_H_idx + small_shift_H;
        input_W_idx = strided_output_W_idx + large_shift_W;
        
        T q212 = (input_H_idx >= 0 && input_W_idx >= 0 && input_T_idx >= 0 && \
            input_H_idx < input_H_dim && input_W_idx < input_W_dim && input_T_idx < input_T_dim) ? \
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + \
            input_T_idx * C_dim * input_HW_dim + C_idx * input_HW_dim + \
            input_H_idx * input_W_dim + input_W_idx] : 0;
            
        // ------------------- LARGE T, LARGE H, SMALL W -------------------
        
        input_T_idx = strided_output_T_idx + large_shift_T;
        input_H_idx = strided_output_H_idx + large_shift_H;
        input_W_idx = strided_output_W_idx + small_shift_W;
        
        T q221 = (input_H_idx >= 0 && input_W_idx >= 0 && input_T_idx >= 0 && \
            input_H_idx < input_H_dim && input_W_idx < input_W_dim && input_T_idx < input_T_dim) ? \
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + \
            input_T_idx * C_dim * input_HW_dim + C_idx * input_HW_dim + \
            input_H_idx * input_W_dim + input_W_idx] : 0;
            
        // ------------------- LARGE T, LARGE H, LARGE W -------------------
        
        input_T_idx = strided_output_T_idx + large_shift_T;
        input_H_idx = strided_output_H_idx + large_shift_H;
        input_W_idx = strided_output_W_idx + large_shift_W;
        
        T q222 = (input_H_idx >= 0 && input_W_idx >= 0 && input_T_idx >= 0 && \
            input_H_idx < input_H_dim && input_W_idx < input_W_dim && input_T_idx < input_T_dim) ? \
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + \
            input_T_idx * C_dim * input_HW_dim + C_idx * input_HW_dim + \
            input_H_idx * input_W_dim + input_W_idx] : 0;

        output_tensor_data_ptr[index] = \
        (1 - remainder_T) * \
            ((1 - remainder_H) * \
                (q111 * (1 - remainder_W) + q112 * remainder_W) + \
            remainder_H * \
                (q121 * (1 - remainder_W) + q122 * remainder_W)) + \
        remainder_T * \
            ((1 - remainder_H) * \
                (q211 * (1 - remainder_W) + q212 * remainder_W) + \
            remainder_H * \
                (q221 * (1 - remainder_W) + q222 * remainder_W));
	}
}


template <typename T>
__device__ T interpolate_2D(T p11, T p12, T p21, T p22, T delta_1, T delta_2) {
    
    return p11 * (1 - delta_1) * (1 - delta_2) + \
        p12 * (1 - delta_1) * delta_2 + \
        p21 * delta_1 * (1 - delta_2) + \
        p22 * delta_1 * delta_2;
}


template <typename T>
__global__ void rubiks_shift_3d_backward_cuda(
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
            T* shift_grad_buffer_W_start_ptr) {
        
    const int output_HW_dim = output_H_dim * output_W_dim;
    const int input_HW_dim = input_H_dim * input_W_dim;
        
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
            index < total_num_elements; \
            index += blockDim.x * gridDim.x) {
        
        // Again, batch index for output (NOT total N_number of batches)
		const int N_idx = index / (output_T_dim * C_dim * output_HW_dim);
		const int within_N_idx = index % (output_T_dim * C_dim * output_HW_dim);
        
        // Timestep index (NOT total number of T_dim)
		const int T_idx = within_N_idx / (C_dim * output_HW_dim);
		const int within_T_idx = within_N_idx % (C_dim * output_HW_dim);
        
        // Channel index (NOT total number of C_dim)
		const int C_idx = within_T_idx / output_HW_dim;
		const int within_C_idx = within_T_idx % output_HW_dim;
        
        // Height; width indices
		const int H_idx = within_C_idx / output_W_dim;
		const int W_idx = within_C_idx % output_W_dim;

        // Same thing as forward -- accounting for stride and padding
        const int T_offset = T_idx * stride_T - pad_T;
		const int H_offset = H_idx * stride_H - pad_H;
		const int W_offset = W_idx * stride_W - pad_W;

		// output: {3 x C, H x W}
        // Output into shift temp buffers so we can aggregate later.
        // This calculates the correct index within the temp buffer
		T* shift_grad_buffer_T_ptr = shift_grad_buffer_T_start_ptr + C_idx * output_HW_dim + within_C_idx;
		T* shift_grad_buffer_H_ptr = shift_grad_buffer_H_start_ptr + C_idx * output_HW_dim + within_C_idx;
		T* shift_grad_buffer_W_ptr = shift_grad_buffer_W_start_ptr + C_idx * output_HW_dim + within_C_idx;

        const T shift_T = shift_tensor_data_T_ptr[C_idx];
        const T shift_H = shift_tensor_data_H_ptr[C_idx];
        const T shift_W = shift_tensor_data_W_ptr[C_idx];

        // Computes four offsets and integer truncation differences
        const int small_shift_T = floorf(shift_T);
        const int large_shift_T = small_shift_T + 1;
        const int small_shift_H = floorf(shift_H);
        const int large_shift_H = small_shift_H + 1;
        const int small_shift_W = floorf(shift_W);
        const int large_shift_W = small_shift_W + 1;
        const T remainder_T = shift_T - small_shift_T;
        const T remainder_H = shift_H - small_shift_H;
        const T remainder_W = shift_W - small_shift_W;

        // Calculates actual indices which we grabbed from in the forward pass
        const int input_small_T = T_offset + small_shift_T;
        const int input_large_T = T_offset + large_shift_T;
        const int input_small_Ta = input_small_T + ((remainder_T == 0) ? -1 : 0);

        const int input_small_H = H_offset + small_shift_H;
        const int input_large_H = H_offset + large_shift_H;
        const int input_small_Ha = input_small_H + ((remainder_H == 0) ? -1 : 0);

        const int input_small_W = W_offset + small_shift_W;
        const int input_large_W = W_offset + large_shift_W;
        const int input_small_Wa = input_small_W + ((remainder_W == 0) ? -1 : 0);

        // -------------------------- SMALL T, SMALL H, SMALL W --------------------------
        
        const T q111 = (input_small_T >= 0 && input_small_H >= 0 && input_small_W >= 0 && \
            input_small_T < input_T_dim && input_small_H < input_H_dim && input_small_W < input_W_dim) ? 
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_small_T * C_dim * input_HW_dim + \
            C_idx * input_HW_dim + input_small_H * input_W_dim + input_small_W] : 0;
            
        // -------------------------- SMALL T, SMALL H, LARGE W --------------------------
        
        const T q112 = (input_small_T >= 0 && input_small_H >= 0 && input_large_W >= 0 && \
            input_small_T < input_T_dim && input_small_H < input_H_dim && input_large_W < input_W_dim) ? 
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_small_T * C_dim * input_HW_dim + \
            C_idx * input_HW_dim + input_small_H * input_W_dim + input_large_W] : 0;
            
        // -------------------------- SMALL T, LARGE H, SMALL W --------------------------
        
        const T q121 = (input_small_T >= 0 && input_large_H >= 0 && input_small_W >= 0 && \
            input_small_T < input_T_dim && input_large_H < input_H_dim && input_small_W < input_W_dim) ? 
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_small_T * C_dim * input_HW_dim + \
            C_idx * input_HW_dim + input_large_H * input_W_dim + input_small_W] : 0;
            
        // -------------------------- SMALL T, LARGE H, LARGE W --------------------------
        
        const T q122 = (input_small_T >= 0 && input_large_H >= 0 && input_large_W >= 0 && \
            input_small_T < input_T_dim && input_large_H < input_H_dim && input_large_W < input_W_dim) ? 
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_small_T * C_dim * input_HW_dim + \
            C_idx * input_HW_dim + input_large_H * input_W_dim + input_large_W] : 0;
            
        // -------------------------- LARGE T, SMALL H, SMALL W --------------------------
        
        const T q211 = (input_large_T >= 0 && input_small_H >= 0 && input_small_W >= 0 && \
            input_large_T < input_T_dim && input_small_H < input_H_dim && input_small_W < input_W_dim) ? 
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_large_T * C_dim * input_HW_dim + \
            C_idx * input_HW_dim + input_small_H * input_W_dim + input_small_W] : 0;
            
        // -------------------------- LARGE T, SMALL H, LARGE W --------------------------
        
        const T q212 = (input_large_T >= 0 && input_small_H >= 0 && input_large_W >= 0 && \
            input_large_T < input_T_dim && input_small_H < input_H_dim && input_large_W < input_W_dim) ? 
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_large_T * C_dim * input_HW_dim + \
            C_idx * input_HW_dim + input_small_H * input_W_dim + input_large_W] : 0;
            
        // -------------------------- LARGE T, LARGE H, SMALL W --------------------------
        
        const T q221 = (input_large_T >= 0 && input_large_H >= 0 && input_small_W >= 0 && \
            input_large_T < input_T_dim && input_large_H < input_H_dim && input_small_W < input_W_dim) ? 
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_large_T * C_dim * input_HW_dim + \
            C_idx * input_HW_dim + input_large_H * input_W_dim + input_small_W] : 0;
            
        // -------------------------- LARGE T, LARGE H, LARGE W --------------------------
        
        const T q222 = (input_large_T >= 0 && input_large_H >= 0 && input_large_W >= 0 && \
            input_large_T < input_T_dim && input_large_H < input_H_dim && input_large_W < input_W_dim) ? 
            input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_large_T * C_dim * input_HW_dim + \
            C_idx * input_HW_dim + input_large_H * input_W_dim + input_large_W] : 0;


        // -------------------------- SMALL T, SMALL H, SMALL W --------------------------
        // One non-ternary statement for clarity. The rest follow, but in ternary.
        T q111a = 0;
        // If it turns out our shifts are exact integers, then we need the smaller versions
        // of the shifted pixels.
        if (remainder_T == 0 || remainder_H == 0 || remainder_W == 0) {
            // Check if the modified starts are in bounds. 
            if (input_small_Ta >= 0 && input_small_Ha >= 0 && input_small_Wa >= 0 && \
                 input_small_Ta < input_T_dim && input_small_Ha < input_H_dim && input_small_Wa < input_W_dim) {
                q111a = input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_small_Ta * C_dim * input_HW_dim + \
                    C_idx * input_HW_dim + input_small_Ha * input_W_dim + input_small_Wa];
            } else {
                q111a = 0;
            }
        // Otherwise, we take what we've got from the regular interpolation formula.
        } else {
            q111a = q111;
        }

        // -------------------------- SMALL T, SMALL H, LARGE W --------------------------
        const T q112a = (remainder_T == 0 || remainder_H == 0) ? 
            ((input_small_Ta >= 0 && input_small_Ha >= 0 && input_large_W >= 0 && \
                 input_small_Ta < input_T_dim && input_small_Ha < input_H_dim && input_large_W < input_W_dim) ?
                input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_small_Ta * C_dim * input_HW_dim + \
                    C_idx * input_HW_dim + input_small_Ha * input_W_dim + input_large_W]
            : 0)
        :q112;

        // -------------------------- SMALL T, LARGE H, SMALL W --------------------------
        const T q121a = (remainder_T == 0 || remainder_W == 0) ? 
            ((input_small_Ta >= 0 && input_large_H >= 0 && input_small_Wa >= 0 && \
                 input_small_Ta < input_T_dim && input_large_H < input_H_dim && input_small_Wa < input_W_dim) ?
                input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_small_Ta * C_dim * input_HW_dim + \
                    C_idx * input_HW_dim + input_large_H * input_W_dim + input_small_Wa]
            : 0)
        :q121;
        
        // -------------------------- SMALL T, LARGE H, LARGE W --------------------------
        const T q122a = (remainder_T == 0) ? 
            ((input_small_Ta >= 0 && input_large_H >= 0 && input_large_W >= 0 && \
                 input_small_Ta < input_T_dim && input_large_H < input_H_dim && input_large_W < input_W_dim) ?
                input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_small_Ta * C_dim * input_HW_dim + \
                    C_idx * input_HW_dim + input_large_H * input_W_dim + input_large_W]
            : 0)
        :q122;

        // -------------------------- LARGE T, SMALL H, SMALL W --------------------------
        const T q211a = (remainder_H == 0 || remainder_W == 0) ? 
            ((input_large_T >= 0 && input_small_Ha >= 0 && input_small_Wa >= 0 && \
                 input_large_T < input_T_dim && input_small_Ha < input_H_dim && input_small_Wa < input_W_dim) ?
                input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_large_T * C_dim * input_HW_dim + \
                    C_idx * input_HW_dim + input_small_Ha * input_W_dim + input_small_Wa]
            : 0)
        :q211;
        
        // -------------------------- LARGE T, SMALL H, LARGE W --------------------------
        const T q212a = (remainder_H == 0) ? 
            ((input_large_T >= 0 && input_small_Ha >= 0 && input_large_W >= 0 && \
                 input_large_T < input_T_dim && input_small_Ha < input_H_dim && input_large_W < input_W_dim) ?
                input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_large_T * C_dim * input_HW_dim + \
                    C_idx * input_HW_dim + input_small_Ha * input_W_dim + input_large_W]
            : 0)
        :q212;

        // -------------------------- LARGE T, LARGE H, SMALL W --------------------------
        const T q221a = (remainder_W == 0) ? 
            ((input_large_T >= 0 && input_large_H >= 0 && input_small_Wa >= 0 && \
                 input_large_T < input_T_dim && input_large_H < input_H_dim && input_small_Wa < input_W_dim) ?
                input_tensor_data_ptr[N_idx * input_T_dim * C_dim * input_HW_dim + input_large_T * C_dim * input_HW_dim + \
                    C_idx * input_HW_dim + input_large_H * input_W_dim + input_small_Wa]
            : 0)
        :q221;
        
        // -------------------------- LARGE T, LARGE H, LARGE W --------------------------
        const T q222a = q222;
        const T local_T_grad_small = interpolate_2D<T>(q111a, q112a, q121a, q122a, remainder_H, remainder_W);
        const T local_T_grad_large = interpolate_2D<T>(q211a, q212a, q221a, q222a, remainder_H, remainder_W);
        const T local_H_grad_small = interpolate_2D<T>(q111a, q112a, q211a, q212a, remainder_T, remainder_W);
        const T local_H_grad_large = interpolate_2D<T>(q121a, q122a, q221a, q222a, remainder_T, remainder_W);
        const T local_W_grad_small = interpolate_2D<T>(q111a, q121a, q211a, q221a, remainder_T, remainder_H);
        const T local_W_grad_large = interpolate_2D<T>(q112a, q122a, q212a, q222a, remainder_T, remainder_H);
        
        const T local_pixel_T_grad = -local_T_grad_small + local_T_grad_large;
        const T local_pixel_H_grad = -local_H_grad_small + local_H_grad_large;
        const T local_pixel_W_grad = -local_W_grad_small + local_W_grad_large;

        const T upstream_grad = output_grad_data_ptr[index];
        const T pixel_T_grad = local_pixel_T_grad * upstream_grad;
        const T pixel_H_grad = local_pixel_H_grad * upstream_grad;
        const T pixel_W_grad = local_pixel_W_grad * upstream_grad;

        atomicAdd(shift_grad_buffer_T_ptr, pixel_T_grad);
        atomicAdd(shift_grad_buffer_H_ptr, pixel_H_grad);
        atomicAdd(shift_grad_buffer_W_ptr, pixel_W_grad);
	}
}


template <typename T>
__global__ void rubiks_shift_3d_backward_input_cuda(\
    const int total_num_elements,
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
    bool quantize) {
    
    const int output_HW_dim = output_H_dim * output_W_dim;
    const int input_HW_dim = input_H_dim * input_W_dim;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
            index < total_num_elements; \
            index += blockDim.x * gridDim.x) {

        // Batch index (NOT total number of batches)
        const int N_idx = index / (input_T_dim * C_dim * input_HW_dim);
        const int within_N_idx = index % (input_T_dim * C_dim * input_HW_dim);

        // Timestep index (NOT total number of timesteps)
        const int T_idx = within_N_idx / (C_dim * input_HW_dim);
        const int within_T_idx = within_N_idx % (C_dim * input_HW_dim);

        // Channel index (NOT total number of channels)
        const int C_idx = within_T_idx / input_HW_dim;
        const int within_C_idx = within_T_idx % input_HW_dim;

        // Height and width spatial location which this particular thread handles
        const int H_idx = within_C_idx / input_W_dim;
        const int W_idx = within_C_idx % input_W_dim;

        // Offsets within the (H, W) feature map in the output gradient tensor to pull from
        // (note that backward input gradient is just output gradient, reverse shifted)
        const int T_offset = T_idx + pad_T;
        const int H_offset = H_idx + pad_H;
        const int W_offset = W_idx + pad_W;

        // Final value to be stuck into the input gradient
        T val = 0;

        const T shift_T = -shift_tensor_data_T_ptr[C_idx];
        const T shift_H = -shift_tensor_data_H_ptr[C_idx];
        const T shift_W = -shift_tensor_data_W_ptr[C_idx];

        // Where in the actual (H, W) feature map we pull from.
        int output_T_idx, output_H_idx, output_W_idx;
        T q111 = 0;
        T q112 = 0;
        T q121 = 0;
        T q122 = 0;
        T q211 = 0;
        T q212 = 0;
        T q221 = 0;
        T q222 = 0;
        
        int small_shift_T = floorf(shift_T);
        int large_shift_T = small_shift_T + 1;
        int small_shift_H = floorf(shift_H);
        int large_shift_H = small_shift_H + 1;
        int small_shift_W = floorf(shift_W);
        int large_shift_W = small_shift_W + 1;

        // Compute interpolation remainders (e.g. 1.4 - 1 = 0.4)
        T remainder_T = shift_T - small_shift_T;
        T remainder_H = shift_H - small_shift_H;
        T remainder_W = shift_W - small_shift_W;


        if (quantize) {
            int quantize_T = (remainder_T < 0.5f) ? small_shift_T : large_shift_T;
            int quantize_H = (remainder_H < 0.5f) ? small_shift_H : large_shift_H;
            int quantize_W = (remainder_W < 0.5f) ? small_shift_W : large_shift_W;


            output_T_idx = (T_offset + quantize_T);
            output_H_idx = (H_offset + quantize_H);
            output_W_idx = (W_offset + quantize_W);

            T q_quantize = 0.f;

            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                q_quantize = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }

            input_grad_data_ptr[index] = q_quantize;
            continue;  // skip the rest to do the next grid-stride loop
        }

        // Special case -- all shifts are zero; only care about strides and padding with NO interpolation.
        if (shift_T == 0 && shift_H == 0 && shift_W == 0) {
            output_T_idx = T_offset;
            output_H_idx = H_offset;
            output_W_idx = W_offset;

            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
            
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                val = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }

        } else {

            // -------------------------- SMALL T, SMALL H, SMALL W --------------------------

            output_T_idx = (T_offset + small_shift_T);
            output_H_idx = (H_offset + small_shift_H);
            output_W_idx = (W_offset + small_shift_W);
            
            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                q111 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }
            // -------------------------- SMALL T, SMALL H, LARGE W --------------------------

            output_T_idx = (T_offset + small_shift_T);
            output_H_idx = (H_offset + small_shift_H);
            output_W_idx = (W_offset + large_shift_W);
            
            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                q112 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }
            // -------------------------- SMALL T, LARGE H, SMALL W --------------------------

            output_T_idx = (T_offset + small_shift_T);
            output_H_idx = (H_offset + large_shift_H);
            output_W_idx = (W_offset + small_shift_W);

            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                q121 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }
            // -------------------------- SMALL T, LARGE H, LARGE W --------------------------

            output_T_idx = (T_offset + small_shift_T);
            output_H_idx = (H_offset + large_shift_H);
            output_W_idx = (W_offset + large_shift_W);

            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                q122 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }
            // -------------------------- LARGE T, SMALL H, SMALL W --------------------------

            output_T_idx = (T_offset + large_shift_T);
            output_H_idx = (H_offset + small_shift_H);
            output_W_idx = (W_offset + small_shift_W);
            
            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                q211 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }
            // -------------------------- LARGE T, SMALL H, LARGE W --------------------------

            output_T_idx = (T_offset + large_shift_T);
            output_H_idx = (H_offset + small_shift_H);
            output_W_idx = (W_offset + large_shift_W);
            
            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                q212 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }
            // -------------------------- LARGE T, LARGE H, SMALL W --------------------------

            output_T_idx = (T_offset + large_shift_T);
            output_H_idx = (H_offset + large_shift_H);
            output_W_idx = (W_offset + small_shift_W);

            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                q221 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }
            // -------------------------- LARGE T, LARGE H, LARGE W --------------------------

            output_T_idx = (T_offset + large_shift_T);
            output_H_idx = (H_offset + large_shift_H);
            output_W_idx = (W_offset + large_shift_W);

            if (output_T_idx % stride_T == 0 && output_H_idx % stride_H == 0 && output_W_idx % stride_W == 0) {
                output_T_idx = output_T_idx / stride_T;
                output_H_idx = output_H_idx / stride_H;
                output_W_idx = output_W_idx / stride_W;

                q222 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            }

            val = \
                (1 - remainder_T) * \
                    ((1 - remainder_H) * \
                        (q111 * (1 - remainder_W) + q112 * remainder_W) + \
                    remainder_H * \
                        (q121 * (1 - remainder_W) + q122 * remainder_W)) + \
                remainder_T * \
                    ((1 - remainder_H) * \
                        (q211 * (1 - remainder_W) + q212 * remainder_W) + \
                    remainder_H * \
                        (q221 * (1 - remainder_W) + q222 * remainder_W));
        }
        input_grad_data_ptr[index] = val;
    }
}


template <typename T>
__global__ void rubiks_shift_3d_backward_input_s1p0_cuda(\
    const int total_num_elements,
    const int N_dim, const int input_T_dim, \
    const int output_T_dim, const int C_dim, \
    const int input_H_dim, const int output_H_dim, \
    const int input_W_dim, const int output_W_dim, \
    const T* shift_tensor_data_T_ptr, \
    const T* shift_tensor_data_H_ptr, \
    const T* shift_tensor_data_W_ptr, \
    const T* input_tensor_data_ptr, \
    const T* output_grad_data_ptr, \
    T* input_grad_data_ptr,
    bool quantize) {

    const int output_HW_dim = output_H_dim * output_W_dim;
    const int input_HW_dim = input_H_dim * input_W_dim;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
            index < total_num_elements; \
            index += blockDim.x * gridDim.x) {

        // Batch index (NOT total number of batches)
        const int N_idx = index / (input_T_dim * C_dim * input_HW_dim);
        const int within_N_idx = index % (input_T_dim * C_dim * input_HW_dim);

        // Timestep index (NOT total number of timesteps)
        const int T_idx = within_N_idx / (C_dim * input_HW_dim);
        const int within_T_idx = within_N_idx % (C_dim * input_HW_dim);

        // Channel index (NOT total number of channels)
        const int C_idx = within_T_idx / input_HW_dim;
        const int within_C_idx = within_T_idx % input_HW_dim;

        // Height and width spatial location which this particular thread handles
        const int H_idx = within_C_idx / input_W_dim;
        const int W_idx = within_C_idx % input_W_dim;

        // Offsets within the (H, W) feature map in the output gradient tensor to pull from
        // (note that backward input gradient is just output gradient, reverse shifted)
        const int T_offset = T_idx;
        const int H_offset = H_idx;
        const int W_offset = W_idx;

        T val = 0;

        const T shift_T = -shift_tensor_data_T_ptr[C_idx];
        const T shift_H = -shift_tensor_data_H_ptr[C_idx];
        const T shift_W = -shift_tensor_data_W_ptr[C_idx];

        // Where in the actual (H, W) feature map we pull from.
        int output_T_idx, output_H_idx, output_W_idx;
        T q111 = 0;
        T q112 = 0;
        T q121 = 0;
        T q122 = 0;
        T q211 = 0;
        T q212 = 0;
        T q221 = 0;
        T q222 = 0;

        int small_shift_T = floorf(shift_T);
        int large_shift_T = small_shift_T + 1;
        int small_shift_H = floorf(shift_H);
        int large_shift_H = small_shift_H + 1;
        int small_shift_W = floorf(shift_W);
        int large_shift_W = small_shift_W + 1;

        // Compute interpolation remainders (e.g. 1.4 - 1 = 0.4)
        T remainder_T = shift_T - small_shift_T;
        T remainder_H = shift_H - small_shift_H;
        T remainder_W = shift_W - small_shift_W;


        if (quantize) {
            int quantize_T = (remainder_T < 0.5f) ? small_shift_T : large_shift_T;
            int quantize_H = (remainder_H < 0.5f) ? small_shift_H : large_shift_H;
            int quantize_W = (remainder_W < 0.5f) ? small_shift_W : large_shift_W;

            output_T_idx = (T_offset + quantize_T);
            output_H_idx = (H_offset + quantize_H);
            output_W_idx = (W_offset + quantize_W);

            T q_quantize = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;

            input_grad_data_ptr[index] = q_quantize;
            continue;  // skip the rest to do the next grid-stride loop
        }

        // Special case -- all shifts are zero; only care about strides and padding with NO interpolation.
        if (shift_T == 0 && shift_H == 0 && shift_W == 0) {
            output_T_idx = T_offset;
            output_H_idx = H_offset;
            output_W_idx = W_offset;

            val = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
        } else {

            // -------------------------- SMALL T, SMALL H, SMALL W --------------------------

            output_T_idx = (T_offset + small_shift_T);
            output_H_idx = (H_offset + small_shift_H);
            output_W_idx = (W_offset + small_shift_W);

            q111 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                    output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                    C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;

            // -------------------------- SMALL T, SMALL H, LARGE W --------------------------

            output_T_idx = (T_offset + small_shift_T);
            output_H_idx = (H_offset + small_shift_H);
            output_W_idx = (W_offset + large_shift_W);

            q112 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            // -------------------------- SMALL T, LARGE H, SMALL W --------------------------

            output_T_idx = (T_offset + small_shift_T);
            output_H_idx = (H_offset + large_shift_H);
            output_W_idx = (W_offset + small_shift_W);

            q121 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            // -------------------------- SMALL T, LARGE H, LARGE W --------------------------

            output_T_idx = (T_offset + small_shift_T);
            output_H_idx = (H_offset + large_shift_H);
            output_W_idx = (W_offset + large_shift_W);

            q122 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            // -------------------------- LARGE T, SMALL H, SMALL W --------------------------

            output_T_idx = (T_offset + large_shift_T);
            output_H_idx = (H_offset + small_shift_H);
            output_W_idx = (W_offset + small_shift_W);

            q211 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            // -------------------------- LARGE T, SMALL H, LARGE W --------------------------

            output_T_idx = (T_offset + large_shift_T);
            output_H_idx = (H_offset + small_shift_H);
            output_W_idx = (W_offset + large_shift_W);

            q212 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            // -------------------------- LARGE T, LARGE H, SMALL W --------------------------

            output_T_idx = (T_offset + large_shift_T);
            output_H_idx = (H_offset + large_shift_H);
            output_W_idx = (W_offset + small_shift_W);

            q221 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;
            // -------------------------- LARGE T, LARGE H, LARGE W --------------------------

            output_T_idx = (T_offset + large_shift_T);
            output_H_idx = (H_offset + large_shift_H);
            output_W_idx = (W_offset + large_shift_W);

            q222 = (output_H_idx >= 0 && output_W_idx >= 0 && output_T_idx >= 0 && \
                output_H_idx < output_H_dim && output_W_idx < output_W_dim && output_T_idx < output_T_dim) ? \
                output_grad_data_ptr[N_idx * output_T_dim * C_dim * output_HW_dim + output_T_idx * C_dim * output_HW_dim + \
                C_idx * output_HW_dim + output_H_idx * output_W_dim + output_W_idx] : 0;

            // Perform interpolation (draw a picture for yourself to clarify -- the q's are the four corner
            // points of the box.)
            val = \
                (1 - remainder_T) * \
                    ((1 - remainder_H) * \
                        (q111 * (1 - remainder_W) + q112 * remainder_W) + \
                    remainder_H * \
                        (q121 * (1 - remainder_W) + q122 * remainder_W)) + \
                remainder_T * \
                    ((1 - remainder_H) * \
                        (q211 * (1 - remainder_W) + q212 * remainder_W) + \
                    remainder_H * \
                        (q221 * (1 - remainder_W) + q222 * remainder_W));
        }
        // Finally, stick the correct value into the correct location in the input grad tensor.
        input_grad_data_ptr[index] = val;
    }
}

    
template <typename T>
__global__ void normalize_shift_grad_3d_cuda(const int C_dim, \
    T* shift_grad_T_data_ptr, T* shift_grad_H_data_ptr, T* shift_grad_W_data_ptr,
    const T normalize_t_factor) {
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
        index < C_dim; \
        index += blockDim.x * gridDim.x) {

        T cur_T_grad, cur_H_grad, cur_W_grad;
        if (normalize_t_factor < 0) {
            cur_T_grad = shift_grad_T_data_ptr[index];
            cur_H_grad = 0;
            cur_W_grad = 0;
        }
        else {
            cur_T_grad = shift_grad_T_data_ptr[index] * normalize_t_factor;
            cur_H_grad = shift_grad_H_data_ptr[index];
            cur_W_grad = shift_grad_W_data_ptr[index];
        }
        const T magnitude = sqrt(cur_T_grad * cur_T_grad + cur_H_grad * cur_H_grad + cur_W_grad * cur_W_grad);
        
        if (magnitude > 0) {
            shift_grad_T_data_ptr[index] = cur_T_grad / magnitude;
            shift_grad_H_data_ptr[index] = cur_H_grad / magnitude;
            shift_grad_W_data_ptr[index] = cur_W_grad / magnitude;
        }
    }
}


#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

// Populates block_count and thread_per_block with properties of CUDA Device
// Remark: the argument "count" is inferred as virtual_thread_count
void get_cuda_device_properties(const c10::Device& d, const int count, \
	int& block_count, int& thread_per_block) {
	int max_thread_per_block = 0;
	cudaErrorCheck( cudaDeviceGetAttribute(&max_thread_per_block, \
		cudaDevAttrMaxThreadsPerBlock, 0) );
			
	int max_thread_per_multiprocessor = 0;
	cudaErrorCheck( cudaDeviceGetAttribute(&max_thread_per_multiprocessor, \
		cudaDevAttrMaxThreadsPerMultiProcessor, 0) );

	int num_multiprocessor = 0;
	cudaErrorCheck( cudaDeviceGetAttribute(&num_multiprocessor, \
		cudaDevAttrMultiProcessorCount, 0) );

	const int virtual_thread_count = count;
	const int physical_thread_count = 
		std::min(num_multiprocessor * max_thread_per_multiprocessor, 
			virtual_thread_count);
			
	thread_per_block = std::min(1024, max_thread_per_block);
	block_count = 
		std::min((physical_thread_count + thread_per_block - 1) / thread_per_block, \
			num_multiprocessor);
}


    
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
            bool quantize) {
    
		// Sets up CUDA multi-threading
		int block_count = 0;
		int thread_per_block = 0;
		get_cuda_device_properties(d, total_num_elements, block_count, thread_per_block);
            
        rubiks_shift_3d_forward_cuda<T>
        <<<block_count, thread_per_block>>>(
            total_num_elements, N_dim, \
            input_T_dim, output_T_dim, C_dim, \
            input_H_dim, output_H_dim, \
            input_W_dim, output_W_dim, \
            shift_tensor_data_T_ptr, \
            shift_tensor_data_H_ptr, \
            shift_tensor_data_W_ptr, \
            pad_T, pad_H, pad_W, \
            stride_T, stride_H, stride_W, \
            input_tensor_data_ptr, output_tensor_data_ptr, quantize);
    }
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
            T* shift_grad_buffer_W_start_ptr) {

		// Sets up CUDA multi-threading
		int block_count = 0;
		int thread_per_block = 0;
		get_cuda_device_properties(d, total_num_elements, block_count, thread_per_block);
    
        // Invoke GPU kernel
		rubiks_shift_3d_backward_cuda<T>
		<<<block_count, thread_per_block>>>(
            total_num_elements, N_dim, \
            input_T_dim, output_T_dim, C_dim, \
            input_H_dim, output_H_dim, \
            input_W_dim, output_W_dim, \
            shift_tensor_data_T_ptr, \
            shift_tensor_data_H_ptr, \
            shift_tensor_data_W_ptr, \
            pad_T, pad_H, pad_W, \
            stride_T, stride_H, stride_W, \
            input_tensor_data_ptr, \
            output_grad_data_ptr, \
            shift_grad_buffer_T_start_ptr, \
            shift_grad_buffer_H_start_ptr, \
            shift_grad_buffer_W_start_ptr);
	}
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
            bool quantize) {
        
		// Sets up CUDA multi-threading
		int block_count = 0;
		int thread_per_block = 0;
		get_cuda_device_properties(d, total_num_elements, block_count, thread_per_block);

		if (stride_T == 1 && stride_H == 1 && stride_W == 1
		    && pad_T == 0 && pad_H == 0 && pad_W == 0) {
            rubiks_shift_3d_backward_input_s1p0_cuda<T>
                <<<block_count, thread_per_block>>>(
                total_num_elements, N_dim, \
            input_T_dim, output_T_dim, C_dim, \
            input_H_dim, output_H_dim, \
            input_W_dim, output_W_dim, \
            shift_tensor_data_T_ptr, \
            shift_tensor_data_H_ptr, \
            shift_tensor_data_W_ptr, \
            input_tensor_data_ptr, \
            output_grad_data_ptr, \
            input_grad_data_ptr,
            quantize);
		}
		else {
            rubiks_shift_3d_backward_input_cuda<T>
                <<<block_count, thread_per_block>>>(
                total_num_elements, N_dim, \
            input_T_dim, output_T_dim, C_dim, \
            input_H_dim, output_H_dim, \
            input_W_dim, output_W_dim, \
            shift_tensor_data_T_ptr, \
            shift_tensor_data_H_ptr, \
            shift_tensor_data_W_ptr, \
            pad_T, pad_H, pad_W, \
            stride_T, stride_H, stride_W, \
            input_tensor_data_ptr, \
            output_grad_data_ptr, \
            input_grad_data_ptr,
            quantize);
		}
	}
};



template <typename T>
struct NormalizeShiftGrad3D {

    void operator()(const c10::Device& d, \
                    const int C_dim, \
                    T* shift_grad_T_data_ptr, \
                    T* shift_grad_H_data_ptr, \
                    T* shift_grad_W_data_ptr,
                    const T normalize_t_factor) {

		int block_count = 0;
		int thread_per_block = 0;
		get_cuda_device_properties(d, C_dim, block_count, thread_per_block);

		normalize_shift_grad_3d_cuda<T>
		<<<block_count, thread_per_block>>>(
            C_dim, shift_grad_T_data_ptr, \
            shift_grad_H_data_ptr, \
            shift_grad_W_data_ptr,
            normalize_t_factor);
                
    }
};

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#undef CUDA_CHECK
#undef cudaErrorCheck

#define STRUCTS(T) \
    template struct RubiksShift3DForward<T>; \
    template struct RubiksShift3DBackward<T>; \
    template struct RubiksShift3DBackwardInput<T>; \
    template struct NormalizeShiftGrad3D<T>;

STRUCTS(float);
STRUCTS(double);

#undef STRUCTS

} // namespace rubiks_shift