#include "utils_cuda.h"
using namespace torchx;

namespace rubiks {

// kernel function argument list
#define DECLARE_KERNEL_DIM_ARGS \
    const uint32_t input_numel,  \
    const uint32_t input_H_dim,  \
    const uint32_t input_W_dim,  \
    const uint32_t input_HW_dim,  \
    const uint32_t input_CHW_dim,  \
    const uint32_t output_numel,  \
    const uint32_t output_H_dim,  \
    const uint32_t output_W_dim,  \
    const uint32_t output_HW_dim,  \
    const uint32_t output_CHW_dim,  \
    const uint32_t C_dim
    
// pass to kernel function from caller
// order must match DECLARE_KERNEL_DIM_ARGS
#define PASS_KERNEL_DIM_ARGS  \
    input_numel,  \
    input_H_dim,  \
    input_W_dim,  \
    input_HW_dim,  \
    input_CHW_dim,  \
    output_numel,  \
    output_H_dim,  \
    output_W_dim,  \
    output_HW_dim,  \
    output_CHW_dim,  \
    C_dim

// calculate in caller function and pass as args to kernel
#define CALCULATE_DIM_ARGS(input, output) \
    const uint32_t C_dim = input.size(1); \
    const uint32_t input_numel = input.numel();  \
    const uint32_t input_H_dim = input.size(2);  \
    const uint32_t input_W_dim = input.size(3);  \
    const uint32_t input_HW_dim = input_H_dim * input_W_dim;  \
    const uint32_t input_CHW_dim = input_HW_dim * C_dim;  \
    const uint32_t output_numel = output.numel();  \
    const uint32_t output_H_dim = output.size(2);  \
    const uint32_t output_W_dim = output.size(3);  \
    const uint32_t output_HW_dim = output_H_dim * output_W_dim;  \
    const uint32_t output_CHW_dim = output_HW_dim * C_dim;

// calculate the "unflattened" 4D index inside kernel's grid-stride loop
// `place` should be either "input" or "output"
#define GET_INDICES(place, N_idx, C_idx, H_idx, W_idx) \
    const uint32_t N_idx = index / place##_CHW_dim;  \
    const uint32_t within_N_idx = index % place##_CHW_dim;  \
    const uint32_t C_idx = within_N_idx / place##_HW_dim;  \
    const uint32_t within_C_idx = within_N_idx % place##_HW_dim;  \
    const uint32_t H_idx = within_C_idx / place##_W_dim;  \
    const uint32_t W_idx = within_C_idx % place##_W_dim;


template<typename T>
__device__ __forceinline__ T interpolate_2d(T pixels[][2], T remainder_H, T remainder_W) {
    return pixels[0][0] * (1 - remainder_H) * (1 - remainder_W)
        + pixels[0][1] * (1 - remainder_H) * remainder_W
        + pixels[1][0] * remainder_H * (1 - remainder_W)
        + pixels[1][1] * remainder_H * remainder_W;
}


template<typename T>
__device__ __forceinline__ int floor_fast(T x) {
    int ix = (int) x;
    return ix - (x < ix);
}


template<typename T>
__device__ __forceinline__ int round_fast(T x) {
    if (x < static_cast<T>(0.0f))
        return (int)(x - static_cast<T>(0.5f));
    else
        return (int)(x + static_cast<T>(0.5f));
}


#define WITHIN_BOUND(H, W) (H >= 0 && W >= 0 && H < input_H_dim && W < input_W_dim)

// set a value if (H, W) are within bounds
#define SET_WITHIN_BOUND(val, H, W) \
    if (WITHIN_BOUND(H, W)) { \
        val = input[N_idx][C_idx][H][W]; \
    }


template<typename T>
__global__ void rubiks2d_forward_kernel(
    const PTA<T, 4> input,
    const PTA<T, 2> shift_field,
    const uint32_t stride_H, const uint32_t stride_W,
    const uint32_t pad_H, const uint32_t pad_W,
    bool quantize,
    PTA<T, 4> output,
    DECLARE_KERNEL_DIM_ARGS
) {
    GRID_STRIDE_LOOP(index, output_numel) {
        GET_INDICES(output, N_idx, C_idx, H_idx, W_idx);
        const int strided_H_idx = H_idx * stride_H - pad_H;
        const int strided_W_idx = W_idx * stride_W - pad_W;

        // TODO add this guard back
//        if (!WITHIN_BOUND(strided_H_idx, strided_W_idx))
//            continue;

        T H_offset = shift_field[0][C_idx];
        T W_offset = shift_field[1][C_idx];

        if (quantize) {
            int temp_H = round_fast(strided_H_idx + H_offset);
            int temp_W = round_fast(strided_W_idx + W_offset);
            SET_WITHIN_BOUND(output[N_idx][C_idx][H_idx][W_idx], temp_H, temp_W);
            continue;  // skip the rest
        }

        const int H_offset_int = floor_fast(H_offset);
        const int W_offset_int = floor_fast(W_offset);
        const T remainder_H = H_offset - H_offset_int;
        const T remainder_W = W_offset - W_offset_int;

        int input_H_idx = strided_H_idx + H_offset_int;
        int input_W_idx = strided_W_idx + W_offset_int;

        T pixels[2][2] = {0};
        int temp_H, temp_W;

        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                temp_H = input_H_idx + h;
                temp_W = input_W_idx + w;
                SET_WITHIN_BOUND(pixels[h][w], temp_H, temp_W);
            }
        }
        // Just interpolate between the four pixels and stick the result in the right output spot.
        output[N_idx][C_idx][H_idx][W_idx] =
            interpolate_2d(pixels, remainder_H, remainder_W);
    }
}

template <typename T>
__global__ void rubiks2d_backward_shift_kernel(
    PTA<T, 4> output_grad,
    const PTA<T, 4> input,
    const PTA<T, 2> shift_field,
    const uint32_t stride_H, const uint32_t stride_W,
    const uint32_t pad_H, const uint32_t pad_W,
    PTA<T, 4> shift_grad_buffer,
    DECLARE_KERNEL_DIM_ARGS
) {
    GRID_STRIDE_LOOP(index, output_numel) {
        GET_INDICES(output, N_idx, C_idx, H_idx, W_idx);
        // Same thing as forward -- accounting for stride and padding
        const int strided_H_idx = H_idx * stride_H - pad_H;
        const int strided_W_idx = W_idx * stride_W - pad_W;

        // TODO add this guard back
//        if (!WITHIN_BOUND(strided_H_idx, strided_W_idx))
//            continue;

        T H_offset = shift_field[0][C_idx];
        T W_offset = shift_field[1][C_idx];

        // Local gradient values to be multiplied by upstream gradient and placed into correct tensor location.
        T local_pixel_W_grad = 0, local_pixel_H_grad = 0;

        const int H_offset_int = floor_fast(H_offset);
        const int W_offset_int = floor_fast(W_offset);

        int input_H_idx = strided_H_idx + H_offset_int;
        int input_W_idx = strided_W_idx + W_offset_int;

        T remainder_H = H_offset - H_offset_int;
        T remainder_W = W_offset - W_offset_int;

#ifdef _DEBUG_GRADIENT_
        if (remainder_H < 1e-7 || remainder_W < 1e-7) {
            printf("small remainder H = %.3e    W = %.3e\n",
                   static_cast<float>(remainder_H), static_cast<float>(remainder_W));
        }
#endif

        T ZERO_TOL = static_cast<T>(1e-7f);

        bool is_h_int_shift = false;
        bool is_w_int_shift = false;
        if (ZERO_TOL > remainder_H && remainder_H > -ZERO_TOL) {
            is_h_int_shift = true;
            remainder_H = 0;
        }
        if (ZERO_TOL > remainder_W && remainder_W > -ZERO_TOL) {
            is_w_int_shift = true;
            remainder_W = 0;
        }

        // regular shift
        {
            T pixels[2][2] = {0};
            int temp_H, temp_W;
            // TODO remove array input_H_indices
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    temp_H = input_H_idx + h;
                    temp_W = input_W_idx + w;
                    SET_WITHIN_BOUND(pixels[h][w], temp_H, temp_W);
                }
            }
            // del L/remainder_H
            local_pixel_H_grad =
                (1 - remainder_W) * (pixels[1][0] - pixels[0][0])
                    + remainder_W * (pixels[1][1] - pixels[0][1]);
            // del L/remainder_W
            local_pixel_W_grad =
                (1 - remainder_H) * (pixels[0][1] - pixels[0][0])
                    + remainder_H * (pixels[1][1] - pixels[1][0]);
        }

        if (is_h_int_shift || is_w_int_shift) {
            // pixels[1][1] is the origin pixel
            // TODO remove array input_H_indices
            T pixels[3][3] = {0};
            int temp_H, temp_W;
            for (int h = 0; h < 3; ++h) {
                for (int w = 0; w < 3; ++w) {
                    if (h == 0 && w == 0 || h == 1 && w == 1)
                        continue;  // don't need p[0][0], p[1][1]
                    temp_H = input_H_idx + h - 1;
                    temp_W = input_W_idx + w - 1;
                    SET_WITHIN_BOUND(pixels[h][w], temp_H, temp_W);
                }
            }
            if (is_h_int_shift) {
                local_pixel_H_grad =
                    static_cast<T>(0.5f) * (
                        (1 - remainder_W) * (pixels[2][1] - pixels[0][1])
                            + remainder_W * (pixels[2][2] - pixels[0][2])
                    );
            }

            if (is_w_int_shift) {
                local_pixel_W_grad =
                    static_cast<T>(0.5f) * (
                        (1 - remainder_H) * (pixels[1][2] - pixels[1][0])
                            + remainder_H * (pixels[2][2] - pixels[2][0])
                    );
            }
        }

        // Multiply by the upstream gradient
        const T og = output_grad[N_idx][C_idx][H_idx][W_idx];
        T pixel_H_grad = local_pixel_H_grad * og;
        T pixel_W_grad = local_pixel_W_grad * og;

        // output: {2 x C, H x W}
        // H_offset = u_h * H + u_w * W + u_t;
        // W_offset = v_h * H + v_w * W + v_t;
        atomicAdd(&shift_grad_buffer[0][C_idx][H_idx][W_idx], pixel_H_grad);
        atomicAdd(&shift_grad_buffer[1][C_idx][H_idx][W_idx], pixel_W_grad);
    }
}


template <typename T>
__global__ void rubiks2d_backward_input_kernel(
    PTA<T, 4> output_grad,
    const PTA<T, 4> input,
    const PTA<T, 2> shift,
    const uint32_t stride_H, const uint32_t stride_W,
    const uint32_t pad_H, const uint32_t pad_W,
    bool quantize,
    PTA<T, 4> input_grad,
    DECLARE_KERNEL_DIM_ARGS
) {
    GRID_STRIDE_LOOP(index, input_numel) {
        GET_INDICES(input, N_idx, C_idx, H_idx, W_idx);

        // Offsets within the (H, W) feature map in the output gradient tensor to pull from
        // (note that backward input gradient is just output gradient, reverse shifted)
        const int H_offset = H_idx + pad_H;
        const int W_offset = W_idx + pad_W;

        // Final value to be stuck into the input gradient
        T val = 0;

        const T shift_H = -shift[0][C_idx];
        const T shift_W = -shift[1][C_idx];

        if (quantize) {
            int temp_H = round_fast(H_offset + shift_H);
            int temp_W = round_fast(W_offset + shift_W);

            if (temp_H % stride_H == 0 && temp_W % stride_W == 0) {
                temp_H /= stride_H;
                temp_W /= stride_W;

                if (temp_H >= 0 && temp_W >= 0 &&
                    temp_H < output_H_dim && temp_W < output_W_dim) {
                    input_grad[N_idx][C_idx][H_idx][W_idx] =
                        output_grad[N_idx][C_idx][temp_H][temp_W];
                }
            }
            continue;  // skip the rest
        }

        int output_H_idx, output_W_idx;

        // "Small" and "large" shifts in each direction. If our shift is 1.4, for instance, the
        // "small shift" will give us the pixel offset by 1 and the "large shift" will give us
        // the pixel offset by 2.
        int small_shift_H = floor_fast(shift_H);
        int large_shift_H = small_shift_H + 1;
        int small_shift_W = floor_fast(shift_W);
        int large_shift_W = small_shift_W + 1;

        // Special case -- both shifts are zero; only care about strides and padding with NO interpolation.
        if (shift_W == 0 && shift_H == 0) {
            output_H_idx = H_offset;
            output_W_idx = W_offset;

            // Check and see if we're a strided sample or not.
            if (output_W_idx % stride_W == 0 && output_H_idx % stride_H == 0) {
                output_W_idx = output_W_idx / stride_W;
                output_H_idx = output_H_idx / stride_H;

                // Basically an in-bounds checker -- if things are in bounds, go ahead and pull from the output
                // gradient tensor; otherwise just give zero.
                if (output_H_idx >= 0 && output_W_idx >= 0 && \
                    output_H_idx < output_H_dim && output_W_idx < output_W_dim) {
                    val = output_grad[N_idx][C_idx][output_H_idx][output_W_idx];
                } else {
                    val = 0;
                }
            }

        } else {

            int output_H_indices[2];
            int output_W_indices[2];

            output_H_indices[0] = H_offset + small_shift_H;
            output_W_indices[0] = W_offset + small_shift_W;

            output_H_indices[1] = H_offset + large_shift_H;
            output_W_indices[1] = W_offset + large_shift_W;

            T pixels[2][2] = {0};
            int temp_H, temp_W;

            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    temp_H = output_H_indices[h];
                    temp_W = output_W_indices[w];

                    if (temp_H % stride_H == 0 && temp_W % stride_W == 0) {
                        temp_H /= stride_H;
                        temp_W /= stride_W;

                        if (temp_H >= 0 && temp_W >= 0 &&
                            temp_H < output_H_dim && temp_W < output_W_dim) {
                            pixels[h][w] =
                                output_grad[N_idx][C_idx][temp_H][temp_W];
                        }
                    }
                }
            }

            T remainder_H = shift_H - small_shift_H;
            T remainder_W = shift_W - small_shift_W;
            val = interpolate_2d(pixels, remainder_H, remainder_W);
        }
        input_grad[N_idx][C_idx][H_idx][W_idx] = val;
    }
}

template<typename T>
__global__ void rubiks2d_normalize_shift_grad_kernel(
    PTA<T, 2> shift_grad
) {
    // total elements is C_dim
    GRID_STRIDE_LOOP(c, shift_grad.size(1)) {
        // TODO wrong
        const T cur_H_grad = shift_grad[0][c];
        const T cur_W_grad = shift_grad[1][c];
        const T magnitude = sqrt(cur_H_grad * cur_H_grad + cur_W_grad * cur_W_grad);

        if (magnitude > 0) {
            shift_grad[0][c] = cur_H_grad / magnitude;
            shift_grad[1][c] = cur_W_grad / magnitude;
        }
    }
}
    
    
inline bool is_s1p0(
    const std::vector<int>& strides, const std::vector<int>& paddings
) {
    return strides[0] == 1 && strides[1] == 1
        && paddings[0] == 0 && paddings[1] == 0;
}


void rubiks2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor shift_field,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    bool quantize,
    torch::Tensor output
) {
    int blocks = 0;
    int threads_per_block = 0;
    CALCULATE_DIM_ARGS(input, output);

    get_cuda_device_properties(output_numel, blocks, threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "rubiks2d_forward_cuda", ([&] {
        rubiks2d_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            GET_PTA(input, 4),
            GET_PTA(shift_field, 2),
            strides[0], strides[1], paddings[0], paddings[1],
            quantize,
            GET_PTA(output, 4),
            PASS_KERNEL_DIM_ARGS
        );
    }));
}


void rubiks2d_backward_shift_cuda(
    torch::Tensor output_grad,
    torch::Tensor input,
    torch::Tensor shift_field,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    torch::Tensor shift_grad_buffer
) {
    int blocks = 0;
    int threads_per_block = 0;
    CALCULATE_DIM_ARGS(input, output_grad);
    get_cuda_device_properties(output_numel, blocks, threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "rubiks2d_backward_shift_cuda", ([&] {
        rubiks2d_backward_shift_kernel<scalar_t><<<blocks, threads_per_block>>>(
            GET_PTA(output_grad, 4),
            GET_PTA(input, 4),
            GET_PTA(shift_field, 2),
            strides[0], strides[1], paddings[0], paddings[1],
            GET_PTA(shift_grad_buffer, 4),
            PASS_KERNEL_DIM_ARGS
        );
    }));
}


void rubiks2d_backward_input_cuda(
    torch::Tensor output_grad,
    torch::Tensor input,
    torch::Tensor shift_field,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    bool quantize,
    torch::Tensor input_grad
) {
    int blocks = 0;
    int threads_per_block = 0;
    CALCULATE_DIM_ARGS(input, output_grad);
    get_cuda_device_properties(input_numel, blocks, threads_per_block);

    // TODO use s1p0
    if (true or !is_s1p0(strides, paddings)) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "rubiks2d_backward_input_cuda", ([&] {
            rubiks2d_backward_input_kernel<scalar_t><<<blocks, threads_per_block>>>(
                GET_PTA(output_grad, 4),
                GET_PTA(input, 4),
                GET_PTA(shift_field, 2),
                strides[0], strides[1], paddings[0], paddings[1],
                quantize,
                GET_PTA(input_grad, 4),
                PASS_KERNEL_DIM_ARGS
            );
        }));
    }
    else {
//        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "rubiks2d_backward_input_s1p0", ([&] {
//            rubiks2d_backward_input_kernel_s1p0<scalar_t><<<blocks, threads_per_block>>>(
//                GET_PTA(input, 4),
//                GET_PTA(shift, 2),
//                GET_PTA(output_grad, 4),
//                GET_PTA(input_grad, 4),
//                PASS_KERNEL_DIM_ARGS
//            );
//        }));
    }
}

void rubiks2d_normalize_shift_grad_cuda(
    torch::Tensor shift_grad
) {

    int blocks = 0;
    int threads_per_block = 0;
    uint32_t total_elements = shift_grad.size(1);  // C_dim
    get_cuda_device_properties(total_elements, blocks, threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(shift_grad.type(), "rubiks2d_normalize_shift_grad_cuda", ([&] {
        rubiks2d_normalize_shift_grad_kernel<scalar_t><<<blocks, threads_per_block>>>(
            GET_PTA(shift_grad, 2)
        );
    }));
}

} // namespace rubiks