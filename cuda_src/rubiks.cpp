#include "utils.h"
#include "rubiks3d.h"

#include <cfloat>
#include <iterator>
#include <map>
#include <cmath>

using namespace torchx;


namespace rubiks {

inline int64_t compute_output_len(const int64_t input_size, const int stride, const int padding) {
	if (stride <= 0) {
		std::cerr << "Invalid argument: stride must be > 0, but got " << stride << std::endl;
	}
	int64_t output_size = (input_size + 2 * padding - 1) / stride + 1;

	if (output_size < 0) {
		std::cerr 
			<< "Invalid argument: computed output size would be negative: "
			<< output_size
			<< " [input_size: " << input_size
			<< ", stride: " << stride
			<< ", padding: " << padding << "]" 
			<< std::endl;
	}
	return output_size;
}


// forward declaration
void rubiks2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor shift_field,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    bool quantize,
    torch::Tensor output
);


int rubiks2d_forward(
    torch::Tensor input,
    torch::Tensor shift,
	const std::vector<int>& strides,
	const std::vector<int>& paddings,
    bool quantize,
	torch::Tensor output
) {
    TX_CHECK_TENSOR(input);
    TX_CHECK_TENSOR(shift);
    TX_CHECK_TENSOR(output);

    if (is_sys_debug()) {
        pyprint("rubiks2d_forward args: strides=", strides,
            "; padding=", paddings,
            "; quantize=", quantize);
    }
    int64_t C_dim = input.size(1);
    std::vector<int64_t> shift_shape {2, C_dim};
    TX_CHECK_SHAPE(shift_shape, shift.sizes(), "rubiks shift");

    rubiks2d_forward_cuda(input, shift, strides, paddings, quantize, output);
    return 0;
}


void rubiks2d_backward_shift_cuda(
    torch::Tensor upstream_grad,
    torch::Tensor input,
    torch::Tensor shift,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    torch::Tensor shift_grad_buffer
);

void rubiks2d_backward_input_cuda(
    torch::Tensor upstream_grad,
    torch::Tensor input,
    torch::Tensor shift,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    bool quantize,
    torch::Tensor input_grad
);


void rubiks2d_normalize_shift_grad_cuda(
    torch::Tensor shift_grad
);

int rubiks2d_backward(
    torch::Tensor output_grad,
    torch::Tensor input,
    torch::Tensor shift_field,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    bool normalize_grad,
    bool enable_shift_grad,
    bool quantize,
    torch::Tensor input_grad,
    torch::Tensor shift_grad  // shape {2, C}
) {
    // IMPORTANT: input_grad and shift_grad must be all zero tensors
    // because backward() adds to them, not set value
    TX_CHECK_TENSOR(output_grad);
    TX_CHECK_TENSOR(input);
    TX_CHECK_TENSOR(shift_field);
    TX_CHECK_TENSOR(input_grad);
    TX_CHECK_TENSOR(shift_grad);

    const int C_dim = input.size(1);
    const int output_H_dim = output_grad.size(2);
    const int output_W_dim = output_grad.size(3);

    if (is_sys_debug()) {
        pyprint("rubiks2d_backward args: strides=", strides,
                "; padding=", paddings, "normalize_grad=", normalize_grad,
                "; enable_shift_grad=", enable_shift_grad,
                "; quantize=", quantize);
    }

    // Does not support learnable shift field YET
    if (enable_shift_grad) {
        torch::Tensor shift_grad_buffer = torch::zeros(
            {2, C_dim, output_H_dim, output_W_dim}, input.options()
        );
        torch::Tensor ones_vector = torch::ones(
            {output_H_dim, output_W_dim}, input.options()
        );
        rubiks2d_backward_shift_cuda(
            output_grad, input, shift_field, strides, paddings, shift_grad_buffer
        );
        // Uses matrix-vector multiplication with a vector of all ones to compute sums
        // across the spatial dimension (i.e. (2 * C, HW) \times (HW, 1) -> (2 * C, 1)).
        // Now shift_grad is (2 * C) and contains all the shift gradients, summed over all
        // the relevant pixels.
        (shift_grad.flatten()).addmv_(
            shift_grad_buffer.view({2 * C_dim, output_H_dim * output_W_dim}),
            ones_vector.flatten(), 0, 1
        );

        // Normalize gradient by dividing by magnitude per channel
        if (normalize_grad) {
            rubiks2d_normalize_shift_grad_cuda(shift_grad);
        }
    }

    rubiks2d_backward_input_cuda(
        output_grad, input, shift_field, strides, paddings, quantize, input_grad
    );
    return 0;
}

/*
 * **************************** Rubiks 3D ****************************
 */

void compute_output_shape(const int input_size, const int stride, const int padding, int* output_size) {
	if (stride <= 0) {
		std::cerr << "Invalid argument: stride must be > 0, but got " << stride << std::endl;
	}

	*output_size = (input_size + 2 * padding - 1) / stride + 1;


	if (*output_size < 0) {
		std::cerr
			<< "Invalid argument: computed output size would be negative: "
			<< *output_size
			<< " [input_size: " << input_size
			<< ", stride: " << stride
			<< ", padding: " << padding << "]"
			<< std::endl;
	}
}


template <typename T>
int rubiks_shift_3d_forward(at::Tensor* input_tensor_ptr, at::Tensor* shift_tensor_ptr, \
	const std::vector<int>& strides, const std::vector<int>& paddings,
	bool quantize, at::Tensor* output_tensor_ptr) {

    // Strides
    int stride_T = strides[0];
    int stride_H = strides[1];
    int stride_W = strides[2];

    // Paddings
    int pad_T = paddings[0];
    int pad_H = paddings[1];
    int pad_W = paddings[2];

    // Batch size
    const int N_dim = input_tensor_ptr->size(0);

    // Temporal dimension
    const int input_T_dim = input_tensor_ptr->size(1);

    // Number of input channels
    const int C_dim = input_tensor_ptr->size(2);

    // Spatial dimensions for input
    const int input_H_dim = input_tensor_ptr->size(3);
    const int input_W_dim = input_tensor_ptr->size(4);

    // Spatial dimension for output
    int output_T_dim, output_H_dim, output_W_dim;
    compute_output_shape(input_T_dim, stride_T, pad_T, &output_T_dim);
    compute_output_shape(input_H_dim, stride_H, pad_H, &output_H_dim);
    compute_output_shape(input_W_dim, stride_W, pad_W, &output_W_dim);

    // Layer Set-up ends
    #define CHECK_CUDA(x) AT_ASSERTM(x->type().is_cuda(), #x " must be a CUDA tensor")
    #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x->is_contiguous(), #x " must be contiguous")
    #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

    CHECK_CONTIGUOUS(input_tensor_ptr);
    CHECK_CONTIGUOUS(shift_tensor_ptr);
    CHECK_CONTIGUOUS(output_tensor_ptr);

    // Device
    c10::Device device(c10::DeviceType::CUDA);

    // Setup pointers. data<T>() definitely returns a pointer to the same tensor memory.
    T* input_tensor_data_ptr = (input_tensor_ptr->to(device)).data<T>();
    T* shift_tensor_data_ptr = (shift_tensor_ptr->to(device)).data<T>();
    T* output_tensor_data_ptr = (output_tensor_ptr->to(device)).data<T>();

    // Counts number of elements in output_tensor_ptr
    const int output_num_elements = output_tensor_ptr->numel();

    CHECK_CONTIGUOUS(input_tensor_ptr);
    CHECK_CONTIGUOUS(shift_tensor_ptr);
    CHECK_CONTIGUOUS(output_tensor_ptr);

    RubiksShift3DForward<T>()(device, \
            output_num_elements, N_dim, \
            input_T_dim, output_T_dim, C_dim, \
            input_H_dim, output_H_dim, input_W_dim, output_W_dim, \
            shift_tensor_data_ptr, shift_tensor_data_ptr + C_dim, \
            shift_tensor_data_ptr + 2 * C_dim, \
            pad_T, pad_H, pad_W, stride_T, stride_H, stride_W, \
            input_tensor_data_ptr, output_tensor_data_ptr, quantize);

    CHECK_CONTIGUOUS(input_tensor_ptr);
    CHECK_CONTIGUOUS(shift_tensor_ptr);
    CHECK_CONTIGUOUS(output_tensor_ptr);

	return 0;
}


template <typename T>
int rubiks_shift_3d_backward(const at::Tensor* input_tensor_ptr, const at::Tensor* shift_tensor_ptr,
        const at::Tensor* output_grad_ptr, const std::vector<int>& strides, const std::vector<int>& paddings,
        at::Tensor* input_grad_ptr, at::Tensor* shift_grad_ptr,
        const bool normalize_grad, const T normalize_t_factor, bool quantize) {

    // Strides
    int stride_T = strides[0];
    int stride_H = strides[1];
    int stride_W = strides[2];

    // Paddings
    int pad_T = paddings[0];
    int pad_H = paddings[1];
    int pad_W = paddings[2];

    // Batch size
    const int N_dim = input_tensor_ptr->size(0);

    // Number of time channels
    const int input_T_dim = input_tensor_ptr->size(1);

    // Number of input channels
    const int C_dim = input_tensor_ptr->size(2);

    // Spatial dimension for input
    const int input_H_dim = input_tensor_ptr->size(3);
    const int input_W_dim = input_tensor_ptr->size(4);

    // Sptial dimension for output
    const int output_T_dim = output_grad_ptr->size(1);
    const int output_H_dim = output_grad_ptr->size(3);
    const int output_W_dim = output_grad_ptr->size(4);

    // Device
    c10::Device device(c10::DeviceType::CUDA);

    // TensorOptions is literally an object you pass into a tensor constructor as a config
    at::TensorOptions options = torch::TensorOptions().dtype(input_tensor_ptr->dtype()).device(device).requires_grad(true);
    at::Tensor shift_grad_buffer = torch::zeros({3 * C_dim, output_H_dim, output_W_dim}, options);
    at::Tensor* shift_grad_buffer_ptr = &shift_grad_buffer;

    // To sum up shift gradients across all affected pixels
    at::Tensor ones_vector = torch::ones({output_H_dim, output_W_dim}, options);

    // Grabbing raw data pointers
    T* input_tensor_data_ptr = (input_tensor_ptr->to(device)).data<T>();
    T* shift_tensor_data_ptr = (shift_tensor_ptr->to(device)).data<T>();
    T* output_grad_data_ptr = (output_grad_ptr->to(device)).data<T>();
    T* input_grad_data_ptr = (input_grad_ptr->to(device)).data<T>();
    T* shift_grad_data_ptr = (shift_grad_ptr->to(device)).data<T>();
    T* shift_grad_buffer_data_ptr = shift_grad_buffer_ptr->data<T>();

    // count: output gradient's total size (number of elements)
    int total_num_output = output_grad_ptr->numel();

    // Total amount of offset between two dimensions for shift gradient buffer
    const int shift_grad_buffer_offset = C_dim * output_H_dim * output_W_dim;

    // Grabbing starts to various dimensions of shift and shift gradient buffer
    T* shift_tensor_data_T_ptr = shift_tensor_data_ptr;
    T* shift_tensor_data_H_ptr = shift_tensor_data_ptr + C_dim;
    T* shift_tensor_data_W_ptr = shift_tensor_data_ptr + 2 * C_dim;
    T* shift_grad_buffer_T_start_ptr = shift_grad_buffer_data_ptr;
    T* shift_grad_buffer_H_start_ptr = shift_grad_buffer_data_ptr + shift_grad_buffer_offset;
    T* shift_grad_buffer_W_start_ptr = shift_grad_buffer_data_ptr + 2 * shift_grad_buffer_offset;

    // Compute gradient contribution for shift params from each pixel
    RubiksShift3DBackward<T>()(device, \
            total_num_output, N_dim, \
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

    // Uses matrix-vector multiplication with a vector of all ones to compute sums
    // across the spatial dimension (i.e. (3 * C, HW) \times (HW, 1) -> (3 * C, 1))
    // Now shift_grad_ptr is (3 * C) and contains all the shift gradients, summed over all
    // the relevant pixels.
    (shift_grad_ptr->flatten()).addmv_(shift_grad_buffer.view({3 * C_dim, output_H_dim * output_W_dim}), \
        ones_vector.flatten(), 0, 1);

    // Grabs relevant index locations for start of T, H, and W shift grad
    T* shift_grad_T_data_ptr = shift_grad_data_ptr;
    T* shift_grad_H_data_ptr = shift_grad_data_ptr + C_dim;
    T* shift_grad_W_data_ptr = shift_grad_data_ptr + 2 * C_dim;

    if (normalize_grad) {
        NormalizeShiftGrad3D<T>()(device, C_dim, \
                shift_grad_T_data_ptr, \
                shift_grad_H_data_ptr, \
                shift_grad_W_data_ptr,
                normalize_t_factor);
    }

    // Backprop for input gradient
    const int num_input_elements = input_tensor_ptr->numel();

    RubiksShift3DBackwardInput<T>()(device, \
            num_input_elements, N_dim, \
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

    return 0;
}

} // namespace rubiks


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rubiks2d_forward", &rubiks::rubiks2d_forward, "Forward Pass rubiks 2D",
          "input"_a, "shift"_a, "strides"_a, "paddings"_a, "quantize"_a, "output"_a);
    m.def("rubiks2d_backward", &rubiks::rubiks2d_backward, "Backward Pass rubiks 2D",
          "upstream_grad"_a, "input"_a, "shift"_a, "strides"_a, "paddings"_a,
          "normalize_grad"_a, "enable_shift_grad"_a, "quantize"_a,
          "input_grad"_a, "shift_grad"_a);

	m.def("rubiks_shift_3d_forward_double", &rubiks::rubiks_shift_3d_forward<double>, "Forward Pass Rubiks 3D");
	m.def("rubiks_shift_3d_forward_float", &rubiks::rubiks_shift_3d_forward<float>, "Forward Pass Rubiks 3D");
	m.def("rubiks_shift_3d_backward_double", &rubiks::rubiks_shift_3d_backward<double>, "Backward Pass Rubiks 3D");
	m.def("rubiks_shift_3d_backward_float", &rubiks::rubiks_shift_3d_backward<float>, "Backward Pass Rubiks 3D");
}
