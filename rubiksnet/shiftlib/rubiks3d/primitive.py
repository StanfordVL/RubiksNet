import torch  # this line is necessary for CUDAExtension to load
import rubiksnet_cuda
from rubiksnet.utils import *


__all__ = [
    "rubiks_shift_3d_forward",
    "rubiks_shift_3d_backward",
    "rubiks_shift_3d",
]


def _make_tuple(elem, repeats):
    """
    expand 3 into (3, 3) for strides/paddings
    """
    if isinstance(elem, int):
        return [elem] * repeats
    else:
        assert len(elem) == repeats
        return [int(x) for x in elem]


def _get_output_dim(orig, stride, padding):
    return (orig + 2 * padding - 1) / stride + 1


def compute_output_shape(x, stride, padding, shift_dim):
    batch, T_in, C_in, H_in, W_in = x.size()
    T_out, H_out, W_out = T_in, H_in, W_in

    strides = _make_tuple(stride, shift_dim)
    paddings = _make_tuple(padding, shift_dim)

    if shift_dim == 1:
        T_out = _get_output_dim(T_in, strides[0], paddings[0])
    elif shift_dim == 2:
        H_out = _get_output_dim(H_in, strides[0], paddings[0])
        W_out = _get_output_dim(W_in, strides[1], paddings[1])
    elif shift_dim == 3:
        T_out = _get_output_dim(T_in, strides[0], paddings[0])
        H_out = _get_output_dim(H_in, strides[1], paddings[1])
        W_out = _get_output_dim(W_in, strides[2], paddings[2])
    else:
        raise NotImplementedError("only 1D, 2D, 3D shifts supported")

    return batch, int(T_out), C_in, int(H_out), int(W_out)


# ========================================================
# ===== CUDA forward primitive =====
# ========================================================
def make_rubiks_forward(forward_float, forward_double, dim):
    def _rubiks_forward(x, shift, stride, padding, quantize=False, output=None):
        """
        Pure forward pass primitive, no gradient computation
        """
        strides = _make_tuple(stride, repeats=dim)
        paddings = _make_tuple(padding, repeats=dim)
        # x: (N, T, C, H, W), shift: (DIM, C)
        assert x.is_cuda, "rubiks shift only works on CUDA tensors"
        assert x.size(2) == shift.size(
            1
        ), "x tensor channel dim[2] must match shift channel dim[1]"
        assert x.dtype == shift.dtype, "x and shift must have the same dtype"
        if x.dtype == torch.float32:
            shift_func = forward_float
        elif x.dtype == torch.float64:
            shift_func = forward_double
        else:
            raise ValueError(
                "rubiks_shift_{}d only supports float32 and float64 (double) dtypes.".format(
                    dim
                )
            )
        output_shape = compute_output_shape(x, strides, paddings, shift_dim=dim)
        output = allocate_output(output, x, output_shape)
        ret = shift_func(x, shift, strides, paddings, quantize, output)
        assert ret == 0, "CUDA kernel return code {} != 0, error".format(ret)
        return output

    _rubiks_forward.__name__ = "rubiks_shift_{}d_forward".format(dim)
    return _rubiks_forward


# ========================================================
# ===== CUDA backward primitive =====
# ========================================================
def make_rubiks_backward(backward_float, backward_double, dim):
    def _rubiks_backward(
        upstream_grad,
        x,
        shift,
        stride,
        padding,
        normalize_grad,
        normalize_t_factor=1.0,
        quantize=False,
        x_grad_output=None,
        shift_grad_output=None,
    ):
        """
        Pure backward pass primitive.
        Args:
            upstream_grad: Receives gradient w.r.t output from upstream.
            x: original input tensor
            shift: original shift tensor
        """
        strides = _make_tuple(stride, repeats=dim)
        paddings = _make_tuple(padding, repeats=dim)
        assert (
            x.is_cuda and upstream_grad.is_cuda
        ), "rubiks shift only works on CUDA tensors"
        if x.dtype == torch.float32:
            grad_func = backward_float
        elif x.dtype == torch.float64:
            grad_func = backward_double
        else:
            raise ValueError(
                "rubiks_shift_{}d only supports float32 and float64 (double) dtypes.".format(
                    dim
                )
            )

        x_grad = allocate_output(x_grad_output, x, x.size())
        shift_grad = allocate_output(shift_grad_output, shift, shift.size())
        ret = grad_func(
            x,
            shift,
            upstream_grad,
            strides,
            paddings,
            x_grad,
            shift_grad,
            normalize_grad,
            normalize_t_factor,
            quantize,
        )
        assert ret == 0, "CUDA return code {} != 0, error".format(ret)
        return x_grad, shift_grad

    _rubiks_backward.__name__ = "rubiks_shift_{}d_backward".format(dim)
    return _rubiks_backward


def make_rubiks_functional(forward_method, backward_method, dim):
    # make primitive autograd.Function class
    class _RubiksShiftFunc(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, x, shift, stride, padding, normalize_grad, normalize_t_factor, quantize
        ):
            assert isinstance(normalize_grad, bool)
            ctx.stride = stride
            ctx.padding = padding
            ctx.normalize_grad = normalize_grad
            ctx.normalize_t_factor = normalize_t_factor
            ctx.quantize = quantize
            ctx.save_for_backward(x, shift)
            return forward_method(x, shift, stride, padding, quantize=quantize)

        @staticmethod
        def backward(ctx, grad_output):
            """
            Refer to https://pytorch.org/docs/stable/notes/extending.html
            """
            x, shift = ctx.saved_tensors
            x_grad = shift_grad = None

            # compute grad only if either X or shift needs gradient
            if any(ctx.needs_input_grad):
                _x_grad, _shift_grad = backward_method(
                    grad_output,
                    x,
                    shift,
                    stride=ctx.stride,
                    padding=ctx.padding,
                    normalize_grad=ctx.normalize_grad,
                    normalize_t_factor=ctx.normalize_t_factor,
                    quantize=ctx.quantize,
                )
                if ctx.needs_input_grad[0]:
                    x_grad = _x_grad
                if ctx.needs_input_grad[1]:
                    shift_grad = _shift_grad

            # must match the number of input args
            return x_grad, shift_grad, None, None, None, None, None

    _RubiksShiftFunc.__name__ = "RubiksShift{}DFunc".format(dim)

    # user facing functional
    def _rubiks_shift(
        x,
        shift,
        stride=1,
        padding=0,
        normalize_grad=True,
        normalize_t_factor=1.0,
        quantize=False,
    ):
        """
        Also supports grouped shift
        """
        assert len(x.size()) == 5, "x must be [N, T, C, H, W]"
        H, T, C, H, W = x.size()
        shift_channel = shift.size(1)
        assert C == shift_channel, "group shift is deprecated. Now C dim must match."
        if normalize_t_factor == "auto":
            normalize_t_factor = T / H
        else:
            assert isinstance(normalize_t_factor, (int, float))
        return _RubiksShiftFunc.apply(
            x, shift, stride, padding, normalize_grad, normalize_t_factor, quantize
        )

    _rubiks_shift.__name__ = "rubiks_shift_{}d".format(dim)

    return _rubiks_shift


rubiks_shift_3d_forward = make_rubiks_forward(
    rubiksnet_cuda.rubiks_shift_3d_forward_float,
    rubiksnet_cuda.rubiks_shift_3d_forward_double,
    dim=3,
)

rubiks_shift_3d_backward = make_rubiks_backward(
    rubiksnet_cuda.rubiks_shift_3d_backward_float,
    rubiksnet_cuda.rubiks_shift_3d_backward_double,
    dim=3,
)

rubiks_shift_3d = make_rubiks_functional(
    rubiks_shift_3d_forward, rubiks_shift_3d_backward, 3
)

