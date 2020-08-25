import torch
import rubiksnet_cuda
from rubiksnet.utils import *


__all__ = ["rubiks2d", "rubiks2d_forward", "rubiks2d_backward"]


def _get_output_dim(orig, stride, padding):
    return (orig + 2 * padding - 1) / stride + 1


def compute_output_shape(x, stride, padding, shift_dim=2):
    batch, C_in, H_in, W_in = x.size()
    assert shift_dim == 2, "TODO"

    strides = make_tuple(stride, shift_dim)
    paddings = make_tuple(padding, shift_dim)

    H_out = _get_output_dim(H_in, strides[0], paddings[0])
    W_out = _get_output_dim(W_in, strides[1], paddings[1])

    return batch, C_in, int(H_out), int(W_out)


def reshape_shift_field(shift):
    S, C = shift.size()
    assert S == 6
    # return shift.transpose(0,1).view((C, 3, 2)).transpose(1,2).contiguous().view((C, 2, 3))
    new_shift = shift.new_zeros((C, 2, 3))
    for i in range(C):
        for p in range(2):
            for q in range(3):
                new_shift[i, p, q] = shift[q + p * 3, i]
    return new_shift


# ========================================================
# ===== CUDA forward primitive =====
# ========================================================
def make_rubiks_forward(forward_func, dim=2):
    assert dim == 2, "TODO"

    def _rubiks_forward(x, shift, stride=1, padding=0, quantize=False, output=None):
        """
        Pure forward pass primitive, no gradient computation
        """
        strides = make_tuple(stride, repeats=dim)
        paddings = make_tuple(padding, repeats=dim)
        # x: (N, C, H, W), shift: (DIM, C)
        assert x.is_cuda, "shift only works on CUDA tensors"
        # assert x.size(1) == shift.size(1), 'x tensor channel dim[1] must match shift channel dim[1]'
        assert x.dtype == shift.dtype, "x and shift must have the same dtype"
        output_shape = compute_output_shape(x, strides, paddings, shift_dim=dim)
        output = allocate_output(output, x, output_shape)
        ret = forward_func(
            input=x,
            shift=shift,
            strides=strides,
            paddings=paddings,
            quantize=quantize,
            output=output,
        )
        assert ret == 0, "CUDA kernel return code {} != 0, error".format(ret)
        return output

    _rubiks_forward.__name__ = "rubiks{}d_forward".format(dim)
    return _rubiks_forward


# ========================================================
# ===== CUDA backward primitive =====
# ========================================================
def make_rubiks_backward(backward_func, dim=2):
    assert dim == 2, "TODO"

    def _rubiks_backward(
        upstream_grad,
        x,
        shift,
        stride,
        padding,
        normalize_grad=True,
        enable_shift_grad=True,
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
        strides = make_tuple(stride, repeats=dim)
        paddings = make_tuple(padding, repeats=dim)
        assert (
            x.is_cuda and upstream_grad.is_cuda and shift.is_cuda
        ), "shift only works on CUDA tensors"
        x_grad = allocate_output(x_grad_output, x, x.size())
        shift_grad = allocate_output(shift_grad_output, shift, shift.size())
        ret = backward_func(
            upstream_grad=upstream_grad,
            input=x,
            shift=shift,
            strides=strides,
            paddings=paddings,
            normalize_grad=normalize_grad,
            enable_shift_grad=enable_shift_grad,
            quantize=quantize,
            input_grad=x_grad,
            shift_grad=shift_grad,
        )
        assert ret == 0, "CUDA return code {} != 0, error".format(ret)
        return x_grad, shift_grad

    _rubiks_backward.__name__ = "rubiks{}d_backward".format(dim)
    return _rubiks_backward


# ========================================================
# ===== User-facing functional interface with autograd =====
# ========================================================


def make_rubiks_functional(forward_method, backward_method, dim=2):
    assert dim == 2, "TODO"

    # make primitive autograd.Function class
    class _RubiksFunc(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, x, shift, stride, padding, normalize_grad, enable_shift_grad, quantize
        ):
            assert isinstance(normalize_grad, bool)
            assert isinstance(enable_shift_grad, bool)
            ctx.stride = stride
            ctx.padding = padding
            ctx.normalize_grad = normalize_grad
            ctx.enable_shift_grad = enable_shift_grad
            ctx.quantize = quantize
            ctx.save_for_backward(x, shift)
            return forward_method(x, shift, stride, padding, quantize)

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
                    enable_shift_grad=ctx.enable_shift_grad,
                    quantize=ctx.quantize,
                )
                if ctx.needs_input_grad[0]:
                    x_grad = _x_grad
                if ctx.needs_input_grad[1]:
                    shift_grad = _shift_grad

            # must match the number of input args
            return x_grad, shift_grad, None, None, None, None, None

    _RubiksFunc.__name__ = "VFS{}DFunc".format(dim)

    # user facing functional
    def _rubiks_shift(
        x,
        shift,
        stride=1,
        padding=0,
        normalize_grad=True,
        enable_shift_grad=True,
        quantize=False,
    ):
        """
        Also supports grouped shift
        """
        assert len(x.size()) == 4, "x must be [N, C, H, W]"
        return _RubiksFunc.apply(
            x, shift, stride, padding, normalize_grad, enable_shift_grad, quantize
        )

    _rubiks_shift.__name__ = "rubiks{}d".format(dim)

    return _rubiks_shift


rubiks2d_forward = make_rubiks_forward(rubiksnet_cuda.rubiks2d_forward,)


rubiks2d_backward = make_rubiks_backward(rubiksnet_cuda.rubiks2d_backward)


rubiks2d = make_rubiks_functional(rubiks2d_forward, rubiks2d_backward)
