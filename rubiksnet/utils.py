import torch


def make_tuple(elem, repeats):
    """
    E.g. expand 3 into (3, 3) for things like strides/paddings
    """
    if isinstance(elem, int):
        return [elem] * repeats
    else:
        assert len(elem) == repeats
        return [int(x) for x in elem]


def allocate_output(output, tensor_like, desired_shape):
    """
    C++ output tensor
    Args:
        output: if None, allocates a new CUDA tensor of the desired shape
            else fills in the existing output tensor
        tensor_like: a tensor that has the desired dtype and device from which
            `output` will be derived
        desired_shape: tuple
    """
    if output is None:
        output_ = tensor_like.new_zeros(desired_shape)
    else:
        assert torch.is_tensor(output)
        assert (
            output.size() == desired_shape
        ), "output tensor has wrong shape {}, which should be {}".format(
            output.size(), desired_shape
        )
        assert (
            output.dtype == tensor_like.dtype
        ), "output tensor has wrong dtype {}, which should be {}".format(
            output.dtype, tensor_like.dtype
        )
        assert (
            output.device == tensor_like.device
        ), "output tensor has wrong device {}, which should be {}".format(
            output.device, tensor_like.device
        )
        output_ = output
    return output_
