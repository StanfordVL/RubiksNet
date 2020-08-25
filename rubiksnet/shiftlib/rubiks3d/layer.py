import torch
import torch.nn as nn
from .primitive import *
from ..rubiks2d.layer import RubiksShift2D
from rubiksnet.utils import *


__all__ = [
    "RubiksShift3D",
    "RubiksShiftBase",
    "init_shift_uniform",
    "init_shift1d_nfold",
]


# ========================================================
# ===== User-facing nn.Module interface =====
# ========================================================


def init_shift_uniform(shift):
    nn.init.uniform_(shift, -1, 1)


def init_shift1d_nfold(shift, nfold=8, noise=1e-3):
    """
    Mimic TSM hard-coded shift.
    First 1/nfold channels shift one time step to the past
    Second 1/nfold channels shift one time step to the future
    Remaining channels remain unchanged
    """
    dim, channels = shift.size()
    assert dim == 1, "only works with rubiks1d"
    with torch.no_grad():
        group = channels // nfold
        shift[:, :group] = 1
        shift[:, group : 2 * group] = -1
        # perturb to stay away from zero
        shift[:, 2 * group :].uniform_(-noise, noise)
    return shift


class RubiksShiftBase(nn.Module):
    def __init__(
        self,
        num_channels,
        stride=1,
        padding=0,
        normalize_grad=True,
        normalize_t_factor=1.0,
        shift_groups=1,
        quantize=False,
        *,
        dim,
        shift_function,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.padding = padding
        self.normalize_grad = normalize_grad
        self.normalize_t_factor = normalize_t_factor
        self.quantize = quantize
        assert (
            num_channels % shift_groups == 0
        ), "Does not satisfy num_channels % shift_groups == 0"
        self.shift = nn.Parameter(torch.zeros(dim, num_channels // shift_groups))
        init_shift_uniform(self.shift)
        self.shift_function = shift_function

    def forward(self, x):
        return self.shift_function(
            x,
            self.shift,
            stride=self.stride,
            padding=self.padding,
            normalize_grad=self.normalize_grad,
            normalize_t_factor=self.normalize_t_factor,
            quantize=self.quantize,
        )

    def extra_repr(self):
        return "shift_channels={}".format(self.num_channels)


class RubiksShift3D(RubiksShiftBase):
    def __init__(
        self,
        num_channels,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        normalize_grad=True,
        normalize_t_factor=1.0,
        quantize=False,
        shift_groups=1,
    ):
        super().__init__(
            num_channels,
            stride,
            padding,
            normalize_grad,
            normalize_t_factor,
            shift_groups,
            quantize=quantize,
            dim=3,
            shift_function=rubiks_shift_3d,
        )


# TODO: delete
def create_3d_from_2d(
    module_2d, init_mode="tsm", normalize_t_factor=1.0, quantize=False
):
    assert isinstance(module_2d, RubiksShift2D)
    module_3d = RubiksShift3D(
        module_2d.num_channels,
        stride=(1, *make_tuple(module_2d.stride, 2)),
        padding=(0, *make_tuple(module_2d.padding, 2)),
        normalize_grad=True,
        normalize_t_factor=normalize_t_factor,
        quantize=quantize,
    )
    # print('Shift init mode:', init_mode)
    with torch.no_grad():
        D, C = module_3d.shift.size()
        assert D == 3, "INTERNAL ERROR"
        module_3d.shift[1:, :] = module_2d.shift
        if init_mode.startswith("tsm-g"):
            stddev = float(init_mode[5:])
            fold = C // 8
            if stddev == 0:
                stddev = 1e-2
            module_3d.shift[0, :][:fold] = 1.0 + torch.randn((fold,)) * stddev
            module_3d.shift[0, :][fold : 2 * fold] = (
                -1.0 + torch.randn((fold,)) * stddev
            )
            module_3d.shift[0, :][2 * fold :] = torch.randn((C - 2 * fold,)) * stddev
        elif init_mode == "tsm":
            fold = C // 8
            module_3d.shift[0, :][:fold].fill_(1)
            module_3d.shift[0, :][fold : 2 * fold].fill_(-1)
            module_3d.shift[0, :][2 * fold :].fill_(0)
        elif init_mode.startswith("uni"):
            magnitude = float(init_mode[3:])
            assert magnitude > 0, f"uniform random magnitude must > 0: {magnitude}"
            # original init is uniform[-1, 1]
            module_3d.shift[0, :] *= magnitude
        elif init_mode.lower() == "none":
            # module_3d value must be loaded downstream
            # fill it with NaN as a safeguard
            module_3d.shift.fill_(float("nan"))
        else:
            raise NotImplementedError(f"unknown init mode {init_mode}")

    return module_3d
