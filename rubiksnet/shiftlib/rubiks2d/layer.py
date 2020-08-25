import torch
import torch.nn as nn
from .primitive import rubiks2d


def init_shift_group(shift, kernel_size):
    K = kernel_size
    C = shift.size(1)
    s = kernel_size // 2
    r = torch.arange(-s, s+1, dtype=shift.dtype)
    groups = C // K ** 2
    alpha = r.repeat(K * groups)
    beta = r.repeat_interleave(K).repeat(groups)
    shift[0, :] = alpha
    shift[1, :] = beta


class RubiksShift2D(nn.Module):
    def __init__(self, num_channels,
                 stride=1, padding=0,
                 normalize_grad=True,
                 quantize=False,
                 init_shift='uniform'):
        super().__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.padding = padding
        self.normalize_grad = normalize_grad
        self.quantize = quantize
        self.shift = nn.Parameter(torch.zeros(2, num_channels))
        with torch.no_grad():
            if init_shift == 'uniform':
                nn.init.uniform_(self.shift, -1, 1)
            elif init_shift.startswith('group'):
                group_kernel = int(init_shift[5:])
                assert group_kernel > 1
                # print('initializing 2D shift with 0flop group', group_kernel)
                init_shift_group(self.shift, group_kernel)
            else:
                raise NotImplementedError(f'unrecognized init shift {init_shift}')

    def forward(self, x):
        return rubiks2d(
            x, self.shift,
            stride=self.stride, padding=self.padding,
            normalize_grad=self.normalize_grad,
            enable_shift_grad=True,
            quantize=self.quantize
        )

    def extra_repr(self):
        return 'shift_channels={}'.format(self.num_channels)
