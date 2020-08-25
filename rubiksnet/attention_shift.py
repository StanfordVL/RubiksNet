import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionShift(nn.Module):
    def __init__(self, n_segment):
        super().__init__()
        self.n_segment = n_segment
        self.kernel_size = 3
        self.T = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.weight = None

    def forward(self, x):
        out = self.attention_shift(x)
        return out

    def attention_shift(self, x):
        nt, c, h, w = x.size()
        c_new = c * h * w
        n_batch = nt // self.n_segment
        x = x.view((n_batch, self.n_segment, c_new)).transpose(1, 2)
        # initialize weight
        if self.weight is None:
            weight = torch.rand(c, self.kernel_size)
            weight = weight.to(x.device)
            self.weight = nn.Parameter(weight)

        weight = self.weight / (torch.std(self.weight, dim=1, keepdim=True) + 1e-6)
        weight = F.softmax(weight / self.T, dim=1)

        inflated_weight = torch.repeat_interleave(weight, repeats=h * w, dim=0).view(
            c_new, 1, self.kernel_size
        )

        out = F.conv1d(
            x, inflated_weight, padding=self.kernel_size // 2, groups=c_new
        )
        return out.transpose(1, 2).contiguous().view(nt, c, h, w)
