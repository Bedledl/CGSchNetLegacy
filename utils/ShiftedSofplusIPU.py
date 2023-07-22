import torch


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        u = torch.log1p(torch.exp(-x.sign() * x))
        v = torch.clamp_min(x, 0.0)
        return u + v - self.shift
