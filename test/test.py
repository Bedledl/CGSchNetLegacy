import poptorch

import torch
import math


class ShiftedSoftplusOld(torch.nn.Module):
    """
    use torch.nn.Softplus for softplus function. instead of functional.softplus
    like it is used in schnetpack.nn.activations.shifted_softplus
    """
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor):
        return self.softplus(x) - math.log2(2.0)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        u = torch.log1p(torch.exp(-x.sign() * x))
        v = torch.clamp_min(x, 0.0)
        return u + v - self.shift


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.ssp = ShiftedSoftplus()
        self.lin1 = torch.nn.Linear(2, 1)
        self.softplus_cpu = poptorch.CPU(self.ssp, "MyCPUOp")

    def forward(self, x):#, on_ipu):
        x.requires_grad_()

        y = self.ssp(x)
        #y = self.softplus_cpu(x)

        y = self.lin1(y)
        return torch.autograd.grad(y, x)


def main():
    nn = SimpleNN()

    inputs = [
        torch.tensor([1., 4.]),
        torch.tensor([6., 4.]),
    ]
    nn_pop = poptorch.inferenceModel(nn)

    for i in inputs:
        print(nn(i, False))
        nn.zero_grad()
        print(nn_pop(i, True))
        nn_pop.zero_grad()

    torch.save(nn, "simple_nn_test_save")


if __name__ == '__main__':
    main()
