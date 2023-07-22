import torch


class DummyCutoff(torch.nn.Identity):
    def __init__(self, cutoff):
        super(DummyCutoff, self).__init__()
        self.cutoff = cutoff
