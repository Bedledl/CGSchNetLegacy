from typing import Dict

import torch

import schnetpack
import schnetpack.properties as properties


class PairwiseDistancesIPU(torch.nn.Module):
    """
    Compute pair-wise distances from indices provided by a neighbor list transform.
    """

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        R = inputs[properties.R]
        offsets = inputs[properties.offsets]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        #pos_j = torch.index_select(R, 0, idx_j)
        #pos_i = torch.index_select(R, 0, idx_i)
        #pos_i = R[idx_j]
        #pos_j = R[idx_i]
        d_ij = torch.nn.PairwiseDistance()
        Rij = pos_j - pos_i + offsets
        inputs[properties.Rij] = Rij
        return inputs
