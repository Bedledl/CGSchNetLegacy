from typing import Dict

import torch
from schnetpack import properties
from schnetpack.transform import Transform

import poptorch
#if poptorch.isRunningOnIpu() or poptorch:
from poptorch_geometric.ops import knn_graph
#else:
#    from torch_geometric.nn.pool import knn_graph


class KNNNeighborList(Transform):
    """
    Returns the k-nearest Neighbors.
    This class does not inherit from the Schnetpack Neighbor Transformations
    because we can process whole batches with this implementation,
    because of poptorch_geometric.knn_graph
    """
    def __init__(self, k):
        super(KNNNeighborList, self).__init__()
        self.k = int(k)

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        positions = inputs[properties.position]
        batch_tensor = inputs[properties.idx_m]

        pairs = knn_graph(
            positions,
            self.k + 1,
            batch=batch_tensor,
            loop=True
        )
        print(pairs.grad_fn)

        pairs = pairs.reshape(2, -1, self.k + 1)[:, :, 1:].reshape(2, -1)
        idx_i, idx_j = torch.unbind(pairs)

        # TODO constants offsets
        offsets = torch.tensor([[0, 0, 0]]).repeat(idx_i.shape[0], 1)

        inputs[properties.idx_i] = idx_i
        inputs[properties.idx_j] = idx_j
        inputs[properties.offsets] = offsets

        return inputs


#TODO this is only for testing:
def knn_neighborlist_transfo(positions, batch_tensor, k):

    pairs = knn_graph(
        positions,
        k + 1,
        batch=batch_tensor,
        loop=True
    )

    pairs = pairs.reshape(2, -1, k + 1)[:, :, 1:].reshape(2, -1)
    idx_i, idx_j = torch.unbind(pairs)

    # TODO constants offsets
    offsets = torch.tensor([[0, 0, 0]]).repeat(idx_i.shape[0], 1)

    return idx_i, idx_j, offsets
