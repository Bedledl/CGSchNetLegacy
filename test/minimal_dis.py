from typing import Dict

import torch
import poptorch
from schnetpack.transform import Transform

from example_schnet_input import example_schnet_input

from schnetpack import properties as props
from poptorch_geometric.ops import knn_graph

"""
This minimal example shows the problem of the IPU with index_select operations in the backward pass

"""


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
        positions = inputs[props.position]
        batch_tensor = inputs[props.idx_m]

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

        inputs[props.idx_i] = idx_i
        inputs[props.idx_j] = idx_j
        inputs[props.offsets] = offsets

        return inputs


class PairwiseDistancesIndex(torch.nn.Module):
    def forward(self, positions, indeces_i, indeces_j, offsets):
        pos_i = positions[indeces_i]
        pos_j = positions[indeces_j]
        Rij = pos_j - pos_i + offsets
        return Rij


class PairwiseDistancesIndexSelect(torch.nn.Module):
    def forward(self, positions, indeces_i, indeces_j, offsets):
        pos_i = torch.index_select(positions, 0, indeces_i)
        pos_j = torch.index_select(positions, 0, indeces_j)
        Rij = pos_j - pos_i + offsets
        return Rij


class NeighborDistanceModule(torch.nn.Module):
    def __init__(self, neighborlist, distance):
        super(NeighborDistanceModule, self).__init__()
        self.neighborlist = neighborlist
        self.distance = distance

    def forward(self, positions):
        positions.requires_grad_()
        inputs = {
            props.position: positions,
            props.idx_m: torch.zeros(positions.shape[0])
        }
        inputs = self.neighborlist(inputs)

        distances = self.distance(
            inputs[props.position],
            inputs[props.idx_i],
            inputs[props.idx_j],
            inputs[props.offsets]
        )
        distances_norm = distances.pow(2).sum(-1).sqrt()
        distances_norm_sum = [distances_norm.sum()]
        print(distances_norm_sum)
        grad = torch.autograd.grad(distances_norm_sum, [positions])
        return distances_norm, grad


class TopKModule(torch.nn.Module):
    def __init__(self, k):
        super(TopKModule, self).__init__()
        self.k = k

    def forward(self, positions, batch):
        positions.requires_grad_(True)
        print(positions.grad_fn)
        batch_x = batch
        batch_y = batch

        x = positions
        y = positions

        print(x.grad_fn)
        print(y.grad_fn)

        x_expanded = x.expand(y.size(0), *x.shape)
        y_expanded = y.reshape(y.size(0), 1, y.size(1))

        diff = x_expanded - y_expanded
        norm = diff.pow(2).sum(-1)
        # because we didn't filter out the loops yet, we have some 0 values here in the backward pass
        norm = torch.sqrt(norm + 1e-8)

        dist, col = torch.topk(norm,
                               k=self.k + 1,
                               dim=-1,
                               largest=False,
                               sorted=True)
        dist = dist.flatten()
        #row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, self.k)

        dist = dist.reshape(1, -1, self.k + 1)[:, :, 1:].reshape(1, -1)

        grad = torch.autograd.grad([dist.sum()], [positions])

        return dist, grad




dis_index = NeighborDistanceModule(KNNNeighborList(3), PairwiseDistancesIndex())
dis_index_select = NeighborDistanceModule(KNNNeighborList(3), PairwiseDistancesIndexSelect())


example_pos = example_schnet_input["_positions"]
example_batch = example_schnet_input[props.idx_m]


cpu_result_index = dis_index(example_pos)
cpu_result_index_select = dis_index_select(example_pos)

dis_index = poptorch.inferenceModel(dis_index)
dis_index_select = poptorch.inferenceModel(dis_index_select)

ipu_result_index = ""
ipu_result_index_select = ""
try:
    ipu_result_index = dis_index(example_pos)
except Exception as exc:
    print(exc)
try:
    ipu_result_index_select = dis_index_select(example_pos)
except Exception as exc:
    print(exc)

print(cpu_result_index)
print(cpu_result_index_select)
print(ipu_result_index)
print(ipu_result_index_select)

# And now the Topk Module:

topk_module = TopKModule(3)
cpu_result_topk = topk_module(example_pos, example_batch)

topk_module = poptorch.inferenceModel(topk_module)
ipu_result_topk = ""
try:
    ipu_result_topk = topk_module(example_pos, example_batch)
except Exception as exc:
    print(exc)

print(cpu_result_topk)
print(ipu_result_topk)
