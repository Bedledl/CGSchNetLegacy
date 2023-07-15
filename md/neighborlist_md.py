from typing import Dict

import torch
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import NeighborListTransform


class IPUNeighborListMD(NeighborListMD, torch.nn.Module):
    """
    Currently this implementation updates the neighborlist at every step.
    We can implement a condition with poptorch cond.conditions that
    takes the cutoff_shell into consideration.

    One big difference between this implementation and the NeighborListMD is
    that we have a static number of molecules, that is given in the inputs[n_molecules]
    constant.
    """
    def __init__(
            self,
            cutoff: float,
            cutoff_shell: float,
            base_nbl: NeighborListTransform,
            requires_triples: bool = False,
            collate_fn: callable = _atoms_collate_fn,
    ):
        super(IPUNeighborListMD, self).__init__(
            cutoff, cutoff_shell, base_nbl, requires_triples, collate_fn)

    def get_neighbors(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute neighbor indices from positions and simulations cells.

        Args:
            inputs (dict(str, torch.Tensor)): input batch.

        Returns:
            torch.tensor: indices of neighbors.
        """
        atom_types = inputs[properties.Z]
        positions = inputs[properties.R]
        n_atoms = inputs[properties.n_atoms]
        cells = inputs[properties.cell]
        pbc = inputs[properties.pbc]
        n_molecules = inputs[properties.n_molecules]

        input_batch = self._split_batch(
            atom_types, positions, n_atoms, cells, pbc, n_molecules
        )

        if self.molecular_indices is None:
            self.molecular_indices = [{} for _ in range(n_molecules)]

        for idx in range(n_molecules):
            # Get neighbors and if necessary triple indices
            self.molecular_indices[idx] = self.transform(input_batch[idx])

            # Remove superfluous entries before aggregation
            del self.molecular_indices[idx][properties.R]
            del self.molecular_indices[idx][properties.Z]
            del self.molecular_indices[idx][properties.cell]
            del self.molecular_indices[idx][properties.pbc]

        neighbor_idx = self._collate(self.molecular_indices)

        # Remove n_atoms
        del neighbor_idx[properties.n_atoms]

        # Move everything to correct device
        neighbor_idx = {p: neighbor_idx[p].to(positions.device) for p in neighbor_idx}

        # we leave out filter_indices()

        return neighbor_idx

    def to(self, device):
        new_self = super(IPUNeighborListMD, self).to(device)
        new_self.previous_positions = new_self.previous_positions.to(device)
        return new_self
