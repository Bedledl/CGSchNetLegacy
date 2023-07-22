from typing import Dict, List

import torch
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import NeighborListTransform


class IPUNeighborListMD(torch.nn.Module, NeighborListMD):
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
        torch.nn.Module.__init__(self)
        NeighborListMD.__init__(self,
        #super(IPUNeighborListMD, self).__init__(
            cutoff, cutoff_shell, base_nbl, requires_triples, collate_fn)

    def get_neighbors(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute neighbor indices from positions and simulations cells.

        Args:
            inputs (dict(str, torch.Tensor)): input batch.

        Returns:
            torch.tensor: indices of neighbors.
        """
        return self.transform(inputs)

        atom_types = inputs[properties.Z]
        positions = inputs[properties.R]
        n_atoms = inputs[properties.n_atoms]
        cells = inputs[properties.cell]
        pbc = inputs[properties.pbc]
        n_molecules = inputs[properties.n_molecules]

        input_batch = self._split_batch(
            atom_types, positions, 10, cells, pbc, n_molecules
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
        neighbor_idx = {p: neighbor_idx[p].to(device=positions.device) for p in neighbor_idx}

        # we leave out filter_indices()

        return neighbor_idx

    @staticmethod
    def _split_batch(
            atom_types: torch.Tensor,
            positions: torch.Tensor,
            n_atoms: int,
            cells: torch.Tensor,
            pbc: torch.Tensor,
            n_molecules: int,
    ) -> List[Dict[str, torch.tensor]]:
        """
        The original method calls .cpu(). This make obviously no sense when running on
        the ipu.

        Split the tensors containing molecular information into the different molecules for neighbor list computation.
        Args:
            atom_types (torch.Tensor): Atom type tensor.
            positions (torch.Tensor): Atomic positions.
            n_atoms int: every molecule has to have the same number of atoms
            cells (torch.Tensor): Simulation cells.
            pbc (torch.Tensor): Periodic boundary conditions used for each molecule.
            n_molecules (int): Number of molecules.

        Returns:
            list(dict(str, torch.Tensor))): List of input dictionaries for each molecule.
        """
        input_batch = []

        idx_c = 0
        # TODO poptorch for loop? probably not
        for idx_mol in range(n_molecules):
            curr_n_atoms = n_atoms
            inputs = {
                properties.n_atoms: torch.tensor([curr_n_atoms]),
                properties.Z: atom_types[idx_c: idx_c + curr_n_atoms],
                properties.R: positions[idx_c: idx_c + curr_n_atoms],
            }

            if cells is None:
                inputs[properties.cell] = None
                inputs[properties.pbc] = None
            else:
                inputs[properties.cell] = cells[idx_mol]
                inputs[properties.pbc] = pbc[idx_mol]

            idx_c += curr_n_atoms
            input_batch.append(inputs)

        return input_batch
