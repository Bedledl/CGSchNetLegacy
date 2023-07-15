"""
This file contains a subclass of the schnetpack.system class.
In this state the simulation ignores pbc or cells.
"""
import torch

from typing import Union, List

from moleculekit.molecule import Molecule
from schnetpack.md import System
import schnetpack.units
from schnetpack.md.utils import NormalModeTransformer

from simulation_utils import AA2INT


class IPUSystem(System):
    def __init__(
            self, normal_mode_transform: NormalModeTransformer = NormalModeTransformer
    ):
        super(IPUSystem, self).__init__(normal_mode_transform)
        self.index_m = self.index_m.to(torch.int32)
        self.n_atoms = self.n_atoms.to(torch.int32)
        self.atom_types = self.atom_types.to(torch.int32)
        self._dd_dummy = self._dd_dummy.to(torch.float32)

    def load_molecules(self,
                       molecules: Union[Molecule, List[Molecule]],
                       n_replicas: int = 1,
                       position_unit_input: Union[str, float] = "Angstrom",
                       mass_unit_input: Union[str, float] = 1.0,
                       ):
        """
        This method fills the System with the initial inforation about the simulated molecule.
        The original SchNetPack.System loads ASE objects, which are not suitable for coarse graining.
        """
        if n_replicas > 1:
            raise NotImplementedError("Multiple replicas are not implemented.")

        self.n_replicas = n_replicas

        if isinstance(molecules, Molecule):
            molecules = [molecules]

        self.n_molecules = len(molecules)

        if self.n_molecules != 1:
            raise NotImplementedError("Multiple molecules are currently not tested.")

        self.n_atoms = torch.zeros(self.n_molecules, dtype=torch.int32)

        for i in range(self.n_molecules):
            self.n_atoms[i] = molecules[i].numAtoms

        self.total_n_atoms = torch.sum(self.n_atoms).item()

        self.index_m = torch.zeros(self.total_n_atoms, dtype=torch.int32)

        # 3) Construct basic property arrays
        self.atom_types = torch.zeros(self.total_n_atoms, dtype=torch.int32)
        self.masses = torch.zeros(1, self.total_n_atoms, 1)

        # Relevant for dynamic properties: positions, momenta, forces
        self.positions = torch.zeros(self.n_replicas, self.total_n_atoms, 3)
        self.momenta = torch.zeros(self.n_replicas, self.total_n_atoms, 3)
        self.forces = torch.zeros(self.n_replicas, self.total_n_atoms, 3)

        self.energy = torch.zeros(self.n_replicas, self.n_molecules, 1)

        # Relevant for periodic boundary conditions and simulation cells
        self.cells = torch.zeros(self.n_replicas, self.n_molecules, 3, 3)
        self.stress = torch.zeros(self.n_replicas, self.n_molecules, 3, 3)
        self.pbc = torch.zeros(1, self.n_molecules, 3).bool()

        mass2internal = schnetpack.units.unit2internal(1.0)  # out masses are in Dalton, which is standard
        positions2internal = schnetpack.units.unit2internal("Angstrom")

        offset = 0
        for mol_num, mol in enumerate(molecules):
            self.positions[0][offset: offset + mol.numAtoms] = \
                torch.from_numpy(mol.coords.squeeze()) * positions2internal

            for index, aa_name in enumerate(mol.resname):
                self.atom_types[offset + index] = AA2INT[aa_name]
                self.masses[0, offset + index, 0] = mol.masses[index] * mass2internal
                self.index_m[offset + index] = mol_num
            else:
                offset += index + 1

        # TODO we do not need cells and pbc in out case so just leave it constant

        # Set normal mode transformer
        self.nm_transform = self._nm_transformer(n_replicas)

    def to(self, device):
        new_self = super(IPUSystem, self).to(device)
        new_self._dd_dummy = new_self._dd_dummy.to(device)
        return new_self
