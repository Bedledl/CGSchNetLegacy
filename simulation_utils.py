import os
from typing import List, Type

import schnetpack.units
import torch

from ase.io import read
from ase.io.proteindatabank import read_proteindatabank

from schnetpack import properties
from schnetpack.md import System, UniformInit, Simulator
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.md.simulation_hooks import MoleculeStream, FileLogger, PropertyStream
from schnetpack.md.simulation_hooks.thermostats import LangevinThermostat
from schnetpack.transform import KNNNeighborList, ASENeighborList, CompleteNeighborList
from SchNetPackCalcIpu import MultiSimCalc, SchNetPackCalcCPU, SchNetPackCalcIPU

from moleculekit.molecule import Molecule
from torchmd_cg.utils.psfwriter import pdb2psf_CA
import tempfile
import numpy as np
from ase.data import atomic_numbers, atomic_masses

AA2INT = {
    'ALA': 1,
    'GLY': 2,
    'PHE': 3,
    'TYR': 4,
    'ASP': 5,
    'GLU': 6,
    'TRP': 7,
    'PRO': 8,
    'ASN': 9,
    'GLN': 10,
    'HIS': 11,
    'HSD': 11,
    'HSE': 11,
    'SER': 12,
    'THR': 13,
    'VAL': 14,
    'MET': 15,
    'CYS': 16,
    'NLE': 17,
    'ARG': 18,
    'LYS': 19,
    'LEU': 20,
    'ILE': 21
}


def get_calculator(model_path, device, cutoff, cutoff_shell, pipe_endpoint):
    md_neighborlist = NeighborListMD(
        cutoff,
        cutoff_shell,
        KNNNeighborList,
    )

    args = [model_path, "forces", "kcal/mol", "Angstrom", md_neighborlist]
    kwargs = {"energy_key": "energy", "required_properties": []}

    calculator_class = None
    if device is "cpu":
        if pipe_endpoint:
            calculator_class = MultiSimCalc
            kwargs["pipe_endpoint"] = pipe_endpoint
        else:
            calculator_class = SchNetPackCalcCPU
    else:
        # Todo multicsim ipu
        calculator_class = SchNetPackCalcIPU

    return calculator_class(*args, **kwargs)


def get_molecule_obj(pdb_file):
    atom_level_mol = Molecule(pdb_file)

    with tempfile.NamedTemporaryFile() as psf_tmp_file:
        pdb2psf_CA(pdb_file, psf_tmp_file.name)
        amino_level_mol = Molecule(psf_tmp_file.name)

    index = atom_level_mol.resid - 1
    n_atoms = atom_level_mol.numAtoms

    amino_level_mol.coords = np.zeros((amino_level_mol.numAtoms, 3, 1))
    amino_level_mol.masses = np.zeros(amino_level_mol.numAtoms)

    for i in range(n_atoms):
        amino_level_mol.coords[index[i]] += atom_level_mol.coords[i]
        atomic_number = atomic_numbers[atom_level_mol.element[i]]
        amino_level_mol.masses[index[i]] += atomic_masses[atomic_number]

    amino_indices, amino_atom_num = np.unique(index, return_counts=True)

    for amino_index, num in zip(amino_indices, amino_atom_num):
        amino_level_mol.coords[amino_index] = amino_level_mol.coords[amino_index]/num

    return amino_level_mol


def fill_system_with_amino_mol(system: System, mol: Molecule):
    """
    This method fills the schnetpack.System with the initial inforation about the simulated molecule.
    This is i usually done by the System.load_molecules method, but we don't have ASE objects
    in the coarse-graining case.
    """
    # fill in basic values from molecule
    system.n_replicas = 1
    system.n_molecules = 1,
    system.n_atoms = torch.tensor([mol.numAtoms])
    system.total_n_atoms = mol.numAtoms

    # initialize index vector for aggregation
    system.index_m = torch.zeros(system.total_n_atoms)
    # we have only one molecule, therefore index 0 for every atom/amino is fine

    # 3) Construct basic property arrays
    system.atom_types = torch.zeros(system.total_n_atoms)
    system.masses = torch.zeros(1, system.total_n_atoms, 1)

    # Relevant for dynamic properties: positions, momenta, forces
    system.positions = torch.zeros(system.n_replicas, system.total_n_atoms, 3)
    system.momenta = torch.zeros(system.n_replicas, system.total_n_atoms, 3)
    system.forces = torch.zeros(system.n_replicas, system.total_n_atoms, 3)

    system.energy = torch.zeros(system.n_replicas, system.n_molecules, 1)

    # Relevant for periodic boundary conditions and simulation cells
    system.cells = torch.zeros(system.n_replicas, system.n_molecules, 3, 3)
    system.stress = torch.zeros(system.n_replicas, system.n_molecules, 3, 3)
    system.pbc = torch.zeros(1, system.n_molecules, 3).bool()

    mass2internal = schnetpack.units.unit2internal(1.0) # out masses are in Dalton, which is standard
    positions2internal = schnetpack.units.unit2internal("Angstrom")
    # fill in values from molecule
    for index, aa_name in enumerate(mol.resname):
        system.atom_types[index] = AA2INT[aa_name]

        system.masses[0, index, 0] = mol.masses[index] * mass2internal

    system.positions[0] = mol.coords.squeeze() * positions2internal

    # TODO we do not need cells and pbc in out case so just leave it constant
