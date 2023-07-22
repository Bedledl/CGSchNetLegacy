from utils.neighborlist import KNNNeighborList

from moleculekit.molecule import Molecule
from torchmd_cg.utils.psfwriter import pdb2psf_CA
import tempfile
import numpy as np
from ase.data import atomic_numbers, atomic_masses

from md.calculator import IPUSchNetPackCalc
from md.neighborlist_md import IPUNeighborListMD

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
    md_neighborlist = IPUNeighborListMD(
        cutoff,
        cutoff_shell,
        KNNNeighborList,
    )

    args = [model_path, "forces", "kcal/mol", "Angstrom", md_neighborlist]
    kwargs = {"energy_key": "energy", "required_properties": []}

    return IPUSchNetPackCalc(*args, **kwargs)


def get_molecule_obj(pdb_file):
    atom_level_mol = Molecule(pdb_file)

    with tempfile.NamedTemporaryFile(suffix=".psf") as psf_tmp_file:
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
