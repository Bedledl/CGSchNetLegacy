import schnetpack as spk
import schnetpack.transform as trn
import torch.nn

#from utils.DummyCutoff import DummyCutoff
from utils import KNNNeighborList, PairwiseDistancesIPU, ShiftedSoftplus, DummyCutoff


def create_model(
        n_atom_basis,
        n_rbf,
        k_neighbors,
        rbf_cutoff=5.,
        energy_key="energy",
        forces_key="forces"
):
    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=energy_key)
    #pred_forces = spk.atomistic.Forces(energy_key=energy_key, force_key=forces_key)
    neighborlist = KNNNeighborList(k_neighbors)
    pairwise_distance = PairwiseDistancesIPU()

    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=rbf_cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=DummyCutoff(rbf_cutoff),
        activation=ShiftedSoftplus(),
    )

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[neighborlist, pairwise_distance],
        output_modules=[pred_energy],
        postprocessors=[
            trn.CastTo64(),
        ]
    )

    return nnpot
