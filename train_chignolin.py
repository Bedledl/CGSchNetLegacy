import os
import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn
import torchmetrics
import torchmdnet
from torchmdnet import datasets
from torchmdnet.utils import make_splits
from torch.utils.data import Subset
import torch

import torch_geometric.data.data
from typing import Dict

n_atom_basis = 32
energy_key = "energy"
forces_key = "forces"
n_rbf = 20
cutoff = 5.
coords = "data/chignolin_ca_coords.npy"
forces = "data/chignolin_ca_forces.npy"
embeddings = "data/chignolin_ca_embeddings.npy"


class DummyCutoff(torch.nn.Identity):
    def __init__(self, cutoff):
        super(DummyCutoff, self).__init__()
        self.cutoff = cutoff


class AddRequiredProps(trn.Transform):
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        print("-----------------------------")
        print("Add Required Props")
        print(inputs)
        inputs[spk.properties.cell] = torch.zeros(1, 3, 3)
        inputs[spk.properties.pbc] = torch.zeros(1, 3).bool()
        print(inputs)
        print("")
        print("-------------------------------------")
        return inputs


def create_model():
    neighborlist = trn.KNNNeighborList(5)
    pairwise_distance = spk.atomistic.PairwiseDistances()
    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=energy_key)
    pred_forces = spk.atomistic.Forces(energy_key=energy_key, force_key=forces_key)
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=DummyCutoff(cutoff)
    )

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[AddRequiredProps(), neighborlist, pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            trn.CastTo64(),
#            trn.AddOffsets("energy", add_mean=True, add_atomrefs=False)
        ]
    )

    output_energy = spk.task.ModelOutput(
        name=energy_key,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.01,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name=forces_key,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.99,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    model = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_forces],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": 1e-4}
    )
    return model


class SchNetTorchMDDataModule(spk.data.AtomsDataModule):
    def __init__(self,
                 dataset,
                 batch_size,
                 transforms,
                 num_train,
                 num_val,
                 num_test,
                 num_workers,
                 pin_memory):
        super(SchNetTorchMDDataModule, self).__init__(
            datapath="dummy.db",
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=None,
            format=None,
            transforms=transforms,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        if dataset is None:
            raise AttributeError("Please add a torch_geometric.data.Dataset as dataset argument in kwargs.")

        self.dataset = dataset

    def setup(self, stage=None):
        self.train_idx, self.val_idx, self.test_idx = make_splits(
            len(self.dataset),
            self.num_train,
            self.num_val,
            self.num_test,
            1
        )
        self._train_dataset = Subset(self.dataset, self.train_idx)
        self._val_dataset = Subset(self.dataset, self.val_idx)
        self._test_dataset = Subset(self.dataset, self.test_idx)

        self._setup_transforms()


class CustomDatasetSchnetpack(torchmdnet.datasets.Custom):
    def get(self, idx):
        data_obj = super(CustomDatasetSchnetpack, self).get(idx)
        return {
            spk.properties.Z: data_obj.z,
            spk.properties.position: data_obj.pos,
            spk.properties.forces: data_obj.neg_dy,
            spk.properties.n_atoms: torch.tensor([len(data_obj.z)]),
            spk.properties.n_molecules: torch.tensor(1),
            spk.properties.idx_m: torch.zeros(len(data_obj.z))
        }


def create_chignolin_datamodule(coord_file, embed_file, force_file):
    transforms = [
#        trn.RemoveOffsets(energy_key, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ]

    dataset = CustomDatasetSchnetpack(coord_file, embed_file, forceglob=force_file)
    lightning_data_module = SchNetTorchMDDataModule(
        batch_size=32,
        num_train=0.7,
        num_val=0.2,
        num_test=0.1,
        transforms=transforms,
        num_workers=6,
        pin_memory=False,
        dataset=dataset
    )

    lightning_data_module.setup()

    return lightning_data_module


def main():
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join("train_chignolin", "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        ),
#        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]

#    tb_logger = pl.loggers.TensorBoardLogger("train_chignolin")
    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=4,
        default_root_dir="train_chignolin",
        callbacks=callbacks,
#        logger=tb_logger,
        precision=32,
        inference_mode=False
    )

    model = create_model()

    data_module = create_chignolin_datamodule(
        coord_file=coords,
        embed_file=embeddings,
        force_file=forces
    )

    trainer.fit(model,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader())

    # run test set after completing the fit
    trainer.test(model,
                 dataloaders=data_module.test_dataloader())
#    trainer.validate(model, validate)


if __name__ == "__main__":
    main()