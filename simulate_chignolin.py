import math

from schnetpack import properties
from schnetpack.md import UniformInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.simulation_hooks import MoleculeStream, PropertyStream, FileLogger, LangevinThermostat

from md.integrator import VelocityVerletIPU
from md.system import IPUSystem
from simulation_utils import get_calculator, get_molecule_obj

import torch
import poptorch

from md.simulator import IPUSimulator

# TODO verlagere in anderen File
from train_chignolin import DummyCutoff, AddRequiredProps

cutoff = 5.
cutoff_shell = 2.
model_path = "train_chignolin/best_inference_model"
device = "cpu"
time_step = 0.5
pdb_file = "data/chignolin_cln025.pdb"
topology = "data/chignolin_ca_top.psf"
coordinates = "data/chignolin_ca_initial_coords.xtc"
forcefield = "data/chignolin_priors_fulldata.yaml"
initial_temperature = 300
log_file = "sim_chognolin_log.log"


def main():
    amino_mol = get_molecule_obj(pdb_file)

    md_calculator = get_calculator(
        model_path,
        device,
        cutoff,
        cutoff_shell,
        None,
    )

    md_integrator = VelocityVerletIPU(time_step)
    md_system = IPUSystem()

    md_system.load_molecules(amino_mol)

    # Initializes the system momenta according to a uniform distribution
    # scaled to the given temperature.
    md_initializer = UniformInit(
        initial_temperature,
        remove_center_of_mass=True,
        remove_translation=True,
        remove_rotation=True,
    )

    md_initializer.initialize_system(md_system)

    buffer_size = 100
    # Set up data streams to store positions, momenta and the energy
    data_streams = [
        MoleculeStream(store_velocities=True),
        PropertyStream(target_properties=[properties.energy]),
    ]

#    file_logger = FileLogger(
#        log_file,
#        buffer_size,
#        data_streams=data_streams,
#        every_n_steps=1,  # logging frequency
#        precision=32,  # floating point precision used in hdf5 database
#    )

    thermostat = LangevinThermostat(300, 10)

    # build simulator_hooks
    simulator_hooks = [
        thermostat,
#        file_logger
    ]

    # And now, we can create the Simulator Object which has pointers to all the
    # other components like Integrator or Calculator
    md_simulator = IPUSimulator(
        md_system,
        md_integrator,
        md_calculator,
        simulator_hooks=[],# todo for testing remove theromstat
        gradients_required=False
    )
    thermostat.on_simulation_start(md_simulator)

    print("start cpu run")
    #print(md_simulator(2))

    #md_simulator = md_simulator.to(torch.float32)

    print(next(md_simulator.calculator.model.parameters()).device)
    md_simulator = poptorch.inferenceModel(md_simulator)
    print(next(md_simulator.calculator.model.parameters()).device)

    print("start ipu run")
    result = md_simulator(2)
    print(next(md_simulator.calculator.model.parameters()).device)
    md_simulator.copyWeightsToHost()
    print(result)
    #md_simulator.simulate(10)

if __name__ == "__main__":
    main()
