from schnetpack import properties
from schnetpack.md import System, UniformInit, Simulator
from schnetpack.md.integrators import Integrator
from schnetpack.md.simulation_hooks import MoleculeStream, PropertyStream, FileLogger, LangevinThermostat

from simulation_utils import get_calculator, get_molecule_obj, fill_system_with_amino_mol


cutoff = 5.
cutoff_shell = 2.
model_path = "train_chignolin/best_inference_model"
device = "cpu"
time_step = 0.5
pdb_file = "chignolin_cln025.pdb"
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
        None
    )

    md_integrator = Integrator(time_step)
    md_system = System()
    fill_system_with_amino_mol(md_system, amino_mol)

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

    file_logger = FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams,
        every_n_steps=1,  # logging frequency
        precision=32,  # floating point precision used in hdf5 database
    )

    thermostat = LangevinThermostat(300, 10)

    # build simulator_hooks
    simulator_hooks = [
        thermostat,
        file_logger
    ]

    # And now, we can create the Simulator Object which has pointers to all the
    # other components like Integrator or Calculator
    md_simulator = Simulator(
        md_system,
        md_integrator,
        md_calculator,
        simulator_hooks=simulator_hooks
    )

    md_simulator.simulate(10)
