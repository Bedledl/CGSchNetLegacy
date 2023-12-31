from typing import Union, List, Dict

import poptorch
import torch

from schnetpack.md import System
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.properties import n_molecules, idx_i, idx_j, offsets


class SchNetPackCalcIPU(SchNetPackCalculator):
    def __init__(
            self,
            model_file: str,
            force_key: str,
            energy_unit: Union[str, float],
            position_unit: Union[str, float],
            neighbor_list: NeighborListMD,
            energy_key: str = None,
            stress_key: str = None,
            required_properties: List = [],
            property_conversion: Dict[str, Union[str, float]] = {},
            script_model: bool = False,
    ):
        super(SchNetPackCalcIpu, self).__init__(
            model_file,
            force_key,
            energy_unit,
            position_unit,
            neighbor_list,
            energy_key,
            stress_key,
            required_properties,
            property_conversion,
            script_model
        )
        self.model.to(torch.float32)
        self.ipu_executor = poptorch.inferenceModel(self.model)
        self.idx_i = None
        self.idx_j = None
        self.offsets = None

    def calculate(self, system: System):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        This method overwrites the SchNetPackCalculator Method with the exact same logic,
        except, that the inference is calculated by calling the PoplarExecutor, generated by pytorch.inferenceModel.
        In this way the inference is calculated on the IPU instead of the CPU.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        inputs = self._generate_input(system)
        self.results = self.ipu_executor(inputs)
        self._update_system(system)

    def _generate_input(self, system: System) -> Dict[str, torch.Tensor]:
        inputs = super(SchNetPackCalcIpu, self)._generate_input(system)
        inputs[n_molecules] = system.n_molecules

        n_atoms = system.total_n_atoms
        if self.idx_i is None:
            self.idx_i = torch.arange(n_atoms).repeat_interleave(n_atoms)
        if self.idx_j is None:
            self.idx_j = torch.arange(n_atoms).repeat(n_atoms)
        if self.offsets is None:
            self.offsets = torch.tensor([[0, 0, 0]]).repeat(n_atoms * n_atoms, 1)

        inputs[idx_i] = self.idx_i
        inputs[idx_j] = self.idx_j
        inputs[offsets] = self.offsets
        return inputs


class MultiSimCalc(SchNetPackCalculator):
    def __init__(
            self,
            model_file: str,
            force_key: str,
            energy_unit: Union[str, float],
            position_unit: Union[str, float],
            neighbor_list: NeighborListMD,
            energy_key: str = None,
            stress_key: str = None,
            required_properties: List = [],
            property_conversion: Dict[str, Union[str, float]] = {},
            script_model: bool = False,
            pipe_endpoint=None,
    ):
        super(MultiSimCalc, self).__init__(
            model_file,
            force_key,
            energy_unit,
            position_unit,
            neighbor_list,
            energy_key,
            stress_key,
            required_properties,
            property_conversion,
            script_model
        )
        if pipe_endpoint is None:
            raise ValueError("pipe_endpoint is None in MultiSimCalc")
        self.pipe_endpoint = pipe_endpoint
        self.idx_i = None
        self.idx_j = None
        self.offsets = None

    def calculate(self, system: System):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        This method overwrites the SchNetPackCalculator Method with the exact same logic,
        except, that the inference is calculated by calling the PoplarExecutor, generated by pytorch.inferenceModel.
        In this way the inference is calculated on the IPU instead of the CPU.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        inputs = self._generate_input(system)
        self.pipe_endpoint.send(inputs)
        self.results = self.pipe_endpoint.recv()
        self._update_system(system)

    def _get_system_molecules(self, system: System):
        inputs = super(MultiSimCalc, self)._get_system_molecules(system)
        inputs[n_molecules] = system.n_molecules

        n_atoms = system.total_n_atoms
        if self.idx_i is None:
            self.idx_i = torch.arange(n_atoms).repeat_interleave(n_atoms)
        if self.idx_j is None:
            self.idx_j = torch.arange(n_atoms).repeat(n_atoms)
        if self.offsets is None:
            self.offsets = torch.tensor([[0, 0, 0]]).repeat(n_atoms * n_atoms, 1)

        inputs[idx_i] = self.idx_i
        inputs[idx_j] = self.idx_j
        inputs[offsets] = self.offsets
        return inputs


class SchNetPackCalcCPU(SchNetPackCalculator):
    def __init__(
            self,
            model_file: str,
            force_key: str,
            energy_unit: Union[str, float],
            position_unit: Union[str, float],
            neighbor_list: NeighborListMD,
            energy_key: str = None,
            stress_key: str = None,
            required_properties: List = [],
            property_conversion: Dict[str, Union[str, float]] = {},
            script_model: bool = False,
    ):
        super(SchNetPackCalcCPU, self).__init__(
            model_file,
            force_key,
            energy_unit,
            position_unit,
            neighbor_list,
            energy_key,
            stress_key,
            required_properties,
            property_conversion,
            script_model
        )
        self.idx_i = None
        self.idx_j = None
        self.offsets = None

        self.steps = 0
        self.d = 0

    def calculate(self, system: System):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        This method overwrites the SchNetPackCalculator Method with the exact same logic,
        except, that the inference is calculated by calling the PoplarExecutor, generated by pytorch.inferenceModel.
        In this way the inference is calculated on the IPU instead of the CPU.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        self._update_system(system)

    def _generate_input(self, system: System) -> Dict[str, torch.Tensor]:
        inputs = super(SchNetPackCalcCPU, self)._generate_input(system)
        inputs[n_molecules] = system.n_molecules

#        n_atoms = system.total_n_atoms
#        if self.idx_i is None:
#            self.idx_i = torch.arange(n_atoms).repeat_interleave(n_atoms)
#        if self.idx_j is None:
#            self.idx_j = torch.arange(n_atoms).repeat(n_atoms)
#        if self.offsets is None:
#            self.offsets = torch.tensor([[0, 0, 0]]).repeat(n_atoms * n_atoms, 1)
#
#        inputs[idx_i] = self.idx_i
#        inputs[idx_j] = self.idx_j
#        inputs[offsets] = self.offsets
        return inputs

