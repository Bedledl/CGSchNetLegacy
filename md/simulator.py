"""
This class defines a subclass of the schnetpack simulator that is optimised for
the use on the Graphcore IPU with Poptorch. The most important change is that
we now use the forward method to run the simulation instead of the simulate()
method and use the poptorch.for_loop method as main simulation loop.
"""
import torch
import poptorch
from schnetpack import Simulator
from contextlib import nullcontext

from tqdm import trange


class IPUSimulator(torch.nn.Module):
    def simulate(self, n_steps: int):
        raise NotImplementedError("Please use the forward method to start the simulation.")

    def __simulation_loop(self):
        for hook in self.simulator_hooks:
            hook.on_step_begin(self)

        # Do half step momenta
        self.integrator.half_step(self.system)

        # Do propagation MD/PIMD
        self.integrator.main_step(self.system)

        # Compute new forces
        self.calculator.calculate(self.system)

        # Call hook after forces
        for hook in self.simulator_hooks:
            hook.on_step_middle(self)

        # Do half step momenta
        self.integrator.half_step(self.system)

        # Call hooks after second half step
        # Hooks are called in reverse order to guarantee symmetry of
        # the propagator when using thermostat and barostats
        for hook in self.simulator_hooks[::-1]:
            hook.on_step_end(self)

        # Logging hooks etc
        for hook in self.simulator_hooks:
            hook.on_step_finalize(self)

        self.step += 1
        self.effective_steps += 1

    def forward(self, n_steps):
        self.n_steps = n_steps

        progressbar = None
        if self.progress:
            progressbar = trange

        # Check, if computational graph should be built
        if self.gradients_required:
            grad_context = torch.no_grad()
        else:
            grad_context = nullcontext()

        with grad_context:
            # Perform initial computation of forces
            self.calculator.calculate(self.system)

            # Call hooks at the simulation start
            for hook in self.simulator_hooks:
                hook.on_simulation_start(self)

            poptorch.for_loop(self.n_steps, self.__simulation_loop())

            # Call hooks at the simulation end
            for hook in self.simulator_hooks:
                hook.on_simulation_end(self)

        return self.system.positions

    def to(self, device):
        new_self = super(IPUSimulator, self).to(device)
        new_self.system = new_self.system.to(device)
        new_self.calculator = new_self.calculator.to(device)
        return new_self
