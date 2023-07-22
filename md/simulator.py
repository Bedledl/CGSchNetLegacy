"""
This class defines a subclass of the schnetpack simulator that is optimised for
the use on the Graphcore IPU with Poptorch. The most important change is that
we now use the forward method to run the simulation instead of the simulate()
method and use the poptorch.for_loop method as main simulation loop.
"""
import torch
from schnetpack.md import System, Simulator
from contextlib import nullcontext


class IPUSimulator(Simulator):
    def __init__(self,
                 system: System,
                 integrator,
                 calculator,
                 simulator_hooks: list = [],
                 step: int = 0,
                 restart: bool = False,
                 gradients_required: bool = False,
                 progress: bool = True,
                 ):
        torch.nn.Module.__init__(self)
        self.register_module("system", system)
        self.register_module("integrator", integrator)
        self.register_module("calculator", calculator)
        self.register_module("simulator_hooks", torch.nn.ModuleList(simulator_hooks))
        self.step = step
        self.n_steps = None
        self.restart = restart
        self.gradients_required = gradients_required
        self.progress = progress

        # Keep track of the actual simulation steps performed with simulate calls
        self.effective_steps = 0

    def simulate(self, n_steps: int):
        raise NotImplementedError("Please use the forward method to start the simulation.")

    def __simulation_loop(self, positions, momenta, forces, energy, atom_type,
                          n_atoms, n_mol, index_m, pbc, cells, total_n_atoms):
        #for hook in self.simulator_hooks:
        #    hook.on_step_begin(self)

        # Do half step momenta
        #self.integrator.half_step(self.system)

        # Do propagation MD/PIMD
        #self.integrator.main_step(self.system)

        # Compute new forces
        #self.calculator.calculate(self.system)

        # Call hook after forces
        #for hook in self.simulator_hooks:
        #    hook.on_step_middle(self)

        # Do half step momenta
        #self.integrator.half_step(self.system)

        # Call hooks after second half step
        # Hooks are called in reverse order to guarantee symmetry of
        # the propagator when using thermostat and barostats
        #for hook in self.simulator_hooks[::-1]:
        #    hook.on_step_end(self)

        # Logging hooks etc
        #for hook in self.simulator_hooks:
        #    hook.on_step_finalize(self)

        #self.step += 1
        #self.effective_steps += 1

        return self.calculator.calculate_in_loop(positions, momenta, forces, energy,
                                                 atom_type, n_atoms, n_mol, index_m, pbc, cells, total_n_atoms)

    def forward(self, n_steps):
        #with torch.autograd.profiler.profile(with_stack=True, profile_memory=True,
        #                                     experimental_config=torch._C._profiler._ExperimentalConfig(
        #                                             verbose=True)) as prof:
        self.n_steps = n_steps

        progressbar = None
        #if self.progress:
        #   progressbar = trange
        # TODO progressbar

        # Check, if computational graph should be built
        if self.gradients_required:
            grad_context = torch.no_grad()
        else:
            grad_context = nullcontext()

        with grad_context:
            # Perform initial computation of forces
            #self.calculator.calculate(self.system)

            # Call hooks at the simulation start
            for hook in self.simulator_hooks:
                hook.on_simulation_start(self)

            loop_inputs = [
                self.system.positions[0],#0 isc crrent replication
                self.system.momenta[0],
                self.system.forces[0],
                self.system.energy[0],
                self.system.atom_types,
                self.system.n_atoms,
                torch.tensor(self.system.n_molecules),
                self.system.index_m,
                self.system.pbc[0],
                self.system.cells[0],
                torch.tensor(self.system.total_n_atoms)
            ]
            #loop_inputs = self.__simulation_loop(*loop_inputs)
            #loop_inputs = poptorch.for_loop(3, self.__simulation_loop, loop_inputs)
            for _ in range(3):
                loop_inputs = self.__simulation_loop(*loop_inputs)
            return loop_inputs

            #for _ in range(1):
            #self.calculator.calculate(self.system)

            #self.__simulation_loop(torch.tensor(1))
            #poptorch.for_loop(1, self.__simulation_loop, [torch.tensor(1)])

            # Call hooks at the simulation end
 #           for hook in self.simulator_hooks:
#                hook.on_simulation_end(self)

        #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))

    def state_dict(self, *args, **kwargs):
        return torch.nn.Module.state_dict(self, *args, **kwargs)
