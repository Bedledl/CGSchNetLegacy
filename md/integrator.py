from schnetpack.md import System
from schnetpack.md.integrators import VelocityVerlet


class VelocityVerletIPU(VelocityVerlet):
    def half_step(self, system: System):
        """
        Half steps propagating the system momenta according to:

        ..math::
            p = p + \frac{1}{2} F \delta t

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        system.momenta = system.momenta + 0.5 * system.forces * self.time_step

    def _main_step(self, system: System):
        positions = (
                system.positions + self.time_step * system.momenta / system.masses
        )
        system.positions = positions
