import os
from typing import List

from schnetpack.md import Simulator
from schnetpack.md.simulation_hooks import SimulationHook, DataStream


class TensorLoggerError(Exception):
    """
    Exception for the FileLogger class.
    """

    pass


class TensorLogger(SimulationHook):
    """
    Class for monitoring the simulation and storing the resulting data to a hfd5 dataset. The properties to monitor are
    given via instances of the DataStream class. Uses buffers of a given size, which are accumulated and fushed to the
    main file in regular intervals in order to reduce I/O overhead. All arrays are initialized for the full number of
    requested simulation steps, the current positions in each data group is handled via the 'entries' attribute.

    Args:
        filename (str): Path to the hdf5 database file.
        buffer_size (int): Size of the buffer, once full, data is stored to the hdf5 dataset.
        data_streams list(schnetpack.simulation_hooks.DataStream): List of DataStreams used to collect and log
                                                                   information to the main hdf5 dataset, default are
                                                                   properties and molecules.
        every_n_steps (int): Frequency with which the buffer is updated.
        precision (int): Precision used for storing float data (16, 32, 64 bit, default 32).
    """

    def __init__(
        self,
        buffer_size: int,
        data_streams: List[DataStream] = [],
        every_n_steps: int = 1,
        precision: int = 32,
    ):
        super(TensorLogger, self).__init__()

        self.every_n_steps = every_n_steps
        self.buffer_size = buffer_size
        self.precision = precision

        # Precondition data streams
        self.data_steams = []
        for stream in data_streams:
            self.data_steams += [stream]

    def on_simulation_start(self, simulator: Simulator):
        """
        Initializes all present data streams (creating groups, determining buffer shapes, storing metadata, etc.). In
        addition, the 'entries' attribute of each data stream is read from the existing data set upon restart.

        Args:
            simulator (schnetpack.md.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        simulator.system.

        # Flag, if new database should be started or data appended to old one
        append_data = False

        # Check, whether file already exists
        if os.path.exists(self.filename):

            # If file exists and it is the first call of a simulator without restart,
            # raise and error.
            if (not simulator.restart) and (simulator.effective_steps == 0):
                raise TensorLoggerError(
                    "File {:s} already exists and simulation was not restarted.".format(
                        self.filename
                    )
                )

            # If either a restart is requested or the simulator has already been called,
            # append to file if it exists.
            if simulator.restart or (simulator.effective_steps > 0):
                append_data = True
        else:
            # If no file is found, automatically generate new one.
            append_data = False

        # Create the HDF5 file
        self.file = h5py.File(self.filename, "a", libver="latest")

        # Construct stream buffers and data groups
        for stream in self.data_steams:
            stream.init_data_stream(
                simulator,
                self.file,
                self.buffer_size,
                restart=append_data,
                every_n_steps=self.every_n_steps,
                precision=self.precision,
            )

            # Upon restart, get current position in file
            if append_data:
                self.file_position = stream.data_group.attrs["entries"]

        # Enable single writer, multiple reader flag
        self.file.swmr_mode = True

    def on_step_finalize(self, simulator: Simulator):
        """
        Update the buffer of each stream after each specified interval and flush the buffer to the main file if full.

        Args:
            simulator (schnetpack.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        if simulator.step % self.every_n_steps == 0:
            # If buffers are full, write to file
            if self.buffer_position == self.buffer_size:
                self._write_buffer()

            # Update stream buffers
            for stream in self.data_steams:
                stream.update_buffer(self.buffer_position, simulator)

            self.buffer_position += 1

    def on_simulation_end(self, simulator: Simulator):
        """
        Perform one final flush of the buffers and close the file upon the end of the simulation.

        Args:
            simulator (schnetpack.md.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        # Flush remaining data in buffer
        if self.buffer_position != 0:
            self._write_buffer()

        # Close database file
        self.file.close()

    def _write_buffer(self):
        """
        Write all current buffers to the database file.
        """
        for stream in self.data_steams:
            stream.flush_buffer(self.file_position, self.buffer_position)

        self.file_position += self.buffer_size
        self.buffer_position = 0
