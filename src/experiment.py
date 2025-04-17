import os
import time
from typing import Optional
from datetime import datetime

from src.experiment_utils import write_param_pso
from src.main_dm_las_forms_plots_separados import run_experiment


class Experiment:
    """
    A class representing a PSO experiment with specific parameters.
    
    This class manages the execution of a PSO experiment with given parameters,
    tracks its runtime, and saves the results.
    
    Attributes:
        iterations (int): Number of iterations to run
        particles (int): Number of particles in the swarm
        cmode (int): Coefficient update mode (0=fixed, 1=linear, 2=exponential)
        c1_start (float): Initial cognitive acceleration
        c1_end (float): Final cognitive acceleration
        c2_start (float): Initial social acceleration
        c2_end (float): Final social acceleration
        well_name (str): Name of the well (LAS file) to use
        facies (int): Facies number to select for analysis
        runtime (Optional[float]): Time taken to run the experiment in seconds
        best_error (Optional[float]): Best error achieved by the swarm
        log_buffer (list): Buffer to store log messages
        id (str): Unique identifier for this experiment
    """
    
    def __init__(
        self,
        iterations: int,
        particles: int,
        cmode: int,
        c1_start: float,
        c1_end: float,
        c2_start: float,
        c2_end: float,
        well_name: str,
        facies: int
    ) -> None:
        """
        Initialize a new experiment with the given parameters.
        
        Args:
            iterations: Number of iterations to run
            particles: Number of particles in the swarm
            cmode: Coefficient update mode (0=fixed, 1=linear, 2=exponential)
            c1_start: Initial cognitive acceleration
            c1_end: Final cognitive acceleration
            c2_start: Initial social acceleration
            c2_end: Final social acceleration
            well_name: Name of the well (LAS file) to use, without extension
            facies: Facies number to select for analysis
        """
        self.iterations = iterations
        self.particles = particles
        self.cmode = cmode
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end
        self.well_name = well_name
        self.facies = facies
        self.runtime: Optional[float] = None
        self.best_error: Optional[float] = None
        self.log_buffer = []
        self.generate_unique_id()

    def generate_unique_id(self) -> None:
        """
        Generate a unique ID for this experiment based on its parameters.
        
        The ID includes the coefficient mode, number of iterations, number of particles,
        and the ranges for cognitive and social accelerations.
        """
        match self.cmode:
            case 0:
                cmode = "fixed"
            case 1:
                cmode = "linear"
            case 2:
                cmode = "exponential"
            case _:
                raise ValueError(f"Invalid cmode: {self.cmode}")

        self.id = (
            f"well={self.well_name}_"
            f"facies={self.facies}_"
            f"cmode={cmode}_"
            f"iter={self.iterations}_"
            f"p={self.particles}_"
            f"c1=[{self.c1_start},{self.c1_end}]_"
            f"c2=[{self.c2_start},{self.c2_end}]"
        )

    def log(self, message: str) -> None:
        """Add a message to the log buffer.
        
        Args:
            message: Message to be logged
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_buffer.append(f"[{timestamp}] {message}")
        
    def write_logs(self) -> None:
        """Write the log buffer to a file."""
        if not self.log_buffer:
            return

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Write logs to file
        log_file = os.path.join("results", f"{self.id}", "log.txt")
        with open(log_file, "w") as f:
            f.write("\n".join(self.log_buffer))

    def run(self) -> None:
        """
        Execute the experiment.
        
        This method:
        1. Records the start time
        2. Writes the PSO parameters to the config file
        3. Runs the experiment
        4. Records the end time and calculates runtime
        """
        write_param_pso(
            self.particles,
            self.iterations,
            self.cmode,
            self.c1_start,
            self.c1_end,
            self.c2_start,
            self.c2_end
        )

        # Log start of experiment
        self.log(f"Starting experiment {self.id}")
        self.log(f"Parameters: {self.__dict__}")

        start_time = time.time()
        self.best_error, self.log_buffer = run_experiment(self.id, self.well_name, self.facies, self.log_buffer)
        end_time = time.time()
        
        self.runtime = end_time - start_time        

        # Log end of experiment and runtime
        self.log(f"Best error: {self.best_error}")
        self.log(f"Runtime: {self.runtime:.2f} seconds")

        self.write_logs()

    def save_results(self) -> None:
        """
        Save the experiment results to a log file.
        
        Creates a directory for the experiment results if it doesn't exist,
        and writes the runtime information to a log file.
        """
        directory = os.path.join("results", self.id)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "log.txt"), "w") as f:
            f.write(f"Experiment ran in {self.runtime:.3f} seconds.\n")


if __name__ == "__main__":
    experiment = Experiment(100, 20, 1, 2.0, 0.5, 0.5, 2.0)
    experiment.run()
    experiment.save_results()
