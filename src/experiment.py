from datetime import datetime
import fcntl
import numpy as np
import os
import time
import csv
from typing import Optional, List

from src.experiment_utils import write_param_pso
from src.main_dm_las_forms_plots_separados import run_experiment


class Experiment:
    """
    A class representing a PSO experiment with specific parameters.
    
    This class manages the execution of a PSO experiment with given parameters,
    tracks its runtime, and saves the results.
    
    Attributes:
        well_name (str): Name of the well (LAS file) to use
        facies (int): Facies number to select for analysis
        iterations (int): Number of iterations to run
        particles (int): Number of particles in the swarm
        cmode (int): Coefficient update mode (0=fixed, 1=linear, 2=exponential)
        c1_start (float): Initial cognitive acceleration
        c1_end (float): Final cognitive acceleration
        c2_start (float): Initial social acceleration
        c2_end (float): Final social acceleration
        runtime (Optional[float]): Time taken to run the experiment in seconds
        best_error (Optional[float]): Best error achieved by the swarm
        log_buffer (List[str]): Buffer to store log messages
        id (str): Unique identifier for this experiment
        seed (int): Random seed for the experiment
    """
    
    def __init__(
        self,
        well_name: str,
        facies: int,
        iterations: int,
        particles: int,
        cmode: int,
        c1_start: float,
        c1_end: float,
        c2_start: float,
        c2_end: float,
        seed: Optional[int] = None
    ) -> None:
        self.well_name = well_name
        self.facies = facies
        self.iterations = iterations
        self.particles = particles
        self.cmode = cmode
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.runtime: Optional[float] = None
        self.best_error: Optional[float] = None
        self.log_buffer: List[str] = []
        self.id = self.generate_unique_id()

    def generate_unique_id(self) -> str:
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

        return (
            f"well={self.well_name}_"
            f"facies={self.facies}_"
            f"iter={self.iterations}_"
            f"p={self.particles}_"
            f"cmode={cmode}_"
            f"c1=[{self.c1_start},{self.c1_end}]_"
            f"c2=[{self.c2_start},{self.c2_end}]"
            f"_seed={self.seed}"
        )

    def log(self, message: str) -> None:
        """
        Add a message to the log buffer.
        
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
        
        # Create experiment-specific directory
        experiment_dir = os.path.join("results", f"{self.id}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Write logs to file
        log_file = os.path.join(experiment_dir, "log.txt")
        with open(log_file, "w") as f:
            f.write("\n".join(self.log_buffer))

    def _ensure_results_csv_header(self) -> None:
        """Create the results.csv file with headers if it doesn't exist."""
        csv_path = os.path.join("results", "results.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'well', 'facies', 'iter', 'particles', 'cmode', 'c1_start',
                    'c1_end', 'c2_start', 'c2_end', 'best_error', 'runtime', 'seed'
                ])

    def _append_to_results_csv(self) -> None:
        """Append this experiment's results to the results.csv file."""
        csv_path = os.path.join("results", "results.csv")
        with open(csv_path, 'a', newline='') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.writer(f)
            writer.writerow([
                self.well_name,
                self.facies,
                self.iterations,
                self.particles,
                self.cmode,
                self.c1_start,
                self.c1_end,
                self.c2_start,
                self.c2_end,
                self.best_error,
                self.runtime,
                self.seed
            ])
            fcntl.flock(f, fcntl.LOCK_UN)

    def run(self) -> None:
        """
        Execute the experiment.
        
        This method:
        1. Records the start time
        2. Writes the PSO parameters to the config file
        3. Runs the experiment
        4. Records the end time and calculates runtime
        5. Saves results to both experiment-specific directory and results.csv
        """
        try:
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
            self.log(f"Using random seed: {self.seed}")

            start_time = time.time()
            self.best_error = run_experiment(
                self.id, 
                self.well_name, 
                self.facies, 
                log_callback=self.log,  # Pass the log method as a callback
                seed=self.seed
            )
            end_time = time.time()
            
            self.runtime = end_time - start_time        

            # Log end of experiment and runtime
            self.log(f"Best error: {self.best_error}")
            self.log(f"Runtime: {self.runtime:.2f} seconds")

            # Write logs to experiment-specific directory
            self.write_logs()

            # Ensure results.csv exists and append this experiment's results
            self._ensure_results_csv_header()
            self._append_to_results_csv()

        except Exception as e:
            self.log(f"Experiment failed: {e}")
            raise


if __name__ == "__main__":
    experiment = Experiment("RO_31A", 7, 50, 20, 1, 2.0, 0.5, 0.5, 2.0, 123456)
    experiment.run()
