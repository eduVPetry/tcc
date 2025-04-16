import os
import time

from experiment_utils import write_param_pso
from main_dm_las_forms_plots_separados import run_experiment

class Experiment:
    def __init__(self, iterations, particles, cmode, c1_start, c1_end, c2_start, c2_end):
        self.iterations = iterations
        self.particles = particles
        self.cmode = cmode
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end
        self.runtime = None
        self.generate_unique_id()

    def generate_unique_id(self):
        """Generate a unique ID based on parameters."""
        match self.cmode:
            case 0: cmode = "fixed"
            case 1: cmode = "linear"
            case 2: cmode = "exponential"
            case _: raise ValueError(f"Invalid cmode: {self.cmode}")

        self.id = (
            f"cmode={cmode}_"
            f"iter={self.iterations}_"
            f"p={self.particles}_"
            f"c1=[{self.c1_start},{self.c1_end}]_"
            f"c2=[{self.c2_start},{self.c2_end}]"
        )

    def run(self):
        start_time = time.perf_counter()

        write_param_pso(self.particles, self.iterations, self.cmode, self.c1_start, self.c1_end, self.c2_start, self.c2_end)

        run_experiment(self.id)

        end_time = time.perf_counter()
        self.runtime = end_time - start_time

    def save_results(self):
        directory = os.path.join("results", self.id)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "log.txt"), "w") as f:
            f.write(f"Experiment ran in {self.runtime:.3f} seconds.\n")


if __name__ == "__main__":
    experiment = Experiment(100, 20, 1, 2.0, 0.5, 0.5, 2.0)
    experiment.run()
    experiment.save_results()
