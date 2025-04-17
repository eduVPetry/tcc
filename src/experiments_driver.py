from itertools import product
from typing import List, Tuple

from src.experiment import Experiment

# Well to use for experiments
well: str = "RO_31A"  # Change this to use a different well

# Facies to analyze
facies: int = 7  # Change this to analyze a different facies

# Number of iterations to run for each experiment
iterations: List[int] = [50, 100]

# Number of particles in the swarm
particles: List[int] = [10, 20]

# Cognitive acceleration parameters (Exploration)
c1: List[Tuple[float, float]] = [
    (2.5, 0.5),  # Strong to weak
    (2.5, 1.5),  # Strong to moderate
    (2.0, 1.0),  # Moderate to medium
    (1.5, 0.5),  # Moderate to weak
]

# Social acceleration parameters (Exploitation)
c2: List[Tuple[float, float]] = [
    (0.5, 2.5),  # Weak to strong
    (1.5, 2.5),  # Moderate to strong
    (1.0, 2.0),  # Medium to moderate
    (0.5, 1.5),  # Weak to moderate
]

# Coefficient update modes
# 0: Fixed coefficients
# 1: Linear update
# 2: Exponential update
cmode: List[int] = [0, 1, 2]

# Generate and run all experiment combinations
for _iterations, _particles, (c1_start, c1_end), (c2_start, c2_end), _cmode in product(
    iterations, particles, c1, c2, cmode
):
    experiment = Experiment(
        _iterations,
        _particles,
        _cmode,
        c1_start,
        c1_end,
        c2_start,
        c2_end,
        well,
        facies
    )
    print(f"Running experiment for well {well}, facies {facies}: {experiment.id}")
    experiment.run()
    print(f"Experiment completed in {experiment.runtime:.2f} seconds")
    print(f"Best error achieved: {experiment.best_error:.6f}")
    print()
