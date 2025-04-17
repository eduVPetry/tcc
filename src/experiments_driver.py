from itertools import product
from typing import List, Tuple

from src.experiment import Experiment


well_name: str = "RO_31A"  # Well to use for experiments
facies: int = 7  # Facies to analyze

iterations: List[int] = [50, 100]   # Number of iterations to run
particles: List[int] = [10, 20]  # Number of particles in the swarm

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
    Experiment(
        _iterations,
        _particles,
        _cmode,
        c1_start,
        c1_end,
        c2_start,
        c2_end,
        well_name,
        facies
    ).run()
