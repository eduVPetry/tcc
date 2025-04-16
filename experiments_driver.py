from itertools import product

from experiment import Experiment

iterations = [50, 100]
particles = [10, 20]

c1 = [           # Exploration:
    (2.5, 0.5),  # Strong to weak
    (2.5, 1.5),  # Strong to moderate
    (2.0, 1.0),  # Moderate to medium
    (1.5, 0.5),  # Moderate to weak
]

c2 = [           # Exploitation:
    (0.5, 2.5),  # Weak to strong
    (1.5, 2.5),  # Moderate to strong
    (1.0, 2.0),  # Medium to moderate
    (0.5, 1.5),  # Weak to moderate
]

cmode = [0, 1, 2]

for _iterations, _particles, (c1_start, c1_end), (c2_start, c2_end), _cmode in product(iterations, particles, c1, c2, cmode):
    experiment = Experiment(_iterations, _particles, c1_start, c1_end, c2_start, c2_end, _cmode)
    print(experiment.id)
    print()
