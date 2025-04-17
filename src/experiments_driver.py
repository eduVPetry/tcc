from itertools import product
import numpy as np
import os
import pandas as pd
from typing import List, Tuple, Dict, Any
import traceback

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

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Track failed experiments
failed_experiments: List[Dict[str, Any]] = []

# Calculate total number of experiments
total_experiments = len(iterations) * len(particles) * len(c1) * len(c2) * len(cmode)
print(f"Starting {total_experiments} experiments...")

# Generate and run all experiments
for i, (_iterations, _particles, (c1_start, c1_end), (c2_start, c2_end), _cmode) in enumerate(
    product(iterations, particles, c1, c2, cmode)
):
    experiment_id = f"exp_{i+1}/{total_experiments}"
    print(f"\n{experiment_id}: Running experiment with parameters:")
    print(f"  - Iterations: {_iterations}")
    print(f"  - Particles: {_particles}")
    print(f"  - C1: [{c1_start}, {c1_end}]")
    print(f"  - C2: [{c2_start}, {c2_end}]")
    print(f"  - Cmode: {_cmode}")
    
    try:
        # Create and run the experiment
        experiment = Experiment(
            well_name,
            facies,
            _iterations,
            _particles,
            _cmode,
            c1_start,
            c1_end,
            c2_start,
            c2_end,
            seed=np.random.randint(0, 1000000)
        )
        experiment.run()
        print(f"  ✓ Experiment completed successfully")
    except Exception as e:
        error_msg = f"Error in experiment: {str(e)}"
        print(f"  ✗ {error_msg}")
        traceback.print_exc()
        
        # Record failed experiment
        failed_experiments.append({
            "iterations": _iterations,
            "particles": _particles,
            "c1_start": c1_start,
            "c1_end": c1_end,
            "c2_start": c2_start,
            "c2_end": c2_end,
            "cmode": _cmode,
            "error": str(e)
        })

# Print summary of results
print("\n" + "="*50)
print("EXPERIMENTS SUMMARY")
print("="*50)

# Check if results.csv exists and load it
results_path = os.path.join("results", "results.csv")
if os.path.exists(results_path):
    try:
        df = pd.read_csv(results_path)
        print(f"\nTotal experiments completed: {len(df)}")
        
        # Find best performing experiment
        if not df.empty and 'best_error' in df.columns:
            best_exp = df.loc[df['best_error'].idxmin()]
            print("\nBest performing experiment:")
            print(f"  - Well: {best_exp['well']}")
            print(f"  - Facies: {best_exp['facies']}")
            print(f"  - Iterations: {best_exp['iter']}")
            print(f"  - Particles: {best_exp['particles']}")
            print(f"  - Cmode: {best_exp['cmode']}")
            print(f"  - C1: [{best_exp['c1_start']}, {best_exp['c1_end']}]")
            print(f"  - C2: [{best_exp['c2_start']}, {best_exp['c2_end']}]")
            print(f"  - Best Error: {best_exp['best_error']:.6f}")
            print(f"  - Runtime: {best_exp['runtime']:.2f} seconds")
    except Exception as e:
        print(f"Error reading results: {str(e)}")

# Report failed experiments
if failed_experiments:
    print(f"\nFailed experiments: {len(failed_experiments)}")
    for i, exp in enumerate(failed_experiments):
        print(f"\n{i+1}. Parameters:")
        print(f"   - Iterations: {exp['iterations']}")
        print(f"   - Particles: {exp['particles']}")
        print(f"   - C1: [{exp['c1_start']}, {exp['c1_end']}]")
        print(f"   - C2: [{exp['c2_start']}, {exp['c2_end']}]")
        print(f"   - Cmode: {exp['cmode']}")
        print(f"   Error: {exp['error']}")
else:
    print("\nAll experiments completed successfully!")

print("\n" + "="*50)
