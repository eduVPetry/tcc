from itertools import product
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
import traceback
from typing import List, Tuple, Dict, Any

from src.experiment import Experiment


workers: int = max(1, os.cpu_count() - 1)  # Number of parallel workers

well_name: str = "RO_31A"  # Well to use for experiments
facies: int = 7  # Which facies to be analyzed

iterations: List[int] = [50, 100]   # Number of iterations to run
particles: List[int] = [10, 20]  # Number of particles in the swarm

# Fixed coefficients for cmode=0
fixed_c1_values: List[float] = [2.5, 2.0, 1.5]
fixed_c2_values: List[float] = [0.5, 1.0, 1.5]

# Dynamic cognitive acceleration coefficients (Exploration)
c1: List[Tuple[float, float]] = [
    (2.5, 0.5),  # Strong to weak
    (2.5, 1.5),  # Strong to moderate
    (2.0, 1.0),  # Moderate to medium
    (1.5, 0.5),  # Moderate to weak
]

# Dynamic social acceleration coefficients (Exploitation)
c2: List[Tuple[float, float]] = [
    (0.5, 2.5),  # Weak to strong
    (1.5, 2.5),  # Moderate to strong
    (1.0, 2.0),  # Medium to moderate
    (0.5, 1.5),  # Weak to moderate
]

# Coefficient update modes
# 0: Fixed
# 1: Linear update
# 2: Exponential update
cmode: List[int] = [0, 1, 2]

# Number of repetitions for each parameter combination
num_repetitions: int = 30

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Track failed experiments
failed_experiments: List[Dict[str, Any]] = []


def run_experiment(
    progress: str,
    iterations: int,
    particles: int,
    cmode: int,
    c1_start: float,
    c1_end: float,
    c2_start: float,
    c2_end: float,
    rep: int,
    total_reps: int
) -> None:
    """Run an experiment with the given parameters."""

    # Ensure unique random seed per process using time and PID to avoid identical seeds in multiprocessing
    np.random.seed(int(time.time() * 1000000) % 2**32 + os.getpid())

    print(f"\n{progress}: Running experiment with {'fixed' if cmode == 0 else 'varying'} parameters:")
    print(f"  - Iterations: {iterations}")
    print(f"  - Particles: {particles}")
    if cmode == 0:
        print(f"  - C1: {c1_start} (fixed)")
        print(f"  - C2: {c2_start} (fixed)")
    else:
        print(f"  - C1: [{c1_start}, {c1_end}]")
        print(f"  - C2: [{c2_start}, {c2_end}]")
    print(f"  - Cmode: {cmode}")
    print(f"  - Repetition: {rep + 1}/{total_reps}")
    
    try:
        experiment = Experiment(
            well_name,
            facies,
            iterations,
            particles,
            cmode,
            c1_start,
            c1_end,
            c2_start,
            c2_end,
            seed=np.random.randint(0, 1000000)
        )
        experiment.run()
        print(f"\n✓ {progress} completed successfully")
    except Exception as e:
        error_msg = f"Error in {progress}: {str(e)}"
        print(f"\n✗ {error_msg}")
        traceback.print_exc()
        
        # Record failed experiment
        failed_experiments.append({
            "iterations": iterations,
            "particles": particles,
            "c1_start": c1_start,
            "c1_end": c1_end,
            "c2_start": c2_start,
            "c2_end": c2_end,
            "cmode": cmode,
            "repetition": rep + 1,
            "error": str(e)
        })

def print_results_summary() -> None:
    """Print a summary of the experiment results."""
    print("\n" + "="*50)
    print("EXPERIMENTS SUMMARY")
    print("="*50)

    # Check if results.csv exists and load it
    results_path = os.path.join("results", "results.csv")
    if os.path.exists(results_path):
        try:
            df = pd.read_csv(results_path)
            print(f"\nTotal experiments completed: {len(df)}")
            
            # Group by parameter combination and calculate statistics
            if not df.empty and 'best_error' in df.columns:
                # Create a parameter combination identifier
                df['param_combo'] = df.apply(
                    lambda x: f"iter={x['iter']},part={x['particles']},cmode={x['cmode']},"
                             f"c1={x['c1_start']}-{x['c1_end']},c2={x['c2_start']}-{x['c2_end']}", 
                    axis=1
                )
                
                # Calculate statistics for each parameter combination
                stats = df.groupby('param_combo').agg({
                    'best_error': ['mean', 'std', 'min'],
                    'runtime': ['mean', 'std']
                }).round(6)
                
                print("\nStatistics for each parameter combination:")
                print(stats)
                
                # Find best performing experiment
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
            print(f"   - Repetition: {exp['repetition']}")
            print(f"   Error: {exp['error']}")
    else:
        print("\nAll experiments completed successfully!")

    print("\n" + "="*50)


def build_param_list() -> List[Tuple[str, int, int, int, float, float, float, float, int, int]]:
    param_list = []
    total_experiments = (
        len(iterations) * len(particles) * 
        (len(fixed_c1_values) * len(fixed_c2_values) + len(c1) * len(c2) * (len(cmode) - 1)) *
        num_repetitions
    )

    experiment_counter = 0

    # Fixed cmode (cmode = 0)
    for _iterations, _particles, fixed_c1, fixed_c2 in product(iterations, particles, fixed_c1_values, fixed_c2_values):
        for rep in range(num_repetitions):
            experiment_counter += 1
            progress = f"exp_{experiment_counter}"
            param_list.append((
                progress, _iterations, _particles, 0,  # cmode = 0
                fixed_c1, fixed_c1,
                fixed_c2, fixed_c2,
                rep, num_repetitions
            ))

    # Varying cmodes (cmode = 1 or 2)
    for _iterations, _particles, (c1_start, c1_end), (c2_start, c2_end), _cmode in product(iterations, particles, c1, c2, [1, 2]):
        for rep in range(num_repetitions):
            experiment_counter += 1
            progress = f"exp_{experiment_counter}"
            param_list.append((
                progress, _iterations, _particles, _cmode,
                c1_start, c1_end,
                c2_start, c2_end,
                rep, num_repetitions
            ))

    return param_list

def run_experiment_wrapper(args):
    return run_experiment(*args)


def main():
    print("Preparing experiment configurations...")
    param_list = build_param_list()
    print(f"Launching {len(param_list)} experiments using {workers} parallel workers...")

    with Pool(processes=workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(run_experiment_wrapper, param_list),
            total=len(param_list),
            bar_format='\n{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'
        ):
            pass
        # pool.starmap(run_experiment, param_list)

    print_results_summary()


if __name__ == "__main__":
    main()
