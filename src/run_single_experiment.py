from src.sequential_experiments_driver import run_experiment


def main():
    """
    Run a single experiment.
    """
    iterations = 20  # 50
    particles = 10
    cmode = 0
    c1_start = 2.0
    c1_end = 0.5
    c2_start = 0.5
    c2_end = 2.0
    
    # Use the run_experiment function from experiments_driver.py
    run_experiment(
        "Single Experiment",
        iterations,
        particles,
        cmode,
        c1_start,
        c1_end,
        c2_start,
        c2_end,
        0,  # First (and only) repetition
        1   # Total repetitions
    )
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main() 