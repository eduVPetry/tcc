import os


def write_param_pso(
    particles: int,
    iterations: int,
    cmode: int,
    c1_start: float,
    c1_end: float,
    c2_start: float,
    c2_end: float
) -> None:
    """
    Write PSO parameters to a configuration file.
    
    Args:
        particles: Number of particles in the swarm
        iterations: Number of iterations to run
        cmode: Coefficient update mode (0=fixed, 1=linear, 2=exponential)
        c1_start: Initial cognitive acceleration
        c1_end: Final cognitive acceleration
        c2_start: Initial social acceleration
        c2_end: Final social acceleration
    """
    filename = os.path.join("config", "param_pso.txt")
    config_str = (
        f"particles = {particles}\n"
        f"iterations = {iterations}\n"
        f"w = 0.8\n\n"
        f"# cmode: 0 = fixed, 1 = linear, 2 = exponential\n"
        f"cmode = {cmode}\n\n"
        f"# if cmode = 1 or cmode = 2: use c1_start > c1_end and c2_start < c2_end\n"
        f"c1_start = {c1_start}\n"
        f"c1_end = {c1_end}\n"
        f"c2_start = {c2_start}\n"
        f"c2_end = {c2_end}\n"
    )
    with open(filename, "w") as f:
        f.write(config_str)
