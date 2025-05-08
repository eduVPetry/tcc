import os


def write_param_pso(
    particles: int,
    iterations: int,
    cmode: int,
    c1_start: float,
    c1_end: float,
    c2_start: float,
    c2_end: float,
    experiment_id: str
) -> None:
    """
    Write PSO parameters to a unique configuration file.
    """
    os.makedirs("config", exist_ok=True)
    filename = os.path.join("config", f"param_pso_{experiment_id}.txt")

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
