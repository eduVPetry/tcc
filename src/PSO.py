import numpy as np
from typing import Dict, List, Optional, Tuple, Type


class Particle:
    """
    A particle in the PSO algorithm.
    
    Each particle represents a candidate solution and maintains its current position,
    velocity, and best known position.
    """
    def __init__(self, size: int):
        """Initialize a particle with given size."""
        self.size = size
        self.x = np.zeros(size)  # position
        self.v = np.zeros(size)  # velocity
        self.value = None  # objective value
        self.best_x = self.x.copy()  # best position
        self.best_value = float('inf')
    
    def update(self, new_x: np.ndarray, new_value: float) -> None:
        """Update particle state if new position is better."""
        self.x = new_x
        self.value = new_value
        if new_value < self.best_value:
            self.best_x = new_x.copy()
            self.best_value = new_value


class PSO:
    """
    Particle Swarm Optimization (PSO) algorithm implementation.
    
    PSO is a population-based optimization technique inspired by social behavior
    of bird flocking or fish schooling.
    """
    
    def __init__(self, data: Dict):
        """Initialize PSO with problem-specific data."""
        self.data = data.copy()
        self.ParticleType: Type[Particle] = Particle
    
    def objectiveValue(self, xi: np.ndarray) -> float:
        """Calculate the fitness of particle xi."""
        return 0
    
    def regularize(self, xi: np.ndarray) -> np.ndarray:
        """Apply problem-specific constraints to position."""
        return xi
    
    def newParticle(self, size: int, guide: Optional[np.ndarray] = None) -> Particle:
        """Create a new particle, optionally using a guide particle."""
        rng = np.random.default_rng()
        xi = self.ParticleType(size)
        xi.x = 0.2 + 0.3 * rng.standard_normal(size)
        if guide is not None:
            xi.x = xi.x + guide
        xi.x = self.regularize(xi.x)
        xi.update(xi.x, self.objectiveValue(xi.x))
        return xi
    
    def newSwarm(self, number_of_particles: int, size: int,
                 guide: Optional[np.ndarray] = None) -> List[Particle]:
        """Create a new swarm of particles."""
        return [self.newParticle(size, guide) for _ in range(number_of_particles)]
    
    def optimize(self, number_of_particles: int, size: int, duration: int,
                w: float, c1_start: float, c1_end: float, c2_start: float,
                c2_end: float, cmode: int,
                ParticleType: Type[Particle] = Particle,
                guide: Optional[np.ndarray] = None) -> Tuple[Particle, List[Particle]]:
        """
        Execute PSO optimization.
        
        Args:
            number_of_particles: Number of particles in swarm
            size: Dimension of search space
            duration: Number of iterations
            w: Inertia coefficient
            c1_start: Initial cognitive acceleration
            c1_end: Final cognitive acceleration
            c2_start: Initial social acceleration
            c2_end: Final social acceleration
            cmode: Coefficient update mode (0=fixed, 1=linear, 2=exponential)
            ParticleType: Class to use for particles
            guide: Optional guide particle position
            
        Returns:
            Tuple of (best particle, final swarm)
        """
        if guide is None:
            guide = np.zeros(size)
        
        self.ParticleType = ParticleType
        swarm = self.newSwarm(number_of_particles, size, guide)
        
        # Select initial best particle
        g = min(swarm, key=lambda x: x.value)
        
        # Precompute coefficients based on mode
        if cmode == 0:  # fixed
            c1s = np.full(duration, c1_start)
            c2s = np.full(duration, c2_start)
        elif cmode == 1:  # linear
            c1s = np.linspace(c1_start, c1_end, duration)
            c2s = np.linspace(c2_start, c2_end, duration)
        elif cmode == 2:  # exponential
            c1s = np.logspace(np.log10(c1_start), np.log10(c1_end), duration)
            c2s = np.logspace(np.log10(c2_start), np.log10(c2_end), duration)
        else:
            raise ValueError(f"Invalid cmode: {cmode}")
        
        # Main optimization loop
        for t in range(duration):
            for xi in swarm:
                r1 = 1.0 - np.random.rand()
                r2 = 1.0 - np.random.rand()
                
                # Update velocity
                xi.v = (w * xi.v +
                       c1s[t] * r1 * (xi.best_x - xi.x) +
                       c2s[t] * r2 * (g.x - xi.x))
                
                # Update position
                new_x = self.regularize(xi.v + xi.x)
                new_value = self.objectiveValue(new_x)
                xi.update(new_x, new_value)
            
            # Update global best
            g = min(swarm + [g], key=lambda x: x.value)
        
        return g, swarm
