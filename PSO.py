import numpy as np
import math

class Particle(object):
    def __init__(self, size):
        self.size = size
        #particle data
        self.x = np.zeros(size)
        #velocity
        self.v = np.zeros(size)
        #objective value
        self.value = None
        #best in trajectory
        self.best_x = self.x
        self.best_value = math.inf
        
    def update(self, new_x, new_value): #minimization
        self.x = new_x
        self.value = new_value
        if new_value < self.best_value:
            self.best_x = new_x
            self.best_value = new_value
        
        

class PSO(object):
    def __init__(self, data):
        self.data = data.copy()
	
    #calculate the fitness of particle xi
    def objectiveValue(self, xi):
        return 0
    
    #regularize particle
    def regularize(self, xi):
        return xi
    
    #create a new particle by using a reference of 'guide' particle
    def newParticle(self, size, guide=None):
        rng = np.random.default_rng()
        #create a random particle
        xi = self.ParticleType(size)
        #xi.x = xi.x + np.random.randn(size) + guide
        xi.x = 0.2 + 0.3 * rng.standard_normal(size)
        xi.x = self.regularize(xi.x)
        xi.update(xi.x, self.objectiveValue(xi))
        return xi
    
    #create a new swarm
    def newSwarm(self, number_of_particles, size, guide = None):
        swarm = []
        
        for i in range(number_of_particles):
            swarm.append(self.newParticle(size, guide))
        return swarm
        
    
    #execute PSO
    # w: coeficiente de inércia
    # c1: aceleração na direção da trajetória
    # c2: aceleração na direção do melhor do swarm
    def optimize(self, number_of_particles, size, duration, w, c1_start, c1_end, c2_start, c2_end, cmode, ParticleType = Particle, guide=None):
        #if guide is not define, create a simple reference
        if guide is None: guide = np.zeros(size)
        
        self.ParticleType = ParticleType
        
        #particle swarm 
        swarm = self.newSwarm(number_of_particles, size, guide)

        #selecting best of swarm
        print(swarm)
        g = min(swarm, key=lambda x: x.value)
        print("best: ", g.value)

        # precompute coefficients depending on cmode
        if cmode == 0:  # fixed
            c1s = [c1_start] * duration
            c2s = [c2_start] * duration
        elif cmode == 1:  # linear
            c1s = np.linspace(c1_start, c1_end, duration)
            c2s = np.linspace(c2_start, c2_end, duration)
        elif cmode == 2:  # exponential
            c1s = np.logspace(np.log10(c1_start), np.log10(c1_end), duration)
            c2s = np.logspace(np.log10(c2_start), np.log10(c2_end), duration)

        #for each time window, move particles in the space
        for t in range(duration):
            for xi in swarm:
                r1 = 1.0-np.random.rand()
                r2 = 1.0-np.random.rand()
                xi.v = w*xi.v + c1s[t]*r1*(xi.best_x-xi.x) + c2s[t]*r2*(g.x-xi.x)
                new_x = self.regularize(xi.v + xi.x)
                xi.x = new_x
                new_value = self.objectiveValue(xi)
                xi.update(new_x, new_value)
            #update best of swarm
            g = min(swarm + [g], key=lambda x: x.value)
        return g, swarm
