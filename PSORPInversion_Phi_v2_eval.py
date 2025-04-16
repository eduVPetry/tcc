import numpy as np
from scipy.io import loadmat
from scipy.linalg import norm
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import re, copy


import PSO

class ParticleModel(PSO.Particle):
    def __init__(self, size): 
        PSO.Particle.__init__(self, size)
        # 1. Wyllie model (2.1)
        # 2. Reymer model (2.2)
        # 3. SoftSand model (2.33:2.37)
        # 4. StiffSand model (2.33:2.34;2.38:2.39)
        # 5. Spherical inclusion model (???)
        # 6. Berryman inclusion model (???)
        self.model = ""
        #self.model = np.random.randint(low=1, high=6+1) #se não for definida pelo PSO, então fica com um dos modelos aleatórios
        #self.model_labels = {1:'Wyllie', 2:'Reymer', 3:'Soft sand', 4:'Stiff sand', 5:'Spherical inclusion', 6:'Berryman inclusion'}
        #self.model_colors = {'Wyllie': 'orange', 'Reymer': 'gold', 'Soft sand':'magenta', 'Stiff sand':'lime' , 'Spherical inclusion':'red', 'Berryman inclusion':'darkviolet'}
    
    def getModelLabel(self):
        return self.model_labels[self.model]
        
    def getModelColor(self):
        return self.model_colors[self.model_labels[self.model]]


class PSORPInversion_Phi(PSO.PSO):
    def __init__(self, data, fixed, typeModel, rpms, interest_var, confidence_var, horizon = (0, math.inf)):
        self.rpms = rpms
        self.horizon = horizon
        self.typeModel = typeModel #define o modelo a cada nova solução criada. Se model != 0, fixa o modelo com o valor dado em typeModelIt
        self.last_model = 0
        
        
        
        PSO.PSO.__init__(self, data)
        #fixing parameters 
        self.fixed = copy.deepcopy(fixed) #fixed parameters dictionary
        self.interest = interest_var #variable to be estimated
        self.confidence = confidence_var #variable returned by the models
        #self.Kmat           = fixed["Kmat"]
        #self.Gmat           = fixed["Gmat"]
        #self.RHOmat         = fixed["RHOmat"]
        #self.Kfl            = fixed["Kfl"]
        #self.Gfl            = fixed["Gfl"]
        #self.RHOfl          = fixed["RHOfl"]
        #self.coordnum       = fixed["coordnum"]
        #self.criticalPhi    = fixed["criticalPhi"]
        #self.pressure       = fixed["pressure"]
        #self.Ar             = fixed["pressure"] # for elliptical inclusion model
        
        
    def newParticle(self, size, guide=None):
        #xi = PSO.PSO.newParticle(self, size, guide)
        #if self.typeModel == 0:
        #    self.last_model += 1
        #    if self.last_model == 7: self.last_model = 1
        #    xi.model = self.last_model
        #else:
        #    xi.model = self.typeModelIt
        xi = PSO.PSO.newParticle(self, size, guide)
        xi.model = self.typeModel
        return xi
    
    #calculate the fitness of x from a particle
    def objectiveValue(self, xi):
        #calculating by model:
        OV = None
        if xi.model == "":
            OV = 0
        else:
            OV = self.model(xi.model, self.interest, xi.x[0])
        
        #Storing the VP obtained by the particle xi
        xi.OV = OV
                
                
        #squared error
        value = np.sum((self.data[self.confidence] - OV)**2) #/ len(OV)
        #if math.isnan(value):
        #    print(f"{value} = np.sum(({self.data[self.confidence]} - {OV})**2)")
        #    exit(0)
            
        
        #return objective value
        return value
    
    #banco de funcoes
    #Wyllie:
    #    VPmat = np.sqrt((Kmat + 4.0 / 3.0 * Gmat) / RHOmat)
    #    VSmat = np.sqrt(Gmat / RHOmat)
    #    VPfl = np.sqrt((Kfl + 4.0 / 3.0 * Gfl) / RHOfl)
    #    VSfl = np.sqrt(Gfl /RHOfl)
    #    VP_W = 1 / ((1 - PHI) / VPmat + x / VPfl)
    #    return VP_W
    def model(self, name, param_x, value_x):
        ret = None
        s = None
        newV = None
        variables = None
        command = ""
        try:
            variables = dict()
            for instruction in self.rpms[name]:
                s = instruction.split('=')
                
                var = s[0].strip()
                command = s[1].strip()
                vs = set(re.findall("_[A-Za-z_0-9]*_", command))
                #print("vs", vs)
                for v in vs:
                    newV = None
                    if v == param_x:
                        newV = str(value_x)
                    elif v in variables:
                        newV = f"variables['{v}']"
                    elif v in self.data:
                        newV = f"self.data['{v}']"
                    elif v in self.fixed:
                        newV = f"self.fixed['{v}']"
                    if newV is not None:
                        command = command.replace(v, newV)
                #print(variables)
                #print(f'variables[{var}]=', command)
                #print(command)
                variables[var] = eval(command)
                if var == self.confidence: 
                    return variables[var]
            #print('\nvariables', variables)
            #
            #print('\nself.fixed', self.fixed)
            #print('\nself.data', self.data)
            #print('\n')
            #
            #
            #print(variables["_return_"])
            ret = None
        except Exception as error:
            print(f"The following error was presented when trying to execute model {name}:")
            print("\t",error)
            print(f"\t last variable: {var}")
            print(f"\t last command: {command}")
        return ret
        
        
    #def model_Wyllie(self, x):
    #    #velocity definitions of solid (2.12)
    #    VPmat = np.sqrt((self.Kmat + 4.0 / 3.0 * self.Gmat) / self.RHOmat)
    #    VSmat = np.sqrt(self.Gmat / self.RHOmat)
    #    #velocity definitions of fluid (2.12)
    #    VPfl = np.sqrt((self.Kfl + 4.0 / 3.0 * self.Gfl) / self.RHOfl)
    #    VSfl = np.sqrt(self.Gfl / self.RHOfl)
    #    #Wyllie model (2.1)
    #    VP_W = 1 / ((1 - x) / VPmat + x / VPfl)
    #    #result...
    #    return VP_W
    #
    #def model_Raymer(self, x):
    #    #velocity definitions of solid (2.12)
    #    VPmat = np.sqrt((self.Kmat + 4.0 / 3.0 * self.Gmat) / self.RHOmat)
    #    VSmat = np.sqrt(self.Gmat / self.RHOmat)
    #    #velocity definitions of fluid (2.12)
    #    VPfl = np.sqrt((self.Kfl + 4.0 / 3.0 * self.Gfl) / self.RHOfl)
    #    VSfl = np.sqrt(self.Gfl / self.RHOfl)
    #    #Wyllie model (2.1)
    #    VP_R = (1 - x) ** 2.0 * VPmat + x * VPfl
    #    #result...
    #    return VP_R
    #
    #def model_SoftSand(self, x):
    #    #density model
    #    Rho = (1.0 - x) * self.RHOmat + x * self.RHOfl
    #    # Hertz-Mindlin
    #    Poisson = (3.0 * self.Kmat - 2.0 * self.Gmat) / (6.0 * self.Kmat + 2.0 * self.Gmat)
    #    KHM = ((self.coordnum ** 2.0 * (1.0 - self.criticalPhi) ** 2.0 * self.Gmat **2.0 * self.pressure) / (18.0 * np.pi ** 2.0 * (1 - Poisson) **2)) **(1.0 / 3.0)
    #    GHM = (5.0 - 4.0 * Poisson) / (10.0 - 5.0 * Poisson) * ((3.0 * self.coordnum ** 2.0 * (1.0 - self.criticalPhi) ** 2.0 * self.Gmat **2.0 * self.pressure) / (2 * np.pi ** 2.0 * (1.0 - Poisson) **2.0)) **(1.0 / 3.0)
    #    # Modified Hashin-Shtrikmann lower bounds
    #    Kdry = 1. / ((x / self.criticalPhi) / (KHM + 4.0 / 3.0 * GHM) + (1.0 - x / self.criticalPhi) / (self.Kmat + 4.0 / 3.0 * GHM)) - 4.0 / 3.0 * GHM
    #    psi = (9.0 * KHM + 8.0 * GHM) / (KHM + 2.0 * GHM)
    #    Gdry = 1.0 / ((x / self.criticalPhi) / (GHM + 1.0 / 6.0 * psi * GHM) + (1 - x / self.criticalPhi) / (self.Gmat + 1.0 / 6.0 * psi * GHM)) - 1.0 / 6.0 * psi * GHM
    #    # Gassmann
    #    # Bulk modulus of saturated rock
    #    Ksat = Kdry + ((1 - Kdry / self.Kmat) ** 2) / (x / self.Kfl + (1 - x) / self.Kmat - Kdry / (self.Kmat ** 2))
    #    #minha alteração para evitar NaN //##
    #    Ksat = [0.0 if math.isnan(e) else e for e in Ksat]
    #    # Shear modulus of saturated rock
    #    Gsat = Gdry
    #    # Velocities
    #    Vp_soft = np.sqrt((Ksat + 4.0 / 3.0 * Gsat) / Rho)
    #    #Vs = np.sqrt(Gsat / Rho)        
    #    return Vp_soft
    #        
    #
    #def model_StiffSand(self, x):
    #    #density model
    #    Rho = (1.0 - x) * self.RHOmat + x * self.RHOfl
    #    # Hertz-Mindlin
    #    Poisson = (3 * self.Kmat - 2 * self.Gmat) / (6 * self.Kmat + 2 * self.Gmat)
    #    KHM = ((self.coordnum ** 2 * (1 - self.criticalPhi) ** 2 * self.Gmat ** 2 * self.pressure) / (18 * np.pi ** 2 * (1 - Poisson) ** 2)) ** (1 / 3)
    #    GHM = (5 - 4 * Poisson) / (10 - 5 * Poisson) * ((3 * self.coordnum ** 2 * (1 - self.criticalPhi) ** 2 * self.Gmat ** 2 * self.pressure) / (2 * np.pi ** 2 * (1 - Poisson) ** 2)) ** (1 / 3)
    #
    #    # Modified Hashin-Shtrikmann upper bounds
    #    Kdry = 1. / ((x / self.criticalPhi) / (KHM + 4 / 3 * self.Gmat) + (1 - x / self.criticalPhi) / (self.Kmat + 4 / 3 * self.Gmat)) - 4 / 3 * self.Gmat
    #    psi = (9 * self.Kmat + 8 * self.Gmat) / (self.Kmat + 2 * self.Gmat)
    #    Gdry = 1. / ((x / self.criticalPhi) / (GHM + 1 / 6 * psi * self.Gmat) + (1 - x / self.criticalPhi) / (self.Gmat + 1 / 6 * psi * self.Gmat)) - 1 / 6 * psi * self.Gmat
    #
    #    # Gassmann
    #    # Bulk modulus of saturated rock
    #    Ksat = Kdry + ((1 - Kdry / self.Kmat) ** 2) / (x / self.Kfl + (1 - x) / self.Kmat - Kdry / (self.Kmat ** 2))
    #    
    #    Ksat = [0.0 if math.isnan(e) else e for e in Ksat] #minha alteração para evitar NaN //##
    #    # Shear modulus of saturated rock
    #    Gsat = Gdry
    #
    #    # Velocities
    #    Vp__stiff = np.sqrt((Ksat + 4 / 3 * Gsat) / Rho)
    #    Vp__stiff = [0.0 if math.isnan(e) else e for e in Vp__stiff]#minha alteração para evitar NaN //##
    #    #Vs = np.sqrt(Gsat / Rho)
    #    return Vp__stiff  
    #    
    ##spherical inclusion
    #def model_SphericalInclusion(self, x):
    #    #density model
    #    Rho = (1.0 - x) * self.RHOmat + x * self.RHOfl
    #
    #    # elastic moduli of the dry rock
    #    Kdry = 4 * self.Kmat * self.Gmat * (1 - x) / (3 * self.Kmat * x + 4 * self.Gmat)
    #    Gdry = self.Gmat * (9 * self.Kmat + 8 * self.Gmat) * (1 - x) / ((9 * self.Kmat + 8 * self.Gmat + 6 * (self.Kmat + 2 * self.Gmat) * x))
    #
    #    # Gassmann
    #    # Bulk modulus of saturated rock
    #    Ksat = Kdry + ((1 - Kdry / self.Kmat) ** 2) / (x / self.Kfl + (1 - x) / self.Kmat - Kdry / (self.Kmat ** 2))
    #    Ksat = [0.0 if math.isnan(e) else e for e in Ksat] #minha alteração para evitar NaN //##
    #    # Shear modulus of saturated rock
    #    Gsat = Gdry
    #
    #    # Velocities
    #    Vp_si = np.sqrt((Ksat + 4 / 3 * Gsat) / Rho)
    #    Vp_si = [0.0 if math.isnan(e) else e for e in Vp_si]#minha alteração para evitar NaN //##
    #    #Vs = np.sqrt(Gsat / Rho)        
    #    return Vp_si
    #
    #
    ##spherical inclusion
    #def model_BerrymanInclusion(self, x):
    #    #density model
    #    Rho = (1.0 - x) * self.RHOmat + x * self.RHOfl
    #    
    #    # inclusion properties 
    #    Kinc = self.Kfl
    #    Ginc = 0
    #
    #    # Berryman's formulation
    #    Poisson = (3 * self.Kmat - 2 * self.Gmat) / (2 * (3 * self.Kmat + self.Gmat))
    #    theta = self.Ar / (1 - self.Ar ** 2) ** (3/ 2) * (np.arccos(self.Ar) - self.Ar * np.sqrt(1 - self.Ar ** 2))
    #    g = self.Ar ** 2 / (1 - self.Ar ** 2) * (3* theta - 2)
    #    R = (1 - 2* Poisson) / (2 - 2* Poisson)
    #    A = (Ginc / self.Gmat) - 1
    #    B = 1/ 3* (Kinc / self.Kmat - Ginc / self.Gmat)
    #    F1 = 1 + A * (3/ 2* (g + theta) - R * (3/ 2* g + 5/ 2* theta - 4/ 3))
    #    F2 = 1 + A * (1 + 3/ 2* (g + theta) - R / 2* (3* g + 5 * theta)) + B * (3 - 4* R) + A / 2* (A + 3* B) * (3 - 4* R) * (g + theta - R * (g - theta + 2* theta ** 2))
    #    F3 = 1 + A * (1 - (g + 3 / 2 * theta) + R * (g + theta))
    #    F4 = 1 + A / 4* (g + 3* theta - R * (g - theta))
    #    F5 = A * (R * (g + theta - 4/ 3) - g) + B * theta * (3 - 4 * R)
    #    F6 = 1 + A * (1 + g - R * (theta + g)) + B * (1 - theta) * (3 - 4 * R)
    #    F7 = 2 + A / 4 * (9* theta + 3* g - R * (5* theta + 3* g)) + B * theta * (3 - 4* R)
    #    F8 = A * (1 - 2* R + g / 2* (R - 1) + theta / 2* (5* R - 3)) + B * (1 - theta) * (3 - 4* R)
    #    F9 = A * (g * (R - 1) - R * theta) + B * theta * (3 - 4* R)
    #    Tiijj = 3 * F1 / F2
    #    Tijij = Tiijj / 3 + 2/ F3 + 1/ F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
    #    P = Tiijj / 3
    #    Q = (Tijij - P) / 5
    #
    #    # elastic moduli
    #    Ksat = ((x * (Kinc - self.Kmat) * P) * 4 / 3* self.Gmat + self.Kmat * (self.Kmat + 4 / 3* self.Gmat)) / (self.Kmat + 4 / 3* self.Gmat - (x * (Kinc - self.Kmat) * P))
    #    psi = (self.Gmat * (9 * self.Kmat + 8* self.Gmat)) / (6 * (self.Kmat + 2 * self.Gmat))
    #    Gsat = (psi * (x * (Ginc - self.Gmat) * Q) + self.Gmat * (self.Gmat + psi)) / (self.Gmat + psi - (x * (Ginc - self.Gmat) * Q))
    #
    #    # velocities
    #    Vp_bi = np.sqrt((Ksat + 4 / 3 * Gsat) / Rho)
    #    Vp_bi = [0.0 if math.isnan(e) else e for e in Vp_bi]#minha alteração para evitar NaN //##
    #    #Vs = np.sqrt(Gsat / Rho)
    #    return Vp_bi

    
    #regularize particle x
    def regularize(self, x):
        #lowpass filter
        #cutoff = 3
        #fs = 7
        #order = len(x)
        #b, a =  butter(1, cutoff, fs=fs, btype='low', analog=False)
        #y = lfilter(b, a, x)
        
        #moving averages
        #w = 4
        #x = np.array([np.mean(x[i:i+w]) for i in range(len(x)) ])
        
        #Hodrick-Prescott filter
        #cycle, trend = sm.tsa.filters.hpfilter(x, 0.01)
        #x = np.array(trend)
        
        #outra regularização
        for i in range(len(x)):
            x[i] = 0.000001 if x[i] < 0 else x[i]
            x[i] = 0.5 if x[i] > 0.5 else x[i]
        return x


