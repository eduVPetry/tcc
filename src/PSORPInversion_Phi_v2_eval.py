import copy
import math
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional, Union

import src.PSO as PSO


class ParticleModel(PSO.Particle):
    """
    Extended particle class for rock physics models.
    
    This class extends the base PSO.Particle class to include model-specific
    attributes and methods for rock physics inversion.
    """
    def __init__(self, size: int) -> None:
        """
        Initialize a particle with model-specific attributes.
        
        Args:
            size: Dimension of the particle's position vector
        """
        PSO.Particle.__init__(self, size)
        # Model types:
        # 1. Wyllie model (2.1)
        # 2. Reymer model (2.2)
        # 3. SoftSand model (2.33:2.37)
        # 4. StiffSand model (2.33:2.34;2.38:2.39)
        # 5. Spherical inclusion model
        # 6. Berryman inclusion model
        self.model = ""
        self.OV = None

    def getModelLabel(self) -> str:
        """
        Get the label for the current model.
        
        Returns:
            The label string for the current model
        """
        return self.model_labels[self.model]
        
    def getModelColor(self) -> str:
        """
        Get the color for the current model.
        
        Returns:
            The color string for the current model
        """
        return self.model_colors[self.model_labels[self.model]]


class PSORPInversion_Phi(PSO.PSO):
    """
    Particle Swarm Optimization for Rock Physics Inversion.
    
    This class extends the base PSO class to implement rock physics model inversion
    for porosity estimation.
    """
    def __init__(
        self, 
        data: Dict[str, Any], 
        fixed: Dict[str, Any], 
        typeModel: int, 
        rpms: Dict[str, List[str]], 
        interest_var: str, 
        confidence_var: str, 
        horizon: Tuple[float, float] = (0, math.inf)
    ) -> None:
        """
        Initialize the PSO Rock Physics Inversion.
        
        Args:
            data: Dictionary containing the data for inversion
            fixed: Dictionary containing fixed parameters
            typeModel: Model type to use (0 for random, specific value for fixed model)
            rpms: Dictionary of rock physics models
            interest_var: Variable of interest to estimate
            confidence_var: Variable to use for confidence calculation
            horizon: Tuple of (min, max) values for the horizon
        """
        self.rpms = rpms
        self.horizon = horizon
        self.typeModel = typeModel  # Define the model for each new solution. If model != 0, fix the model with the value given in typeModelIt
        self.last_model = 0
        
        PSO.PSO.__init__(self, data)
        
        # Fixed parameters dictionary
        self.fixed = copy.deepcopy(fixed)
        self.interest = interest_var  # Variable to be estimated
        self.confidence = confidence_var  # Variable returned by the models
        
    def newParticle(self, size: int, guide: Optional[np.ndarray] = None) -> ParticleModel:
        """
        Create a new particle with the specified model type.
        
        Args:
            size: Dimension of the particle's position vector
            guide: Optional guide particle to use for initialization
            
        Returns:
            A new ParticleModel instance
        """
        xi = PSO.PSO.newParticle(self, size, guide)
        xi.model = self.typeModel
        return xi
    
    def objectiveValue(self, xi: Union[ParticleModel, np.ndarray]) -> float:
        """
        Calculate the fitness of a particle or position vector.
        
        Args:
            xi: The particle or position vector to evaluate
            
        Returns:
            The objective value (squared error)
        """
        # If xi is a numpy array, treat it as a position vector
        if isinstance(xi, np.ndarray):
            OV = self.model(self.typeModel, self.interest, xi[0])
        else:
            # Calculate by model
            OV = None
            if not hasattr(xi, 'model') or xi.model == "":
                OV = 0
            else:
                OV = self.model(xi.model, self.interest, xi.x[0])
            
            # Store the VP obtained by particle xi
            xi.OV = OV
                
        # Calculate squared error
        value = np.sum((self.data[self.confidence] - OV)**2)
        
        return value
    
    def model(self, name: str, param_x: str, value_x: float) -> Optional[np.ndarray]:
        """
        Execute a rock physics model with the given parameters.
        
        This method dynamically executes a rock physics model defined in the rpms dictionary.
        It replaces variables in the model definition with their actual values.
        
        Args:
            name: Name of the model to execute
            param_x: Parameter name to replace with value_x
            value_x: Value to use for the parameter
            
        Returns:
            The result of the model calculation or None if an error occurs
        """
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
                # print("vs", vs)

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
                # print(variables)
                # print(f'variables[{var}]=', command)
                # print(command)
                variables[var] = eval(command)
                if var == self.confidence:
                    return variables[var]
            # print('\nvariables', variables)
            #
            # print('\nself.fixed', self.fixed)
            # print('\nself.data', self.data)
            # print('\n')
            #
            # print(variables["_return_"])
            ret = None
        except Exception as error:
            print(f"The following error was presented when trying to execute model {name}:")
            print("\t", error)
            print(f"\t last variable: {var}")
            print(f"\t last command: {command}")
        return ret
        
    def regularize(self, x: np.ndarray) -> np.ndarray:
        """
        Apply regularization to a particle's position.
        
        This method ensures that the particle position stays within valid bounds
        for the rock physics model.
        
        Args:
            x: The particle's position vector to regularize
            
        Returns:
            The regularized position vector
        """
        for i in range(len(x)):
            x[i] = 0.000001 if x[i] < 0 else x[i]
            x[i] = 0.5 if x[i] > 0.5 else x[i]
        return x
