import os
from typing import Dict, Tuple, Union


def readParamInputs(
    fileGeneral: str,
    fileMethod: str,
    fileRPMs: str
) -> Tuple[Dict[str, Union[int, float, str]], Dict[str, Union[int, float, str]], Dict[str, Union[int, float, str]]]:
    """
    Read parameter inputs from configuration files.
    
    This function reads three configuration files containing general parameters,
    method parameters, and rock physics model parameters. It converts numeric
    values to appropriate types (int or float).
    
    Args:
        fileGeneral: Path to the general parameters file
        fileMethod: Path to the method parameters file
        fileRPMs: Path to the rock physics model parameters file
        
    Returns:
        Tuple containing three dictionaries:
        - General parameters dictionary
        - Method parameters dictionary
        - Rock physics model parameters dictionary
    """
    # General parameters
    pgeneral: Dict[str, Union[int, float, str]] = {}
    with open(os.path.join("config", fileGeneral), "r") as fg:
        for line in fg:
            if len(line.strip()) == 0 or line.strip()[0] == '#':
                continue
                
            parts = line.split('=')
            key = parts[0].strip()
            value = parts[1].strip()
            
            # Convert to appropriate numeric type
            if value.isnumeric() or (value[1:].isnumeric() and value[0] in ['+', '-']):
                value = int(value)
            else:
                x = value.split('.')
                if (len(x) == 2 and x[1].isnumeric() and
                        (x[0].isnumeric() or (x[0][1:].isnumeric() and x[0][0] in ['+', '-']))):
                    value = float(value)
                    
            pgeneral[key] = value
        
    # Method parameters    
    pmethod: Dict[str, Union[int, float, str]] = {}
    with open(os.path.join("config", fileMethod), "r") as fm:
        for line in fm:
            if len(line.strip()) == 0 or line.strip()[0] == '#':
                continue
                
            parts = line.split('=')
            key = parts[0].strip()
            value = parts[1].strip()
            
            # Convert to appropriate numeric type
            if value.isnumeric() or (value[1:].isnumeric() and value[0] in ['+', '-']):
                value = int(value)
            else:
                x = value.split('.')
                if (len(x) == 2 and x[1].isnumeric() and
                        (x[0].isnumeric() or (x[0][1:].isnumeric() and x[0][0] in ['+', '-']))):
                    value = float(value)
                    
            pmethod[key] = value
    
    # Rock physics model parameters    
    prpms: Dict[str, Union[int, float, str]] = {}
    with open(os.path.join("config", fileRPMs), "r") as fr:
        for line in fr:
            if len(line.strip()) == 0 or line.strip()[0] == '#':
                continue
                
            parts = line.split('=')
            key = parts[0].strip()
            value = parts[1].strip()
            
            # Convert to appropriate numeric type
            if value.isnumeric() or (value[1:].isnumeric() and value[0] in ['+', '-']):
                value = int(value)
            else:
                x = value.split('.')
                if (len(x) == 2 and x[1].isnumeric() and
                        (x[0].isnumeric() or (x[0][1:].isnumeric() and x[0][0] in ['+', '-']))):
                    value = float(value)
                    
            prpms[key] = value
    
    return pgeneral, pmethod, prpms
