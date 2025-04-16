
import re

def readParamInputs(fileGeneral, fileMethod, fileRPMs):
    #general parameters
    pgeneral = dict()
    fg = open(fileGeneral, "r")
    for line in fg:
        if len(line.strip())==0: continue
        if line.strip()[0]== '#': continue
        parts = line.split('=')
        key = parts[0].strip()
        value = parts[1].strip()
        if value.isnumeric() or (value[1:].isnumeric() and value[0] in ['+','-']): value = int(value)
        else: 
            x = value.split('.')
            if len(x) == 2 and x[1].isnumeric() and (x[0].isnumeric() or (x[0][1:].isnumeric() and x[0][0] in ['+','-'])): value = float(value)        
        pgeneral[key] = value
        
    #method parameters    
    pmethod = dict()
    fg = open(fileMethod, "r")
    for line in fg:
        if len(line.strip())==0: continue
        if line.strip()[0]== '#': continue
        parts = line.split('=')
        key = parts[0].strip()
        value = parts[1].strip()
        if value.isnumeric() or (value[1:].isnumeric() and value[0] in ['+','-']): value = int(value)
        else: 
            x = value.split('.')
            if len(x) == 2 and x[1].isnumeric() and (x[0].isnumeric() or (x[0][1:].isnumeric() and x[0][0] in ['+','-'])): value = float(value)
        pmethod[key] = value
    
    #Rock Physics Models
    prpms = dict()
    fg = open(fileRPMs, "r")
    lastModel = None
    commands = []
    for line in fg:
        if len(line.strip())==0: continue
        if line.strip()[0]== '#': continue
        if line.find(':') > 0:
            if lastModel is not None:
                prpms[lastModel] = commands
                commands = []  
            lastModel = line.split(':')[0].strip()
            continue
        command = line.strip()
        commands.append(command)
    prpms[lastModel] = commands
    
    return pgeneral, pmethod, prpms
