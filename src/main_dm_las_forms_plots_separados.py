import lasio
import math
import matplotlib.pyplot as plt
import numpy as np
import os

import src.PSORPInversion_Phi_v2_eval as pspi
import src.utils as utils

# Set random seed for reproducibility
np.random.seed(555)

def run_experiment(experiment_id: str, well_name: str, facies: int) -> None:
    """
    Run a rock physics inversion experiment using PSO optimization.
    
    This function performs rock physics model inversion using Particle Swarm Optimization
    to estimate porosity from well log data. It processes LAS file data, applies various
    rock physics models, and generates visualization plots of the results.
    
    Args:
        experiment_id: Identifier for the experiment, used for output file organization
        well_name: Name of the well (LAS file) to use, without extension
        facies: Facies number to select for analysis
        
    Returns:
        float: The best error achieved by the swarm across all models
    """
    # Model configuration
    model_labels = {
        'Wyllie': 'Wyllie',
        'Raymer': 'Raymer',
        'SoftSand': 'Soft sand',
        'StiffSand': 'Stiff sand',
        'SphericalInclusion': 'Spherical inclusion',
        'BerrymanInclusion': 'Berryman inclusion'
    }
    
    model_colors = {
        'Wyllie': 'orange',
        'Raymer': 'gold',
        'Soft sand': 'magenta',
        'Stiff sand': 'lime',
        'Spherical inclusion': 'red',
        'Berryman inclusion': 'darkviolet'
    }
    
    # Read parameter files
    pgeneral, pmethod, prpms = utils.readParamInputs(
        "param_general.txt",
        "param_pso.txt",
        "param_rpms.txt"
    )

    # Define depth horizon
    horizon = (0, math.inf)
    
    # Import LAS file data
    las = lasio.read(f"data/{well_name}.las")
    #  print(las.curves)
    #  exit(0)

    # Extract well log data
    aDept = np.array(las["DEPT"])
    aPhi = np.array(las["POROSIDADE"])
    aVp = np.array(las["VP"])
    pK_OFICIAL = np.array(las["K_OFICIAL"])
    pMU_OFICIAL = np.array(las["MU_OFICIAL"])
    pRHO = np.array(las["RHO"])
    pKFLUIDFINAL_AGUA80_QUARTZO = np.array(las["KFLUIDFINAL_AGUA80_QUARTZO"])
    pMU_AGUA80_QUARTZO = np.array(las["MU_AGUA80_QUARTZO"])
    pRHO_AGUA80_QUARTZO = np.array(las["RHO_AGUA80_QUARTZO"])
    pPR_OFICIAL = np.array(las["PR_OFICIAL"])
    pFACIES = np.array(las["FACIES_PETROFISICA8"])

    # Remove zero porosities and NaN values
    posZeradas = []
    for i in range(len(aPhi)):
        if (aPhi[i] == 0.0 or 
            math.isnan(aDept[i]) or 
            math.isnan(aPhi[i]) or 
            math.isnan(aVp[i]) or 
            math.isnan(pK_OFICIAL[i]) or 
            math.isnan(pMU_OFICIAL[i]) or  
            math.isnan(pRHO[i]) or 
            math.isnan(pKFLUIDFINAL_AGUA80_QUARTZO[i]) or 
            math.isnan(pMU_AGUA80_QUARTZO[i]) or 
            math.isnan(pRHO_AGUA80_QUARTZO[i]) or 
            math.isnan(pPR_OFICIAL[i]) or 
            math.isnan(pFACIES[i])):
            posZeradas.append(i)
            
    # Remove invalid data points
    for i in range(len(posZeradas)-1, -1, -1):
        pos = posZeradas[i]
        aDept = np.delete(aDept, pos)
        aPhi = np.delete(aPhi, pos)
        aVp = np.delete(aVp, pos)
        pK_OFICIAL = np.delete(pK_OFICIAL, pos)
        pMU_OFICIAL = np.delete(pMU_OFICIAL, pos)
        pRHO = np.delete(pRHO, pos)
        pKFLUIDFINAL_AGUA80_QUARTZO = np.delete(pKFLUIDFINAL_AGUA80_QUARTZO, pos)
        pMU_AGUA80_QUARTZO = np.delete(pMU_AGUA80_QUARTZO, pos)
        pRHO_AGUA80_QUARTZO = np.delete(pRHO_AGUA80_QUARTZO, pos)
        pPR_OFICIAL = np.delete(pPR_OFICIAL, pos)
        pFACIES = np.delete(pFACIES, pos)
        
    # Filter by facies
    posFacies = []
    for i in range(len(aPhi)):
        if pFACIES[i] != facies:
            posFacies.append(i)
            
    for i in range(len(posFacies)-1, -1, -1):
        pos = posFacies[i]
        aDept = np.delete(aDept, pos)
        aPhi = np.delete(aPhi, pos)
        aVp = np.delete(aVp, pos)
        pK_OFICIAL = np.delete(pK_OFICIAL, pos)
        pMU_OFICIAL = np.delete(pMU_OFICIAL, pos)
        pRHO = np.delete(pRHO, pos)
        pKFLUIDFINAL_AGUA80_QUARTZO = np.delete(pKFLUIDFINAL_AGUA80_QUARTZO, pos)
        pMU_AGUA80_QUARTZO = np.delete(pMU_AGUA80_QUARTZO, pos)
        pRHO_AGUA80_QUARTZO = np.delete(pRHO_AGUA80_QUARTZO, pos)
        pPR_OFICIAL = np.delete(pPR_OFICIAL, pos)
        pFACIES = np.delete(pFACIES, pos)
        
    # Prepare raw data and define non-null horizons
    horizon_top = 0
    horizon_bottom = None
    raw_data = []
    for i in range(len(aDept)):
        if i > 0:
            if horizon_bottom is None and (np.isnan(aPhi[i-1]) or np.isnan(aVp[i-1])):
                horizon_top = i
        if np.isnan(aPhi[i]) == False and np.isnan(aVp[i]) == False:
            horizon_bottom = i
        raw_data.append([i, aDept[i], aPhi[i], aVp[i]])
    raw_data = np.array(raw_data)


    horizon = (aDept[horizon_top], aDept[horizon_bottom])
    print("HORIZONTE:", horizon)
    print(f"TOPO-1 :: DEPT: {aDept[horizon_top-1]},Phi: {aPhi[horizon_top-1]}, Vp: {aVp[horizon_top-1]}" )
    print(f"TOPO :: DEPT: {aDept[horizon_top]},Phi: {aPhi[horizon_top]}, Vp: {aVp[horizon_top]}" )
    print(f"base :: DEPT: {aDept[horizon_bottom]},Phi: {aPhi[horizon_bottom]}, Vp: {aVp[horizon_bottom]}" )
    #print(f"base+1 :: DEPT: {aDept[horizon_bottom+1]},Phi: {aPhi[horizon_bottom+1]}, Vp: {aVp[horizon_bottom+1]}" )


    #importando do Stanford do Grana
    #datafile= "data/data1_softsand.dat"
    #datafile= "data/data4.dat"
    #brute = np.loadtxt(datafile)

    # Slice data considering the depth horizon
    start = 0
    start_found = False
    end = len(raw_data)
    end_found = False
    for i in range(len(raw_data)):
        if not start_found and raw_data[i,1] >= horizon[0]:
            start = i
            start_found = True
        if not end_found and raw_data[i,1] >= horizon[1]:
            end = i
            end_found = True
    print(f"Data range: {start} to {end}")
    raw_data = raw_data[start:end,:]

    # Prepare data dictionary
    data = {
        "_Depth_": raw_data[:,1].reshape(-1, 1),
        "_Phi_": raw_data[:,2].reshape(-1, 1),
        "_Vp_": raw_data[:,3].reshape(-1, 1)
    }

    # Set fixed parameters
    fixed = {
        "_Kmat_": np.nanmean(pK_OFICIAL),
        "_Gmat_": np.nanmean(pMU_OFICIAL),
        "_RHOmat_": np.nanmean(pRHO),
        "_Kfl_": np.nanmean(pKFLUIDFINAL_AGUA80_QUARTZO),
        "_Gfl_": np.nanmean(pMU_AGUA80_QUARTZO),
        "_RHOfl_": np.nanmean(pRHO_AGUA80_QUARTZO),
        "_coordnum_": 7,
        "_criticalPhi_": np.nanmax(aPhi),
        "_pressure_": np.nanmean(pPR_OFICIAL),
        "_Ar_": 0.2  # for elliptical inclusion model
    }

    #GRANA - STANFORD: importing data 1
    #data["Depth"]  = brute[:,0].reshape(-1, 1)
    #data["Depth"]  = data["Depth"].reshape(len(data["Depth"]))
    #data["Phi"]    = brute[:,1].reshape(-1, 1)
    #data["Vp"]     = brute[:,2].reshape(-1, 1)
    #data["Vp"]     = data["Vp"].reshape(len(data["Vp"]))

    #GRANA - STANFORD: importing data 4
    #data["Clay"]   = brute[:,0].reshape(-1, 1)
    #data["Depth"]  = brute[:,1].reshape(-1, 1)
    #data["Depth"]  = data["Depth"].reshape(len(data["Depth"]))
    #data["Facies"] = brute[:,2].reshape(-1, 1)
    #data["Phi"]    = brute[:,3].reshape(-1, 1)
    #data["Rho"]    = brute[:,4].reshape(-1, 1)
    #data["Rhorpm"] = brute[:,5].reshape(-1, 1)
    #data["Sw"]     = brute[:,6].reshape(-1, 1)
    #data["Vp"]     = brute[:,7].reshape(-1, 1)
    #data["Vp"]     = data["Vp"].reshape(len(data["Vp"]))
    #data["Vprpm"]  = brute[:,8].reshape(-1, 1)
    #data["Vs"]     = brute[:,9].reshape(-1, 1)
    #data["Vsrpm"]  = brute[:,10].reshape(-1, 1)
    #data["Facies"] = data["Facies"]-1
    #data["Facies"] = data["Facies"].astype(int)
          
    #Set fixed parameters
    #fixed = dict()
    #fixed["Kmat"]=36
    #fixed["Gmat"]=45
    #fixed["RHOmat"] = 2.65
    #fixed["Kfl"]=2.25
    #fixed["Gfl"]=0
    #fixed["RHOfl"] = 1
    #fixed["coordnum"] = 7
    #fixed["criticalPhi"] = 0.4
    #fixed["pressure"] = 0.02
    #fixed["Ar"] = 0.2  # for elliptical inclusion model


    pso = pspi.PSORPInversion_Phi(data, fixed, 0, prpms, pgeneral["interest"], pgeneral["confidence"])#, horizon)

    particles = pmethod["particles"]
    iterations = pmethod["iterations"]
    w = pmethod["w"]
    c1_start = pmethod["c1_start"]
    c1_end = pmethod["c1_end"]
    c2_start = pmethod["c2_start"]
    c2_end = pmethod["c2_end"]
    cmode = pmethod["cmode"]

    # Extract data for optimization
    data = pso.data["_Vp_"].copy()
    depth = pso.data["_Depth_"].copy()
    phi = pso.data["_Phi_"].copy()

    # Replace NaN porosities with the average of adjacent values
    posNAN = []
    for i in range(len(phi)):
        if math.isnan(phi[i]):
            posNAN.append(i)
    for i in range(len(posNAN)-1, -1, -1):
        # Find values before and after
        pos = posNAN[i]
        antes = pos
        while math.isnan(phi[antes]):
            antes -= 1
        depois = pos
        while math.isnan(phi[depois]):
            depois += 1
        if depois > len(phi)-1:
            depois = antes
        value = 0
        if antes > 0:
            value = (phi[antes] + phi[depois])/2.0
        phi[i] = value

    # Initialize storage for swarm results
    individuals = (particles+1) * len(model_labels)
    SWARM = {ind: {"x":[], "VP":[], "value": [], "m":[]} for ind in range(individuals)}  # store the average values
    SWARM_best_phi = {"x":[], "VP":[], "M":[]}  # store the best values considering PHI error
    X = []  # best considering VP error: data PHI
    M = []  # best considering VP error: data model
    V = []  # best considering VP error: data VP

    # Run optimization for each data point
    for d in range(len(data)):
        best = None
        all_swarm = []
        pso.data["_Vp_"] = data[d:d+1]   
        
        # Calculate best PHI error
        temp_swarm_best_phi = {"error": math.inf, "x":None, "VP":None, "M":None}
        
        for m in model_labels:
            pso.typeModel = m  # Set the model to be used
            swarm = pso.optimize(particles, len(pso.data["_Vp_"]), iterations, w, c1_start, c1_end, c2_start, c2_end, cmode, ParticleType = pspi.ParticleModel, guide=None)
            all_swarm += swarm[1] + [swarm[0]]
            print(f"Best of swarm: {swarm[0].value}")
            if best is None or best.value > swarm[0].value:
                best = swarm[0] 
            
            # Calculate best PHI error
            for xi in swarm[1]:
                if temp_swarm_best_phi["error"] > np.square(xi.x[0] - phi[d]):
                    temp_swarm_best_phi["error"] = np.square(xi.x[0] - phi[d])
                    temp_swarm_best_phi["x"] = xi.x[0]
                    temp_swarm_best_phi["VP"] = xi.OV
                    temp_swarm_best_phi["M"] = xi.model
        
        # Sort individuals            
        all_swarm = sorted(all_swarm, key=lambda xi: xi.x[0])
        for ind in range(len(all_swarm)):
            SWARM[ind]["x"].append(all_swarm[ind].x[0]) 
            SWARM[ind]["VP"].append(all_swarm[ind].OV) 
            SWARM[ind]["value"].append(all_swarm[ind].value) 
            SWARM[ind]["m"].append(all_swarm[ind].model)
                
        # Best considering the expected phi
        SWARM_best_phi["x"].append(temp_swarm_best_phi["x"])
        SWARM_best_phi["VP"].append(temp_swarm_best_phi["VP"])
        SWARM_best_phi["M"].append(temp_swarm_best_phi["M"])
        
        # Best considering the best of Swarm        
        X.append(best.x[0])
        M.append(best.model)
        V.append(best.OV)

    # Calculate new objective function: sum-squared error + difference between adjacent porosities read
    wA = pgeneral["wA"]
    wB = pgeneral["wB"]
    fitness = {ind: float("inf") for ind in range(individuals)} 
    minima = {"m": [], "ind": -1, "value": math.inf}
    print(f"#individuals: {individuals}")
    
    for ind in range(individuals):
        # Squared error
        valueA = sum(SWARM[ind]["value"]) / len(SWARM[ind]["value"])
        # Depth difference
        valueB = sum([abs(SWARM[ind]["x"][i-1]-SWARM[ind]["x"][i]) for i in range(1, len(SWARM[ind]["x"]))]) / len(SWARM[ind]["x"])
        fitness[ind] = wA*valueA + wB*valueB
        
        if len([i for i in range(len(SWARM[ind]['value'])) if math.isnan(SWARM[ind]['value'][i])]) > 0:
            print(f"NAN-Value - ind:{ind}, i:{i}")
        if len([i for i in range(len(SWARM[ind]['x'])) if math.isnan(SWARM[ind]['x'][i])]) > 0:
            print(f"NAN-Phi - ind:{ind}, i:{i}")
            
        print(f"{ind}: {fitness[ind]} = valueA: {valueA} and valueB: {valueB}, sum(SWARM[ind]['value']): {sum(SWARM[ind]['value'])}")
        if minima["value"] > fitness[ind]: 
            minima["ind"] = ind
            minima["value"] = fitness[ind]


    # Create output folder for current experiment
    output_folder = os.path.join("results", experiment_id)
    os.makedirs(output_folder, exist_ok=True)

    # Plot results
    fig, ax = plt.subplots()
    plt.plot(phi, depth, color='k', linewidth=2.0, label='Expected')  # expected
    ind = minima["ind"]
    print(f"Data length: {len(data)}")
    
    for d in range(len(data)-1):
        m = SWARM[ind]["m"][d]
        plt.plot([SWARM[ind]["x"][d], SWARM[ind]["x"][d+1]], [depth[d], depth[d+1]], color='b', linewidth=2.0, label='Swarm-best', linestyle='dashed')  # best

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='upper right')
    
    # Title
    erro = "%.7f" % np.mean(np.square(SWARM[ind]["x"] - phi))
    title = 'Swarm-best Error: ' + str(erro)
    plt.title(title)
    
    # Labels and grid
    plt.xlabel('Porosity($\phi$)')
    plt.ylabel('Depth')
    plt.grid()
    plt.xlim([0, 0.6])
    plt.ylim([min(depth), max(depth)])

    # Save figure
    plt.savefig(os.path.join(output_folder, f'saida0_h{horizon[0]},{horizon[1]}.pdf'))
    plt.close(fig)

    # Plot all individuals
    fig, ax = plt.subplots()
    for ind in range(individuals): 
        for d in range(len(data)-1):
            m = SWARM[ind]["m"][d]
            plt.plot([SWARM[ind]["x"][d], SWARM[ind]["x"][d+1]], [pso.data["_Depth_"][d], pso.data["_Depth_"][d+1]], color=model_colors[model_labels[m]], linewidth=2.0, label=model_labels[m])   
    
    plt.plot(phi, depth, color='k', linewidth=2.0, label='Expected')  # expected 
    ind = minima["ind"]
    print(f"Data length: {len(data)}")
    
    for d in range(len(data)-1):
        m = SWARM[ind]["m"][d]
        plt.plot([SWARM[ind]["x"][d], SWARM[ind]["x"][d+1]], [depth[d], depth[d+1]], color='b', linewidth=2.0, label='Swarm-best', linestyle='dashed')  # best

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='upper right')
    
    # Title
    erro = "%.7f" % np.mean(np.square(SWARM[ind]["x"] - phi))
    title = 'Swarm-best Error: ' + str(erro)
    plt.title(title)
    
    # Labels and grid
    plt.xlabel('Porosity($\phi$)')
    plt.ylabel('Depth')
    plt.grid()
    plt.xlim([0, 0.6])
    plt.ylim([min(depth), max(depth)])

    # Save figure
    plt.savefig(os.path.join(output_folder, f'saida1_h{horizon[0]},{horizon[1]}.pdf'))
    plt.close(fig)

    # Plot VPs
    fig, ax = plt.subplots()
    for m in range(1, 6+1):
        for ind in range(particles+1):
            for d in range(len(data)-1):
                m = SWARM[ind]["m"][d]
                plt.plot([SWARM[ind]["VP"][d], SWARM[ind]["VP"][d+1]], [pso.data["_Depth_"][d], pso.data["_Depth_"][d+1]], color=model_colors[model_labels[m]], linewidth=2.0, label=model_labels[m])       
    
    ind = minima["ind"]
    plt.plot(data, depth, color='k', linewidth=2.0, label='Expected')   
    for d in range(len(data)-1):
        m = SWARM[ind]["m"][d]
        plt.plot([SWARM[ind]["VP"][d], SWARM[ind]["VP"][d+1]], [depth[d], depth[d+1]], color='b', linewidth=2.0, label='Swarm-best', linestyle='dashed')  # best

    plt.grid()
    plt.xlabel('$V_P$')
    plt.ylabel('Depth')
    plt.ylim([min(depth), max(depth)])

    # Save figure
    plt.savefig(os.path.join(output_folder, f'saida2_h{horizon[0]},{horizon[1]}.pdf'))
    plt.close(fig)

    # Plot model distribution
    fig, ax = plt.subplots()
    line_first = None
    ind = minima["ind"]
    for i in range(len(depth)):
        d = depth[i]
        if line_first is None: 
            line_first = d
        m = SWARM_best_phi["M"][i]
        m_sw = SWARM[ind]["m"][i]
        plt.plot([0, 1.0], [d, d], color=model_colors[model_labels[m_sw]], linewidth=2.0, label=model_labels[m_sw], linestyle='solid')  # best
    
    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='upper right')
    
    # Labels
    plt.xlabel('Rock Physics Model of Swarm-best')
    plt.ylabel('Depth')

    plt.xlim([0, 1])
    plt.ylim([min(depth), max(depth)])
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Save figure
    plt.savefig(os.path.join(output_folder, f'saida3_h{horizon[0]},{horizon[1]}.pdf'))
    plt.close(fig)

    # Plot porosity vs VP
    fig, ax = plt.subplots()
    plt.scatter(phi, data, c='k')  # expected 
    ind = minima["ind"]
    for d in range(len(data)-1):
        m = SWARM[ind]["m"][d]
        plt.scatter([SWARM[ind]["x"][d], SWARM[ind]["x"][d+1]], [SWARM[ind]["VP"][d], SWARM[ind]["VP"][d+1]], c='b')  # best

    plt.xlabel('Porosity($\phi$)')
    plt.ylabel('$V_P$')
    plt.grid()

    # Save figure
    plt.savefig(os.path.join(output_folder, f'saida4_h{horizon[0]},{horizon[1]}.pdf'))
    plt.close(fig)
    
    # Return the best error achieved
    return minima["value"]
