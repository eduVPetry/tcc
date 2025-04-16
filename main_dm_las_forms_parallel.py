import numpy as np

np.random.seed(555)

from scipy.io import loadmat
from scipy.linalg import norm
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import math

import statsmodels.api as sm

import utils

import multiprocessing as mp


import PSORPInversion_Phi_v2_eval as pspi 
import lasio.examples

##################################################
# Funcao para executar os modelos paralelamente #
##################################################

def executar_modelo(m, pso, particles, phi, d, iterations, w, c1, c2):
    pso.typeModel = m #fixa o modelo a ser utilizado
    swarm = pso.optimize(particles, len(pso.data["_Vp_"]), iterations, w, c1, c2, \
                        ParticleType = pspi.ParticleModel, guide=None)
    
    temp_swarm_best_phi = {"error": math.inf, "x":None, "VP":None, "M":None}
    best = swarm[0]

    print(f"Melhor do enxame: {swarm[0].value}")

    #calculating best PHI error
    for xi in swarm[1]:
        if  temp_swarm_best_phi["error"] > np.square(xi.x[0] - phi[d]):
            temp_swarm_best_phi["error"] = np.square(xi.x[0] - phi[d])
            temp_swarm_best_phi["x"]  	 = xi.x[0]
            temp_swarm_best_phi["VP"] 	 = xi.OV
            temp_swarm_best_phi["M"]     = xi.model
    
    return swarm, best, temp_swarm_best_phi

###############################
##          MAIN             ##
###############################

if __name__ == '__main__':
 
    #semente fixada
    model_labels = {'Wyllie':'Wyllie', 'Raymer':'Raymer', 'SoftSand':'Soft sand', 'StiffSand':'Stiff sand', 'SphericalInclusion':'Spherical inclusion', 'BerrymanInclusion':'Berryman inclusion'}
    model_colors = {'Wyllie': 'orange', 'Raymer': 'gold', 'Soft sand':'magenta', 'Stiff sand':'lime' , 'Spherical inclusion':'red', 'Berryman inclusion':'darkviolet'}
        
    pgeneral, pmethod, prpms = utils.readParamInputs("param_general.txt","param_pso.txt","param_rpms.txt")


    #horizonte
    horizon = (0, math.inf)
    #horizon = (2750, 2800)

    #importando de arquivo .las
    #las: https://lasio.readthedocs.io/en/latest/basic-example.html
    las = lasio.read("data/RO_31A.las")
    #las = lasio.read("data/RO_033.las")
    #las = lasio.read("data/RO_110D.las")
    #las = lasio.read("data/RO_020.las")
    #print(las.curves)
    #exit(0)

    #detect how many processes will run at the same time
    try:
        workers = mp.cpu_count()
        # workers = int(argv[1])
    except NotImplementedError:
        workers = 1

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


    #removendo porosidades zeradas
    posZeradas = []
    for i in range(len(aPhi)):
        if aPhi[i] == 0.0 or math.isnan(aDept[i]) or math.isnan(aPhi[i]) or math.isnan(aVp[i]) or math.isnan(pK_OFICIAL[i]) or math.isnan(pMU_OFICIAL[i]) or  math.isnan(pRHO[i]) or math.isnan(pKFLUIDFINAL_AGUA80_QUARTZO[i]) or math.isnan(pMU_AGUA80_QUARTZO[i]) or math.isnan(pRHO_AGUA80_QUARTZO[i]) or math.isnan(pPR_OFICIAL[i]) or math.isnan(pFACIES[i]):
            posZeradas.append(i)
            
    for i in range(len(posZeradas)-1, -1, -1):
        pos = posZeradas[i]
        aDept = np.delete(aDept, pos)
        aPhi  = np.delete(aPhi, pos)
        aVp   = np.delete(aVp, pos)
        pK_OFICIAL = np.delete(pK_OFICIAL, pos)
        pMU_OFICIAL = np.delete(pMU_OFICIAL, pos)
        pRHO = np.delete(pRHO, pos)
        pKFLUIDFINAL_AGUA80_QUARTZO = np.delete(pKFLUIDFINAL_AGUA80_QUARTZO, pos)
        pMU_AGUA80_QUARTZO = np.delete(pMU_AGUA80_QUARTZO, pos)
        pRHO_AGUA80_QUARTZO = np.delete(pRHO_AGUA80_QUARTZO, pos)
        pPR_OFICIAL = np.delete(pPR_OFICIAL, pos)
        pFACIES = np.delete(pFACIES, pos)
        
    #removendo outras facies que não as que gostaríamos de tentar
    SELECIONADA = 7
    posFacies = []
    for i in range(len(aPhi)):
        if pFACIES[i] != SELECIONADA:
            posFacies.append(i)
    for i in range(len(posFacies)-1, -1, -1):
        pos = posFacies[i]
        aDept = np.delete(aDept, pos)
        aPhi  = np.delete(aPhi, pos)
        aVp   = np.delete(aVp, pos)
        pK_OFICIAL = np.delete(pK_OFICIAL, pos)
        pMU_OFICIAL = np.delete(pMU_OFICIAL, pos)
        pRHO = np.delete(pRHO, pos)
        pKFLUIDFINAL_AGUA80_QUARTZO = np.delete(pKFLUIDFINAL_AGUA80_QUARTZO, pos)
        pMU_AGUA80_QUARTZO = np.delete(pMU_AGUA80_QUARTZO, pos)
        pRHO_AGUA80_QUARTZO = np.delete(pRHO_AGUA80_QUARTZO, pos)
        pPR_OFICIAL = np.delete(pPR_OFICIAL, pos)
        pFACIES = np.delete(pFACIES, pos)
        
        

    #preparando dados brutos e define horizontes não nulos
    horizon_top = 0
    horizon_bottom = None
    brute = []
    for i in range(len(aDept)):
        if i > 0:
            if horizon_bottom is None and (np.isnan(aPhi[i-1]) or np.isnan(aVp[i-1])):
                horizon_top = i
        if np.isnan(aPhi[i]) == False and np.isnan(aVp[i]) == False:
            horizon_bottom = i
        brute.append([i, aDept[i], aPhi[i], aVp[i]])
    brute = np.array(brute)



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

    #slicing data considering the depth horizon

    start = 0
    st_bool = False
    end = len(brute)
    end_bool = False
    for i in range(len(brute)):
        if st_bool == False and brute[i,1] >= horizon[0]:
            start = i
            st_bool = True
        if end_bool == False and brute[i,1] >= horizon[1]:
            end = i
            end_bool = True
    print(start, end)
    brute = brute[start:end,:]

    data = dict()
    data["_Depth_"]  = brute[:,1].reshape(-1, 1)
    data["_Phi_"]    = brute[:,2].reshape(-1, 1)
    data["_Vp_"]     = brute[:,3].reshape(-1, 1)

    #fixing parameters 
    fixed = dict()
    fixed["_Kmat_"] = np.nanmean(pK_OFICIAL) #36
    fixed["_Gmat_"] = np.nanmean(pMU_OFICIAL)#45
    fixed["_RHOmat_"]      = np.nanmean(pRHO)#2.65
    fixed["_Kfl_"]         = np.nanmean(pKFLUIDFINAL_AGUA80_QUARTZO)#2.25
    fixed["_Gfl_"]         = np.nanmean(pMU_AGUA80_QUARTZO)
    fixed["_RHOfl_"]       = np.nanmean(pRHO_AGUA80_QUARTZO)
    fixed["_coordnum_"]    = 7
    fixed["_criticalPhi_"] = np.nanmax(aPhi)
    fixed["_pressure_"]    = np.nanmean(pPR_OFICIAL)
    fixed["_Ar_"] = 0.2 # for elliptical inclusion model




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


    #fixing parameters 
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
    #fixed["Ar"] = 0.2 # for elliptical inclusion model






    pso = pspi.PSORPInversion_Phi(data, fixed, 0, prpms, pgeneral["interest"], pgeneral["confidence"])#, horizon)


    particles = pmethod["particles"]
    iterations = pmethod["iterations"]
    w = pmethod["w"]
    c1 = pmethod["c1"]
    c2 = pmethod["c2"]

    data = pso.data["_Vp_"].copy()
    depth = pso.data["_Depth_"].copy()
    phi = pso.data["_Phi_"].copy()

    #substituindo porosidades nan pela média das duas porosidades adjacentes
    posNAN = []
    for i in range(len(phi)):
        if math.isnan(phi[i]):
            posNAN.append(i)
    for i in range(len(posNAN)-1, -1, -1):
        #procurar valor antes e depois
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
        phi[i]  = value
        






    individuals = (particles+1) * len(model_labels)

    SWARM = { ind: {"x":[], "VP":[], "value": [], "m":[]} for ind in range (individuals)} #store the average values
    SWARM_best_phi = {"x":[], "VP":[], "M":[]} #store the best values considering PHI error
    X = [] #best considering VP error: data PHI
    M = [] #best considering VP error: data model
    V = [] #best considering VP error: data VP


    #uma execução por modelo
    for d in range(len(data)):
        best = None
        all_swarm = []
        pso.data["_Vp_"] = data[d:d+1]   
        temp_swarm_best_phi = {"error": math.inf, "x":None, "VP":None, "M":None}
        
        with mp.Pool(processes=workers) as pool:
            async_results = [pool.apply_async(executar_modelo, args=(m, pso, particles, phi, d, iterations, w, c1, c2))
                            for m in model_labels]
            
            results = [res.get() for res in async_results]
        
        for swarm, model_best, model_temp_swarm_best_phi in results:
            all_swarm += swarm[1] + [swarm[0]]

            if best is None or best.value > model_best.value:
                best = model_best
            
            if temp_swarm_best_phi["error"] > model_temp_swarm_best_phi["error"]:
                temp_swarm_best_phi["error"] = model_temp_swarm_best_phi["error"]
                temp_swarm_best_phi["x"]     = model_temp_swarm_best_phi["x"]
                temp_swarm_best_phi["VP"]    = model_temp_swarm_best_phi["VP"]
                temp_swarm_best_phi["M"]     = model_temp_swarm_best_phi["M"]
                

        #organizando indivíduos            
        all_swarm = sorted(all_swarm, key= lambda xi: xi.x[0])
        for ind in range(len(all_swarm)):
            SWARM[ind]["x"].append(all_swarm[ind].x[0]) 
            SWARM[ind]["VP"].append(all_swarm[ind].OV) 
            SWARM[ind]["value"].append(all_swarm[ind].value) 
            SWARM[ind]["m"].append(all_swarm[ind].model)
                
        #best considering the expected phi
        SWARM_best_phi["x"].append(temp_swarm_best_phi["x"])
        SWARM_best_phi["VP"].append(temp_swarm_best_phi["VP"])
        SWARM_best_phi["M"].append(temp_swarm_best_phi["M"])
        
        #best considering the best of Swarm        
        X.append(best.x[0])
        M.append(best.model)
        V.append(best.OV)

    #calculate new objective function: sum-squared error + difference between adjacent porosities read.
    wA = pgeneral["wA"]
    wB = pgeneral["wB"]
    fitness = {ind: float("inf") for ind in range (individuals)} 
    minima = {"m":[], "ind": -1 , "value": math.inf}
    print("#individuos: ", individuals)
    for ind in range(individuals):
        #squared error
        valueA = sum(SWARM[ind]["value"]) / len(SWARM[ind]["value"])
        #depth difference
        valueB = sum([abs(SWARM[ind]["x"][i-1]-SWARM[ind]["x"][i]) for i in range(1, len(SWARM[ind]["x"])) ]) / len(SWARM[ind]["x"])
        fitness[ind] = wA*valueA + wB*valueB
        if len([i for i in range(len(SWARM[ind]['value'])) if math.isnan(SWARM[ind]['value'][i])]) > 0:
            print(f"NAN-Value - ind:{ind}, i:{i}")
        if len([i for i in range(len(SWARM[ind]['x'])) if math.isnan(SWARM[ind]['x'][i])]) > 0:
            print(f"NAN-Phi - ind:{ind}, i:{i}")
        print(f"{ind}: {fitness[ind]} = valueA: {valueA} e valueB: {valueB}, sum(SWARM[ind]['value']): {sum(SWARM[ind]['value'])}")
        if minima["value"] > fitness[ind]: 
            minima["ind"] = ind
            minima["value"] = fitness[ind]


    ########################################
    # plotagem
    ########################################


    fig = plt.figure(1)
    plt.subplot(221)
    for ind in range(individuals):
        for d in range(len(data)-1):
            m = SWARM[ind]["m"][d]
            plt.plot([SWARM[ind]["x"][d], SWARM[ind]["x"][d+1]], [pso.data["_Depth_"][d], pso.data["_Depth_"][d+1]], color=model_colors[model_labels[m]], linewidth=2.0, label=model_labels[m])   
    plt.plot(phi, depth, color='k', linewidth=2.0, label = 'Esperado') #expected 
    ind = minima["ind"]
    print(len(data)) #3856
    for d in range(len(data)-1):
        #print(ind, d) #-1 0
        m = SWARM[ind]["m"][d]
        plt.plot([SWARM[ind]["x"][d], SWARM[ind]["x"][d+1]], [depth[d], depth[d+1]], color='b', linewidth=2.0, label = 'Swarm-best', linestyle='dashed')   #best


    #legenda
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='upper right')
    title = ''
    #erro = "%.5f" % np.mean(np.square(SWARM_best_phi["x"] - phi))
    #title = 'Erro Phi-best: '+str(erro)
    erro = "%.7f" % np.mean(np.square(SWARM[ind]["x"] - phi))
    title += 'Erro Swarm-best: '+str(erro)
    plt.title(title)
    plt.xlabel('Porosidade($\phi$)')
    plt.ylabel('Profundidade')
    plt.grid()
    plt.xlim([0, 0.6])
    plt.ylim([min(depth), max(depth)])




    #ploting VPs 
    plt.subplot(222) ## <--- aqui

    for m in range(1, 6+1):
        for ind in range(particles+1):
            for d in range(len(data)-1):
                m = SWARM[ind]["m"][d]
                plt.plot([SWARM[ind]["VP"][d], SWARM[ind]["VP"][d+1]], [pso.data["_Depth_"][d], pso.data["_Depth_"][d+1]], color=model_colors[model_labels[m]], linewidth=2.0, label=model_labels[m])       
    ind = minima["ind"]
    plt.plot(data, depth, color='k', linewidth=2.0, label = 'Esperado')   
    for d in range(len(data)-1):
        m = SWARM[ind]["m"][d]
        plt.plot([SWARM[ind]["VP"][d], SWARM[ind]["VP"][d+1]], [depth[d], depth[d+1]], color='b', linewidth=2.0, label = 'Swarm-best', linestyle='dashed')   #best
        #plt.plot([SWARM[ind]["VP"][d]], [depth[d]], color='b', linewidth=2.0, label = 'Swarm-best', linestyle='dashed')   #best


    #plt.legend()


    plt.grid()
    plt.xlabel('$V_P$')
    plt.ylabel('Profundidade')
    plt.ylim([min(depth), max(depth)])
    #plt.ylabel('Real')


    #plt.subplot(323)
    #plt.plot(phi, depth, color='k', linewidth=2.0, label = 'Esperado') #expected 
    ##plt.plot(SWARM_best_phi["x"], depth, color='r', linewidth=2.0, label = 'Phi-best', linestyle='dashed')   #best
    #m = minima["m"]
    #ind = minima["ind"]
    #for d in range(len(data)-1):
    #    m = SWARM[ind]["m"][d]
    #    plt.plot([SWARM[ind]["x"][d],SWARM[ind]["x"][d+1]] , [depth[d],depth[d+1]], color='b', linewidth=2.0, label = 'Swarm-best', linestyle='dashed')   #best
    #    
    #
    ##legenda
    #handles, labels = plt.gca().get_legend_handles_labels()
    #labels, ids = np.unique(labels, return_index=True)
    #handles = [handles[i] for i in ids]
    #plt.legend(handles, labels, loc='upper right')
    #plt.xlabel('Porosidade($\phi$)')
    #plt.ylabel('Profundidade')
    #plt.grid()
    #plt.xlim([0, 0.6])
    #plt.ylim([min(depth), max(depth)])




    plt.subplot(224)
    line_first = None
    line_last  = None
    ind = minima["ind"]
    for i in range(len(depth)):
        d = depth[i]
        if line_first is None: line_first = d
        line_last = d
        m = SWARM_best_phi["M"][i]
        m_sw = SWARM[ind]["m"][i]
        #plt.plot([0, 0.5], [d, d], color=model_colors[model_labels[m]], linewidth=2.0, label =model_labels[m],linestyle='solid')   #best
        #plt.plot([0.5, 1.0], [d, d], color=model_colors[model_labels[m_sw]], linewidth=2.0, label =model_labels[m_sw],linestyle='solid')   #best
        plt.plot([0, 1.0], [d, d], color=model_colors[model_labels[m_sw]], linewidth=2.0, label =model_labels[m_sw],linestyle='solid')   #best
    #plt.plot([0.5, 0.5], [line_first, line_last], color='k', linewidth=2.0,linestyle='solid')   #best
    #legenda
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='upper right')
    plt.xlabel('Modelo de Física de Rocha do Swarm-best')
    plt.ylabel('Profundidade')

    plt.xlim([0, 1])
    plt.ylim([min(depth), max(depth)])
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)






    #plt.subplot(325)
    #plt.plot(data, depth, color='k', linewidth=2.0, label = 'Esperado') #expected 
    ##plt.plot(SWARM_best_phi["VP"], depth, color='r', linewidth=2.0, label = 'Phi-best', linestyle='dashed')   #best
    #
    #ind = minima["ind"]
    #for d in range(len(data)-1):
    #    m = SWARM[ind]["m"][d]
    #    plt.plot([SWARM[ind]["VP"][d], SWARM[ind]["VP"][d+1]], [depth[d], depth[d+1]], color='b', linewidth=2.0, label = 'Swarm-best', linestyle='dashed')   #best
    ##legenda
    #handles, labels = plt.gca().get_legend_handles_labels()
    #labels, ids = np.unique(labels, return_index=True)
    #handles = [handles[i] for i in ids]
    #plt.legend(handles, labels, loc='upper right')
    #plt.xlabel('$V_P$')
    #plt.ylabel('Profundidade')
    #plt.grid()
    #plt.ylim([min(depth), max(depth)])
    #
    #
    ##plt.show()
    ##exit(0)

    plt.subplot(223)
    plt.scatter(phi, data, c='k') #expected 
    #plt.scatter(SWARM_best_phi["x"], SWARM_best_phi["VP"], c='r')   #best
    ind = minima["ind"]
    for d in range(len(data)-1):
        m = SWARM[ind]["m"][d]
        plt.scatter([SWARM[ind]["x"][d], SWARM[ind]["x"][d+1]], [SWARM[ind]["VP"][d], SWARM[ind]["VP"][d+1]], c='b')   #best

    plt.xlabel('Porosidade($\phi$)')
    plt.ylabel('$V_P$')
    plt.grid()

    plt.show()


    plt.savefig('saida_h'+str(pso.horizon[0])+','+str(pso.horizon[1])+'.pdf') 


    plt.show()
    exit(0)













