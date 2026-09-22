#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.special import softmax
import pandas as pd
import seaborn as sns
import pylab as pl
from scipy.optimize import minimize, differential_evolution
#import belief_mix_model as bmm
import belief_mix_model_with_spatial_mix_normalized as bmm_spat
import pdb
import sys

data_dir = "../Lucas_Cecile/"





processed_data = pd.read_csv(data_dir+"extracted_trajs.csv")

random_agent = pd.read_csv(data_dir+"random_agent.csv")

random_agent['Species'] = 'Random'




all_data = pd.concat([processed_data, random_agent])





fittings = pd.DataFrame()

for grp in all_data.groupby(['Subject','Phase','Species']):
    print(grp[0])
    data_trajs = []
    for grp2 in grp[1].groupby('Session'):
        temp = [(r,c) for r,c in zip(grp2[1].Rows,grp2[1].Cols)]
        data_trajs.append(temp)    

    results = {}
    ll = []
    #for kernel in spatial_kernels:
    #    print(kernel)
    results = bmm_spat.fit_diff_evol(data_trajs)
        #print(results[kernel])
    # Best kernel
    
    best_res = results
    print("best:",best_res)
    temp = pd.DataFrame()
    temp['Subject'] = [grp[0][0]]
    temp['Phase'] = [grp[0][1]]
    temp['Species'] = [grp[0][2]]
    #temp['spatial_bias'] = [spatial_kernels[idx]]
    temp['LL'] = [best_res.fun]
    temp['LL / nchoices'] = [best_res.fun/len(grp[1])]
    temp['Random NLL / nchoices'] = [np.log(25)]

    zRS, zCS, zSS = best_res.x[:3]
    #zK, zR, zM = best_res.x[3:6]

    wRS, wCS, wSS = bmm_spat.softmax_weights([zRS, zCS, zSS])
    wK, wR, wM = best_res.x[3:6] #bmm_spat.softmax_weights([zK, zR, zM])    
    
    
    temp['W(row_scan)'] = [wRS]
    temp['W(column_scan)'] = [wCS]
    temp['W(spiral_scan)'] = [wSS]


    temp['W(spatial_bias)'] = [wK]
    temp['W(reward)'] = [wR]
    temp['W(memory)'] = [wM]
    temp['memory_recall ('+r'$\lambda$'+')'] = [best_res.x[6]]
    temp['beta'] = [best_res.x[7]]

    
    fittings = pd.concat([fittings,temp])
    
    




# fittings.to_csv(data_dir+"fittings_with_beta_wo_effort_wo_random_wo_sigma.csv")
fittings.to_csv(data_dir+"fittings_with_spatial_mixture_bias_normalized.csv")


