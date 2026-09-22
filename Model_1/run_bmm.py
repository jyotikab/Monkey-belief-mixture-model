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
import belief_mix_model as bmm
import pdb
import sys

data_dir = "../Lucas_Cecile/"





processed_data = pd.read_csv(data_dir+"extracted_trajs.csv")

random_agent = pd.read_csv(data_dir+"random_agent.csv")

random_agent['Species'] = 'Random'




all_data = pd.concat([processed_data, random_agent])



spatial_kernels = ["row_scan",
                   "column_scan",
                   "spiral",
#                    "random",
                   ]



fittings = pd.DataFrame()

for grp in all_data.groupby(['Subject','Phase','Species']):
    print(grp[0])
    data_trajs = []
    for grp2 in grp[1].groupby('Session'):
        temp = [(r,c) for r,c in zip(grp2[1].Rows,grp2[1].Cols)]
        data_trajs.append(temp)    

    results = {}
    ll = []
    for kernel in spatial_kernels:
        print(kernel)
        results[kernel] = bmm.fit_diff_evol(data_trajs, kernel)
        ll.append(results[kernel].fun)
        #print(results[kernel])
    # Best kernel
    idx = np.argmin(ll)
    
    best_res = results[spatial_kernels[idx]]
    print("best:",best_res)
    temp = pd.DataFrame()
    temp['Subject'] = [grp[0][0]]
    temp['Phase'] = [grp[0][1]]
    temp['Species'] = [grp[0][2]]
    temp['spatial_bias'] = [spatial_kernels[idx]]
    temp['LL'] = [best_res.fun]


    temp['W(spatial_bias)'] = [best_res.x[0]]
    temp['W(reward)'] = [best_res.x[1]]
    temp['W(memory)'] = [best_res.x[2]]
    temp['memory_recall ('+r'$\lambda$'+')'] = [best_res.x[3]]
    temp['beta'] = [best_res.x[4]]

    
    fittings = pd.concat([fittings,temp])
    
    




# fittings.to_csv(data_dir+"fittings_with_beta_wo_effort_wo_random_wo_sigma.csv")
fittings.to_csv(data_dir+"fittings_with_beta_wo_effort_wo_random_wo_sigma_K_curr_loc_with_random.csv")


