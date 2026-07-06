#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.special import softmax
import pandas as pd
import seaborn as sns
import pylab as pl
from scipy.optimize import minimize, differential_evolution
import belief_mix_model as bmm


# In[2]:


data_dir = "../Lucas_Cecile/"


# In[3]:


processed_data = pd.read_csv(data_dir+"extracted_trajs.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


# eg_trajs = processed_data.loc[(processed_data['Subject']=='Amidala')&(processed_data['Phase']=='Habituation')].copy()
# eg_trajs


# In[5]:


spatial_kernels = ["row_scan",
                   "column_scan",
                   "spiral",
                   "random",
                   ]


# In[ ]:


fittings = pd.DataFrame()

for grp in processed_data.groupby(['Subject','Phase','Species']):
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
    #temp['W(effort)'] = [best_res.x[3]]

    temp['reward_influence'] = [best_res.x[3]]
    temp['memory_recall ('+r'$\lambda$'+')'] = [best_res.x[4]]
    temp['beta'] = [best_res.x[5]]
#     temp['inverse_temperature'] = [best_res.x[5]]
    
    fittings = pd.concat([fittings,temp])
    
    


# In[ ]:


fittings


# In[ ]:


fittings.columns


# In[ ]:


fittings.to_csv(data_dir+"fittings_with_beta_wo_effort.csv")


# In[ ]:


to_plot = fittings.melt(id_vars=['Subject','Species','Phase','spatial_bias','LL'])
to_plot


# In[ ]:


sns.catplot(x="Phase",y="value",hue='Species',data=to_plot,kind='point',dodge=True,col='variable',col_wrap=3,sharey=False)


# In[ ]:


sns.catplot(x="Phase",hue="spatial_bias",col='Species',kind='count',data=to_plot)


# In[ ]:


# data_trajs


# In[ ]:





# In[ ]:





# In[ ]:


res.x


# In[ ]:





# In[ ]:


sim_trajs = []

Nsim = 10

for traj in data_trajs:

    start = traj[0]
    for i in range(Nsim):
        sim, _, _, _ = simulate_case(
            kernel_type="row_scan",
            start_loc=start,
            T=len(traj),
            sigma=sigma,
            decay=decay,
            wK=wK,
            wR=wR,
            wM=wM
        )

    sim_trajs.append(sim)


# In[ ]:


Vobs = sum(visit_map(t) for t in data_trajs)


# In[ ]:


Vsim = sum(visit_map(t) for t in sim_trajs)


# In[ ]:


Vobs/np.sum(Vobs)


# In[ ]:


Vsim


# In[ ]:


fig,ax = plt.subplots(1,2)

ax[0].imshow(Vobs/np.sum(Vobs),vmin=0,vmax=0.1)
ax[0].set_title("Monkey")

im = ax[1].imshow(Vsim/np.sum(Vsim),vmin=0,vmax=0.1)
ax[1].set_title("Model")
pl.colorbar(im)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

# observed
obs = np.array(data_trajs[2])
plt.plot(obs[:,1], obs[:,0], '-o', lw=3, label='Observed')

# simulated
sim = np.array(sim_trajs[0])
plt.plot(sim[:,1], sim[:,0], '--o', alpha=.7, label='Model')

plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()


# In[ ]:


fig,ax = pl.subplots(1,2,figsize=(12,6))
ax[0].pcolor(scan_map)
ax[1].pcolor(spiral_map)
pl.colorbar()


# In[ ]:




