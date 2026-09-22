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

num_tests_per_par = 10

fittings =pd.read_csv(data_dir+"fittings_with_spatial_mixture_bias_normalized.csv")
processed_data = pd.read_csv(data_dir+"extracted_trajs.csv")

random_to_run = np.random.choice(range(0,20),5,replace=False) # Run only a subset of random
print(random_to_run)

recovery_df = pd.DataFrame()

for i in range(len(fittings)):
    sub_dat = fittings.iloc[i].copy()
    
    wRS = sub_dat['W(row_scan)']
    wCS = sub_dat['W(column_scan)']
    wSS = sub_dat['W(spiral_scan)']
    wK = sub_dat['W(spatial_bias)']
    wR = sub_dat['W(reward)']
    wM = sub_dat['W(memory)']
    decay = sub_dat['memory_recall ('+r'$\lambda$'+')']
    beta = sub_dat['beta']
    
    Subject = sub_dat['Subject']
    Phase = sub_dat['Phase']
    Species = sub_dat['Species']
    
    
    recov_temp = pd.DataFrame(sub_dat.copy()).T
    act_data = processed_data.loc[(processed_data['Subject']==Subject)&(processed_data['Phase']==Phase)&(processed_data['Species']==Species)].copy()
    act_trajs = [ [[x,y]  for x,y in zip(tr[1].Rows,tr[1].Cols)] for tr in act_data.groupby('Session') ]

    
    if "Random agent" in sub_dat.Subject and int(sub_dat.Subject.split(' ')[-1]) not in random_to_run:
        continue
    
    for num in range(num_tests_per_par):
        sim_trajs = []
        for i,traj in enumerate(act_trajs):
            start = traj[0]
            
            sim, _, _, _ = bmm_spat.simulate_case(               
                        start_loc=start,
                        T=len(traj)-1,
                        sigma=1.0,
                        decay=decay,
                        wK=wK,
                        wR=wR,
                        wM=wM,
                        beta=beta,
                        wRS = wRS,
                        wCS = wCS,
                        wSS = wSS
                    )                        
            sim_trajs.append(sim)
        
        print(Subject, Phase, Species, num)
        print("==================================")
        best_res = bmm_spat.fit_diff_evol(sim_trajs)
        temp = pd.DataFrame()
        temp['Subject'] = [Subject]
        temp['Phase'] = [Phase]
        temp['Species'] = [Species]        
        temp['LL'] = [best_res.fun]
        temp['LL / nchoices'] = [best_res.fun/len(act_data)]
        temp['Random NLL / nchoices'] = [np.log(25)]

        zRS, zCS, zSS = best_res.x[:3]
        #zK, zR, zM = best_res.x[3:6]

        wRS, wCS, wSS = bmm_spat.softmax_weights([zRS, zCS, zSS])
        wK, wR, wM = best_res.x[3:6] #bmm_spat.softmax_weights([zK, zR, zM])    


        temp['W(row_scan) - sim'] = [wRS]
        temp['W(column_scan) - sim'] = [wCS]
        temp['W(spiral_scan) - sim'] = [wSS]


        temp['W(spatial_bias) - sim'] = [wK]
        temp['W(reward) - sim'] = [wR]
        temp['W(memory) - sim'] = [wM]
        temp['memory_recall ('+r'$\lambda$'+') - sim'] = [best_res.x[6]]
        temp['beta - sim'] = [best_res.x[7]]    
        temp['iter'] = [num]
        
        recov_temp = pd.concat([recov_temp,temp])
    
    recovery_df = pd.concat([recovery_df,recov_temp])
    
recovery_df.to_csv(data_dir+"recovery_df_spatial_mix_normalized.csv")