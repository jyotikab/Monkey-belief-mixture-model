
#!/usr/bin/env python

import numpy as np
import pandas as pd
# import belief_mix_model_reward_comparison as bmm
# import belief_mix_model_reward_comparison_more_models as bmm
# import belief_mix_model_reward_comparison_more_models_with_effort as bmm
import belief_mix_model_reward_comparison_more_models_with_effort_wo_reward as bmm
import os
import pdb
import pylab as pl

DATA_DIR = "../Lucas_Cecile/"

# processed_data = pd.read_csv(DATA_DIR + "extracted_trajs.csv")
processed_data = pd.read_csv(DATA_DIR + "extracted_trajs_updated.csv")
random_agent = pd.read_csv(DATA_DIR + "random_agent.csv")
random_agent["Species"] = "Random"

all_data = pd.concat([processed_data, random_agent], ignore_index=True)

MODELS = ["A", "B", "C"]
# MODELS = ["A", "B"]
# MODELS = [ "B"]

#random_to_run = np.random.choice(range(0,20),5,replace=False) # Run only a subset of random
random_to_run = np.array([ 4, 17, 14,  5,  3])
print(random_to_run)

#postfix = ""
# postfix = "_with_high_res_space_updated_with_effort_"
postfix = "_with_high_res_space_updated_with_effort_wo_reward_"
# postfix = "_with_high_res_space_updated_one_param_less_"

all_fittings = []

for model in MODELS:
    print("\n" + "=" * 80)
    print(f"FITTING MODEL {model}")
    print("=" * 80)

    fittings = []
#     if os.path.exists(DATA_DIR+f"fittings_reward_model{postfix}_{model}.csv"):
#         print("Model "+ model +" already done !")
#         model_df = pd.read_csv(DATA_DIR+f"fittings_reward_model{postfix}_{model}.csv")
#         all_fittings.append(model_df)
#         continue
    
    for group_key, group in all_data.groupby(
        ["Subject", "Phase", "Species"]
    ):
        Subject, Phase, Species = group_key
        if "Random agent" in Subject and int(Subject.split(' ')[-1]) not in random_to_run:
            continue
        
        data_trajs = []
        for _, session in group.groupby("Session"):
            traj = [(r, c) for r, c in zip(session.Rows[:20], session.Cols[:20])] # Take only first 20 jumps
            if len(traj) >= 2:
                data_trajs.append(traj)

        if not data_trajs:
            continue

        print(f"{model}: {Subject}, {Phase}, {Species}")

        res = bmm.fit_diff_evol(
            data_trajs,
            model=model,
            seed=1,
#             tol=1e-3
            tol=1e-1
        )

        nchoices = bmm.count_choices(data_trajs)

        row = {
            "Model": model,
            "Subject": Subject,
            "Phase": Phase,
            "Species": Species,
            "NLL": res.fun,
            "NLL_per_choice": res.fun / nchoices,
            "Random_NLL_per_choice": np.log(25),
            "nchoices": nchoices,
            "success": res.success,
        }

        row.update(bmm.unpack_result(res, model))
        fittings.append(row)

        print(
            f"    NLL={res.fun:.3f}, "
            f"NLL/choice={res.fun/nchoices:.4f}"
        )

    model_df = pd.DataFrame(fittings)
    all_fittings.append(model_df)

    model_df.to_csv(
        DATA_DIR + f"fittings_reward_model{postfix}_{model}.csv",
        index=False
    )

comparison = pd.concat(all_fittings, ignore_index=True)
comparison.to_csv(
    DATA_DIR + "fittings_reward_models"+"_".join(MODELS)+postfix+".csv",
    index=False
)

# Direct paired comparison: same Subject/Phase/Species
wide = comparison.pivot_table(
    index=["Subject", "Phase", "Species"],
    columns="Model",
    values=["NLL", "NLL_per_choice"]
)

wide.to_csv(
#     DATA_DIR + "fittings_reward_models_A_B_C_wide.csv"
    DATA_DIR + "fittings_reward_models"+"_".join(MODELS)+postfix+"_wide.csv"
)

print("\nSaved:")
# print(DATA_DIR + "fittings_reward_models_A_B_C.csv")
# print(DATA_DIR + "fittings_reward_models_A_B_C_wide.csv")
print(DATA_DIR + "fittings_reward_models"+"_".join(MODELS)+postfix+".csv")
print(DATA_DIR + "fittings_reward_models"+"_".join(MODELS)+postfix+"_wide.csv")

'''
MODEL_B_NAMES = ["left", "right", "up", "down","up_left","up_right","down_left","down_right"]
fig,ax = plt.subplots(2,4,figsize=(12,6))
axes = np.hstack(ax)
for i,name in enumerate(MODEL_B_NAMES):
     kernel = directional_kernel(start_r, start_c, DIRECTIONS[name])
     sns.heatmap(kernel,cmap='rocket',vmin=0,vmax=0.4,ax=axes[i])
     axes[i].plot(start_r+0.5,start_c+0.5,'*',color='cyan',ms=10)
     axes[i].set_title(name)
plt.tight_layout()
fig.savefig('figures/brinkman_board/kernels_model_B.png',dpi=300)

MODEL_C_NAMES = ["left", "right", "up", "down","clockwise_spiral", "counter_clockwise_spiral"]
fig,ax = plt.subplots(2,3,figsize=(12,6))
axes = np.hstack(ax)
for i,name in enumerate(MODEL_C_NAMES):
     if 'spiral' not in name:
         kernel = directional_kernel(start_r, start_c, DIRECTIONS[name])
     else:
         if 'counter' in name:
             kernel = counterclockwise_spiral_kernel(start_r, start_c)
         else:
             kernel = clockwise_spiral_kernel(start_r, start_c)
     sns.heatmap(kernel,cmap='rocket',vmin=0,vmax=0.4,ax=axes[i])
     
     axes[i].plot(start_r+0.5,start_c+0.5,'*',color='cyan',ms=10)
     axes[i].set_title(name)
plt.tight_layout()
fig.savefig('figures/brinkman_board/kernels_model_C.png',dpi=300)

     
MODEL_A_NAMES = ['row','column',"clockwise_spiral", "counter_clockwise_spiral"]
var_names = [row_kernel, column_kernel, clockwise_spiral_kernel, counterclockwise_spiral_kernel]
start_r = 2
start_c = 2
fig,ax = plt.subplots(2,2,figsize=(6,6))
axes = np.hstack(ax)
for i,name in enumerate(MODEL_A_NAMES):
    kernel = var_names[i](start_r,start_c)
    sns.heatmap(kernel,cmap='rocket',vmin=0,vmax=0.4,ax=axes[i])
    axes[i].plot(start_r+0.5,start_c+0.5,'*',color='cyan',ms=10)
    axes[i].set_title(name)    
plt.tight_layout()
fig.savefig('figures/brinkman_board/kernels_model_A.png',dpi=300)

'''

