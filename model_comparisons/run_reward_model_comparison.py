
#!/usr/bin/env python

import numpy as np
import pandas as pd
import belief_mix_model_reward_comparison as bmm
import os

DATA_DIR = "../Lucas_Cecile/"

processed_data = pd.read_csv(DATA_DIR + "extracted_trajs.csv")
random_agent = pd.read_csv(DATA_DIR + "random_agent.csv")
random_agent["Species"] = "Random"

all_data = pd.concat([processed_data, random_agent], ignore_index=True)

MODELS = ["A", "B", "C"]

#random_to_run = np.random.choice(range(0,20),5,replace=False) # Run only a subset of random
random_to_run = np.array([ 4, 17, 14,  5,  3])
print(random_to_run)

all_fittings = []

for model in MODELS:
    print("\n" + "=" * 80)
    print(f"FITTING MODEL {model}")
    print("=" * 80)

    fittings = []
    if os.path.exists(DATA_DIR+f"fittings_reward_model_{model}.csv"):
        print("Model "+ model +" already done !")
        model_df = pd.read_csv(DATA_DIR+f"fittings_reward_model_{model}.csv")
        all_fittings.append(model_df)
        continue
    
    for group_key, group in all_data.groupby(
        ["Subject", "Phase", "Species"]
    ):
        Subject, Phase, Species = group_key
        if "Random agent" in Subject and int(Subject.split(' ')[-1]) not in random_to_run:
            continue
        
        data_trajs = []
        for _, session in group.groupby("Session"):
            traj = [(r, c) for r, c in zip(session.Rows, session.Cols)]
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
        DATA_DIR + f"fittings_reward_model_{model}.csv",
        index=False
    )

comparison = pd.concat(all_fittings, ignore_index=True)
comparison.to_csv(
    DATA_DIR + "fittings_reward_models_A_B_C.csv",
    index=False
)

# Direct paired comparison: same Subject/Phase/Species
wide = comparison.pivot_table(
    index=["Subject", "Phase", "Species"],
    columns="Model",
    values=["NLL", "NLL_per_choice"]
)

wide.to_csv(
    DATA_DIR + "fittings_reward_models_A_B_C_wide.csv"
)

print("\nSaved:")
print(DATA_DIR + "fittings_reward_models_A_B_C.csv")
print(DATA_DIR + "fittings_reward_models_A_B_C_wide.csv")
