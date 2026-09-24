
#!/usr/bin/env python

import numpy as np
import pandas as pd
import belief_mix_model_reward_comparison as bmm

DATA_DIR = "../Lucas_Cecile/"

NUM_TESTS_PER_PAR = 10
RANDOM_SEED = 123

rng = np.random.default_rng(RANDOM_SEED)

processed_data = pd.read_csv(DATA_DIR + "extracted_trajs.csv")

# Use all fitted rows from the corresponding model files.
# For C, the additional W(reward_availability) parameter is retained.
fitting_files = {
    "A": DATA_DIR + "fittings_reward_model_A.csv",
    "B": DATA_DIR + "fittings_reward_model_B.csv",
    "C": DATA_DIR + "fittings_reward_model_C.csv",
}

recovery_outputs = []

for model, fitting_file in fitting_files.items():

    fittings = pd.read_csv(fitting_file)

    print("\n" + "=" * 80)
    print(f"RECOVERY TEST: MODEL {model}")
    print("=" * 80)

    for i in range(len(fittings)):

        sub = fittings.iloc[i]

        Subject = sub["Subject"]
        Phase = sub["Phase"]
        Species = sub["Species"]

        act_data = processed_data.loc[
            (processed_data["Subject"] == Subject)
            & (processed_data["Phase"] == Phase)
            & (processed_data["Species"] == Species)
        ].copy()

        act_trajs = [
            [[x, y] for x, y in zip(tr.Rows, tr.Cols)]
            for _, tr in act_data.groupby("Session")
        ]

        act_trajs = [tr for tr in act_trajs if len(tr) >= 2]

        if not act_trajs:
            continue

        true_params = {
            "W(row_scan)": sub["W(row_scan)"],
            "W(column_scan)": sub["W(column_scan)"],
            "W(spiral_scan)": sub["W(spiral_scan)"],
            "W(spatial_bias)": sub["W(spatial_bias)"],
            "W(reward)": sub["W(reward)"],
            "W(memory)": sub["W(memory)"],
            "memory_recall": sub["memory_recall"],
            "beta": sub["beta"],
        }

        if model == "B":
            true_params["reward_decay"] = sub["reward_decay"]

        if model == "C":
            true_params["W(reward_availability)"] = (
                sub["W(reward_availability)"]
            )

        for iteration in range(NUM_TESTS_PER_PAR):

            sim_trajs = []

            for traj in act_trajs:
                start = tuple(traj[0])

                kwargs = dict(
                    start_loc=start,
                    T=len(traj) - 1,
                    sigma=1.0,
                    decay=sub["memory_recall"],
                    wK=sub["W(spatial_bias)"],
                    wR=sub["W(reward)"],
                    wM=sub["W(memory)"],
                    beta=sub["beta"],
                    wRS=sub["W(row_scan)"],
                    wCS=sub["W(column_scan)"],
                    wSS=sub["W(spiral_scan)"],
                    model=model,
                )

                if model == "B":
                    kwargs["reward_decay"] = sub["reward_decay"]

                if model == "C":
                    kwargs["wA"] = sub["W(reward_availability)"]

                sim = bmm.simulate_case(**kwargs)
                sim_trajs.append(sim[0])

            # Refit the same model to the simulated data.
            res = bmm.fit_diff_evol(
                sim_trajs,
                model=model,
                seed=iteration + 1000,
                tol=1e-1
            )

            recovered = bmm.unpack_result(res, model)

            row = {
                "Model": model,
                "Subject": Subject,
                "Phase": Phase,
                "Species": Species,
                "iter": iteration,
                "LL": res.fun,
            }

            for key, value in true_params.items():
                row[key] = value
                row[key + " - sim"] = recovered.get(key, np.nan)

            recovery_outputs.append(row)

            print(
                model, Subject, Phase, Species,
                "iteration", iteration
            )

recovery_df = pd.DataFrame(recovery_outputs)

out = DATA_DIR + "recovery_reward_models_A_B_C.csv"
recovery_df.to_csv(out, index=False)

print("\nSaved:")
print(out)
