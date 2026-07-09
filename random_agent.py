import pandas as pd
import numpy as np
from pathlib import Path

#Import the score20_rhesus and score20_tonk datasets before starting
data_dir = "../Lucas_Cecile/"
score20_rhesus = pd.read_excel(data_dir+"score20_rhesus.xlsx")
score20_tonk = pd.read_excel(data_dir+"score20_tonk.xlsx")

# Merge the two datasets
score20 = pd.concat(
    [
        score20_rhesus.assign(Species="Macaca mulatta"),
        score20_tonk.assign(Species="Macaca tonkeana")
    ],
    ignore_index=True
)

score20["Species"] = pd.Categorical(
    score20["Species"],
    categories=["Macaca mulatta", "Macaca tonkeana"],
    ordered=True
)

# Put the hex codes here if we want to keep the colours of the R plot
col_mulatta = "#4E79A7"
col_tonkeana = "#F28E2B"

species_colors = {
    "Macaca mulatta": col_mulatta,
    "Macaca tonkeana": col_tonkeana
}

# We only keep the test data
score20_test = score20.loc[score20["Phase"] == "Test"].copy()
score20_test["Subject"] = score20_test["Subject"].astype(str)


# Random agent simulation
def simulate_one_session(n_trials, n_doors, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    #choices = rng.integers(1, n_doors + 1, size=n_trials)
    rows = rng.integers(low=0, high=5,size=n_trials)
    cols = rng.integers(low=0, high=5,size=n_trials)
    choices = [(r,c)  for r,c in zip(rows,cols)]
    opened = set()
    outcomes = []

    for door in choices:
        print(door)
        if door in opened:
            #print("already opened, no reward")
            outcomes.append(0)
            opened.add(door)
        else:
            opened.add(door)
            #print("reward found")
            outcomes.append(1)

    return outcomes, rows, cols



columns_ref = np.array(['A','B','C','D','E'])
rows_ref = np.arange(1,6)



# Reproducibility
rng = np.random.default_rng(123)

n_sessions = 10
n_subjects = 20
phases = ['Habituation','Test','Post-test']
random_agent = pd.DataFrame()

for ph in phases:
    for sub in range(n_subjects):
        for ns in range(n_sessions):
            #print("new sesssion ===============================================",ns)
            ans = simulate_one_session(20,25,rng=rng)
            temp = pd.DataFrame()
            temp['Rows'] = ans[1]
            temp['Reward'] = ans[0]
            temp['Cols'] = ans[2]
            temp['Session'] = ns
            temp['Phase'] = ph
            temp['Subject'] = 'Random agent '+str(sub)
            temp['Species'] = 'Random'

            random_agent = pd.concat([random_agent,temp])


random_agent.to_csv(data_dir+"random_agent.csv")
#random_perf = pd.DataFrame({
#    "Subject": ["Random agent"] * 100,
#    "Simulation": np.arange(1, 101),
#    "Rate20": [
#        np.mean(simulate_one_session(20, 25, rng=rng))
#        for _ in range(100)
#    ]
#})
