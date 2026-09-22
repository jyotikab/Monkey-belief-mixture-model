import numpy as np
# from scipy.special import softmax
import pandas as pd
import seaborn as sns
import pylab as pl
from scipy.optimize import minimize, differential_evolution
import scipy as sp
from scipy.spatial.distance import cdist,pdist
import pdb

data_dir = "../Lucas_Cecile/"
processed_data = pd.read_csv(data_dir+"extracted_trajs.csv")



spatial_kernels = ["row_scan",
                   "column_scan",
                   "spiral",
                   "random",
                   ]


GRID = 5
N = GRID * GRID

# Coordinates of each slot
coords = np.array([(r, c) for r in range(GRID) for c in range(GRID)])

dist_matrix = np.zeros((GRID*GRID,GRID*GRID))
for i,(r,c) in enumerate(coords):
    for j, (r1,c1) in enumerate(coords):
        dist_matrix[i][j] = sp.spatial.distance.pdist([(r,c),(r1,c1)])


coord_to_index = {
    tuple(c): i
    for i, c in enumerate(coords)
}

X, Y = np.meshgrid(np.arange(GRID), np.arange(GRID))




def row_kernel(start_r, start_c):
    K = np.zeros((GRID, GRID))

    for r in range(GRID):
        for c in range(GRID):
            # Penalize leaving the starting row
            row_cost = abs(r - start_r)

            # Prefer nearby columns within that row
            col_cost = abs(c - start_c)

            K[r, c] = np.exp(-3.0 * row_cost - 0.3 * col_cost)

    K /= K.sum()
    return K

def column_kernel(start_r, start_c):
    K = np.zeros((GRID, GRID))

    for r in range(GRID):
        for c in range(GRID):
            # Penalize leaving the starting row
            row_cost = abs(r - start_r)

            # Prefer nearby columns within that row
            col_cost = abs(c - start_c)

            K[r, c] = np.exp(-3.0 * col_cost - 0.3 * row_cost)

    K /= K.sum()
    return K    

def spiral_kernel(start_r,start_c):
    center = (start_r, start_c)
    K = np.zeros((GRID, GRID))

    for r in range(GRID):
        for c in range(GRID):

            dist2 = (r-center[0])**2 + (c-center[1])**2

            angle = np.arctan2(r-center[1], c-center[0])
            radial = np.sqrt(dist2)

            K[r,c] = np.exp(-radial/2.0) * (np.cos(2*angle) + 1.5)

    K = K - K.min()
    K /= K.sum()

    return K

def random_kernel(start_r, start_c,seed=None):
    if seed is not None:
        np.random.seed(seed)

    K = np.random.rand(GRID, GRID)
    K /= K.sum()
    return K



def spatial_bias_mixture(start_r, start_c, wRS, wCS, wSS):
    """
    wRS, wCS, wSS = row_scan, column_scan, spiral_scan
    """

    K_row = row_kernel(start_r, start_c)
    K_col = column_kernel(start_r, start_c)
    K_spi = spiral_kernel(start_r, start_c)

    K = (
        wRS * K_row +
        wCS * K_col +
        wSS * K_spi
    )

    K /= K.sum()

    return K


def softmax_weights(z):
    z = np.asarray(z)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def update_memory(BM, loc, decay):
    BM *= decay

    BM[loc] = 1

    return BM


def softmax(x):
    x = x - np.max(x)# No negative exponents
    e = np.exp(x)
    return e / e.sum()

def update_reward(BR, reward, sigma):

    r, c = reward

    dist2 = (X-c)**2 + (Y-r)**2
    kernel = np.exp(-dist2/(2*sigma**2))

    BR += kernel
    BR[r,c] = 0


    return BR

def normalize(u):
    u_norm = u/np.sum(u)
    return u_norm

def policy_from_current(K, BR, BM, current, weights,beta):

#     wK, wR, wM, wE = weights
    wK, wR, wM = weights
    #wRS, wCS, wSS = spatial_weights

    utilities = np.zeros((5,5))
    
    K = normalize(K)
    BR = normalize(BR)
    BM = normalize(BM)
    BE = dist_matrix/np.max(dist_matrix) 
#     pdb.set_trace()
#     try:
#     print("current",current)
    current_id = coord_to_index[tuple(current)]    
#     print("current_id",current_id)
    effort = dist_matrix[current_id].reshape(GRID,GRID) 
    
    

    
    utilities = beta* (
          wK*K
        + wR*BR
        - wM*BM
        #- wE*effort
    )    
    
    return softmax(utilities)


def mixture_weights(z):
    z = np.asarray(z)
    #z = z - np.max(z)
    #a = np.exp(z)
    return z / z.sum()





def neg_log_likelihood(params, data):
    #pdb.set_trace()
#     zK, zR, zM,  decay,beta = params[3:]
    zRS, zCS, zSS = params[:3]
    
    # Spatial kernel weights
    wRS, wCS, wSS = softmax_weights([zRS, zCS, zSS])

    # Utility component weights
    wK, wR, wM,  decay,beta = params[3:]#softmax_weights([zK, zR, zM])
    
    
    sigma = 1.0

    nll = 0.0
    eps = 1e-12

    for traj in data:

        #K = np.ones((GRID, GRID)) / 25
        start = traj[0]
        #K = spatial_bias(start[0], start[1], kernel=kernel)        
        K = spatial_bias_mixture(start[0], start[1],wRS, wCS, wSS)
        
        #alpha = mixture_weights(params[:3])

        
        BM = np.zeros((GRID, GRID))
        BR = np.ones((GRID, GRID))
        

        curr = traj[0]
        BM = update_memory(BM, curr, decay)
        BR = update_reward(BR, curr, sigma)
        
        for i,next_loc in enumerate(traj[1:]):

            #K = spatial_bias(curr[0], curr[1], kernel=kernel)
            K = spatial_bias_mixture(curr[0], curr[1],wRS, wCS, wSS)
#             if i == 0:
#                 prev = curr            
            #print("prev,curr",prev,curr)
            probs = np.hstack(policy_from_current(K, BR, BM,curr, (wK, wR, wM),beta))

            idx = np.where((coords == next_loc).all(axis=1))[0][0]

            nll -= np.log(probs[idx] + eps)

            BM = update_memory(BM, next_loc, decay)
            BR = update_reward(BR, next_loc, sigma)
            curr = next_loc
            
    return nll




def fit_diff_evol(data):
    bounds = [
   
        # Spatial mixture logits
        (-5, 5),   # zRS
        (-5, 5),   # zCS
        (-5, 5),   # zSS

        # Utility mixture logits
        (0.01,10),     # wK
        (0.01,10),     # wR
        (0.01,10),     # wM
        
#         (0,5),     #wE
#         (0.1,3),   # sigma
        (0.2,0.99), # decay
        (0.1,20) # beta
    ]
    res = differential_evolution(
        neg_log_likelihood,
        bounds,
        tol=1e-1,
        args=(data,),
        polish=True,
        seed=1,
        workers=1        
    )    
    
    return res



def simulate_case(
                  start_loc=(2,2), # start location
                  T=25,
                  sigma=1.2,
                  decay=0.9,
                  wK =1/3., 
                  wR=1/3.,
                  wM=1/3.,
#                   wE=1.0,
                  beta=1.0,
                  wRS = 1/3.,
                  wCS = 1/3.,
                  wSS = 1/3.        
                 ):
#                   wE=1.0):
    sigma = 1.0
    # choose kernel
    #print("in simulate case")
    #K = spatial_bias(start_loc[0],start_loc[1],kernel_type)
    K = spatial_bias_mixture(start_loc[0],start_loc[1],wRS,wCS,wSS)

    
    BM = np.zeros((GRID, GRID))
    BR = np.ones((GRID, GRID))

    
    current = start_loc
    traj = [[current[0],current[1]]]
    prev = current
    BM = update_memory(BM, current, decay)
    BR = update_reward(BR, current, sigma)
    
    for t in range(T):
        
        K = spatial_bias_mixture(current[0], current[1], wRS,wCS,wSS)
        probs = policy_from_current(K, BR, BM,current, (wK, wR, wM),beta)
        #print(len(probs))

        idx = np.random.choice(len(coords), p=np.hstack(probs),size=1)
#         idx = np.random.choice(len(poo), p=probs)
        nxt = coords[idx]
        traj.append([nxt[0][0],nxt[0][1]])

        BM = update_memory(BM, nxt, decay)
        BR = update_reward(BR, nxt[0], sigma)

        current = [nxt[0][0],nxt[0][1]]#nxt
        prev = current

    return traj, K, BR, BM



def visit_map(traj):

    V = np.zeros((GRID,GRID))

    for r,c in traj:
        V[r,c] += 1

    return V

def revisit_rate(traj):

    visited = set()

    revisits = 0
    time_until_first_revisit = []
    
    for i,loc in enumerate(traj):

        if tuple(loc) in visited:
            revisits += 1
            time_until_first_revisit.append(i)
            
        visited.add(tuple(loc))
    
    if len(time_until_first_revisit) == 0:
        time_until_first_revisit.append(0)
        
    return revisits / len(traj), time_until_first_revisit[0]

def jump_lengths(traj):

    traj = np.array(traj)

    d = np.abs(np.diff(traj,axis=0))

    total_jumps = d.sum(axis=1)
    mean_jump = np.mean(total_jumps)
    var_jump = np.var(total_jumps)
    return mean_jump, var_jump,total_jumps

def search_entropy(traj):
    visit_dict = dict()
    for t in act_trajs[0]:
        if tuple(t) not in visit_dict.keys():
            visit_dict[tuple(t)] = 1
        else:
            visit_dict[tuple(t)] = visit_dict[tuple(t)]+1
    
def calc_total_distance_covered(traj):
    tot_dist = 0
    for i in range(len(traj)-1):
        d = sp.spatial.distance.pdist([traj[i],traj[i+1]])[0]
        tot_dist = tot_dist + d
    return tot_dist

def calc_dist_bw_traj(traj1,traj2):
    
    
    dist = np.linalg.norm(np.array(traj1)-np.array(traj2),axis=1).sum()
    corr = np.corrcoef(traj1,traj2)[0][1]
    
    return dist,corr




    
