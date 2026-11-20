import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import pdb
import pylab as pl

# =============================================================================
# Configuration
# =============================================================================

GRID = 5
N = GRID * GRID

coords = np.array([(r, c) for r in range(GRID) for c in range(GRID)])
coord_to_index = {tuple(x): i for i, x in enumerate(coords)}
X, Y = np.meshgrid(np.arange(GRID), np.arange(GRID))

# Row increases downwards; column increases to the right.
DIRECTIONS = {
    "left":       (0, -1),
    "right":      (0, 1),
    "up":          (-1, 0),
    "down":        (1, 0),
    "up_left":     (-1, -1),
    "up_right":    (-1, 1),
    "down_left":   (1, -1),
    "down_right":  (1, 1),
}



# Model C: four cardinal directions plus two grouped diagonal biases.
MODEL_C_NAMES = [
    "left", "right", "up", "down",
    "clockwise_spiral", "counter_clockwise_spiral"
]

DIRECTION_NAMES = list(DIRECTIONS)


# =============================================================================
# General helpers
# =============================================================================

def normalize(x):
    x = np.asarray(x, dtype=float)
    total = x.sum()
    if total <= 0:
        return np.ones_like(x) / x.size
    return x / total


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def softmax_weights(z):
    return softmax(z)


# =============================================================================
# Spatial kernels for Model A: global scanning strategies
# =============================================================================

def row_kernel(start_r, start_c):
    """Prefer locations in the same row as the current location."""
    K = np.zeros((GRID, GRID))
    for r in range(GRID):
        for c in range(GRID):
            row_cost = abs(r - start_r)
            col_cost = abs(c - start_c)
            K[r, c] = np.exp(-3.0 * row_cost - 0.3 * col_cost)
    return normalize(K)


def column_kernel(start_r, start_c):
    """Prefer locations in the same column as the current location."""
    K = np.zeros((GRID, GRID))
    for r in range(GRID):
        for c in range(GRID):
            row_cost = abs(r - start_r)
            col_cost = abs(c - start_c)
            K[r, c] = np.exp(-3.0 * col_cost - 0.3 * row_cost)
    return normalize(K)


# def spiral_kernel(start_r, start_c):
#     """Original spiral-like kernel, retained for Model A."""
#     K = np.zeros((GRID, GRID))
#     for r in range(GRID):
#         for c in range(GRID):
#             dr = r - start_r
#             dc = c - start_c
#             radial = np.sqrt(dr**2 + dc**2)
#             angle = np.arctan2(dr, dc)
#             K[r, c] = np.exp(-radial / 2.0) * (np.cos(2.0 * angle) + 1.5)

#     K -= K.min()
#     return normalize(K)

def clockwise_spiral_kernel(start_r, start_c):
    K_left = directional_kernel(
        start_r, start_c, DIRECTIONS["left"]
    )
    K_right = directional_kernel(
        start_r, start_c, DIRECTIONS["right"]
    )
    K_up = directional_kernel(
        start_r, start_c, DIRECTIONS["up"]
    )
    K_down = directional_kernel(
        start_r, start_c, DIRECTIONS["down"]
    )

    # Diagonal directional kernels
    K_ul = directional_kernel(
        start_r, start_c, DIRECTIONS["up_left"]
    )
    K_ur = directional_kernel(
        start_r, start_c, DIRECTIONS["up_right"]
    )
    K_dl = directional_kernel(
        start_r, start_c, DIRECTIONS["down_left"]
    )
    K_dr = directional_kernel(
        start_r, start_c, DIRECTIONS["down_right"]
    )

    K_cw = normalize(
        0.7 * K_ul +
        0.6 * K_up +
        0.5 * K_ur +
        0.4 * K_right +
        0.3 * K_dr +
        0.2 * K_down +
        0.1 * K_dl
    )
    
    return K_cw

def counterclockwise_spiral_kernel(start_r, start_c):
    K_left = directional_kernel(
        start_r, start_c, DIRECTIONS["left"]
    )
    K_right = directional_kernel(
        start_r, start_c, DIRECTIONS["right"]
    )
    K_up = directional_kernel(
        start_r, start_c, DIRECTIONS["up"]
    )
    K_down = directional_kernel(
        start_r, start_c, DIRECTIONS["down"]
    )

    # Diagonal directional kernels
    K_ul = directional_kernel(
        start_r, start_c, DIRECTIONS["up_left"]
    )
    K_ur = directional_kernel(
        start_r, start_c, DIRECTIONS["up_right"]
    )
    K_dl = directional_kernel(
        start_r, start_c, DIRECTIONS["down_left"]
    )
    K_dr = directional_kernel(
        start_r, start_c, DIRECTIONS["down_right"]
    )
    K_ccw = normalize(
        0.7 * K_dl +
        0.6 * K_down +
        0.5 * K_dr +
        0.4 * K_right +
        0.3 * K_ur +
        0.2 * K_up +
        0.1 * K_ul
    )    
    
    return K_ccw

def spatial_bias_A(start_r, start_c, weights):
    """Four-component global scanning mixture for Model A."""
    
#     w_row, w_col, w_spiral = weights
    w_row, w_col, w_cw_spiral, w_ccw_spiral = weights
    K = (
        w_row * row_kernel(start_r, start_c)
        + w_col * column_kernel(start_r, start_c)
        #+ w_spiral * spiral_kernel(start_r, start_c)
        + w_cw_spiral * clockwise_spiral_kernel(start_r, start_c)
        + w_ccw_spiral * counterclockwise_spiral_kernel(start_r, start_c)
    )
    return normalize(K)



# =============================================================================
# Spatial kernels for Model B: eight local movement directions
# =============================================================================

# def directional_kernel(start_r, start_c, direction,
# #                        sigma_parallel=0.5, sigma_perpendicular=0.5):
#                        sigma_parallel=0.75, sigma_perpendicular=0.65):
#     """
#     A directional kernel centered approximately one grid step from the
#     current location in the specified direction.

#     Diagonal direction vectors are normalized, so their preferred displacement
#     has the same Euclidean length as a cardinal one-step displacement.
#     The kernel is truncated at the grid boundaries and then normalized.
#     """
#     dr, dc = direction
#     norm = np.sqrt(dr**2 + dc**2)
#     ur, uc = dr / norm, dc / norm

#     # Displacement from the current position to every candidate location.
#     drow = Y - start_r
#     dcol = X - start_c

#     # Resolve displacement into components parallel/perpendicular to direction.
#     parallel = drow * ur + dcol * uc
#     perpendicular = -drow * uc + dcol * ur

# #     distance = np.sqrt(drow**2 + dcol**2)
# #     distance_scale = 0.5 # penalize distance to reduce overall jump length for B
    
#     # Favor a location about one step forward, with limited lateral spread.
#     K = np.exp(
#         -0.5 * ((parallel - 1.0) / sigma_parallel) ** 2
# #         -0.2 * (perpendicular / sigma_perpendicular) ** 2
# #         -distance / distance_scale 
#     )
#     return normalize(K)
def directional_kernel(start_r, start_c, direction,
                       decay_parallel=0.5):

    """
    Directional kernel with:
    - Zero probability outside the directional axis.
    - Maximum preference one step along the direction.
    - Gradual exponential decay along the parallel direction.
    """

    dr, dc = direction
    norm = np.sqrt(dr**2 + dc**2)

    ur, uc = dr / norm, dc / norm

    # Displacement from current location
    drow = Y - start_r
    dcol = X - start_c

    # Parallel and perpendicular displacement
    parallel = drow * ur + dcol * uc
    perpendicular = -drow * uc + dcol * ur

    # Strict directional constraint
    K = np.zeros((GRID, GRID), dtype=float)

    mask = (np.abs(perpendicular) < 1e-10) & (parallel > 0)

    # Gradual exponential decay along the direction
    K[mask] = np.exp(
        -decay_parallel * (parallel[mask] - 1.0)
    )

    return normalize(K)



def spatial_bias_B(start_r, start_c, direction_weights):
    """Eight-component directional movement mixture for Model B."""
    
    K = np.zeros((GRID, GRID), dtype=float)
    
    for weight, name in zip(direction_weights, DIRECTION_NAMES):
        K += weight * directional_kernel(start_r, start_c, DIRECTIONS[name])

    return normalize(K)


def spatial_bias_C(start_r, start_c, spatial_weights):
    """
    Six-component spatial mixture:
    left, right, up, down, clockwise spiral,
    counter-clockwise spiral.
    """

    w_left, w_right, w_up, w_down, w_cw, w_ccw = spatial_weights
    
    # Cardinal directional kernels
    K_left = directional_kernel(
        start_r, start_c, DIRECTIONS["left"]
    )
    K_right = directional_kernel(
        start_r, start_c, DIRECTIONS["right"]
    )
    K_up = directional_kernel(
        start_r, start_c, DIRECTIONS["up"]
    )
    K_down = directional_kernel(
        start_r, start_c, DIRECTIONS["down"]
    )

    # Diagonal directional kernels
    K_ul = directional_kernel(
        start_r, start_c, DIRECTIONS["up_left"]
    )
    K_ur = directional_kernel(
        start_r, start_c, DIRECTIONS["up_right"]
    )
    K_dl = directional_kernel(
        start_r, start_c, DIRECTIONS["down_left"]
    )
    K_dr = directional_kernel(
        start_r, start_c, DIRECTIONS["down_right"]
    )

    # Clockwise spiral
#     K_cw = normalize(
#         0.4 * K_ul +
#         0.3 * K_ur +
#         0.2 * K_dr +
#         0.1 * K_dl
#     )

#     # Counter-clockwise spiral
#     K_ccw = normalize(
#         0.4 * K_dl +
#         0.3 * K_dr +
#         0.2 * K_ur +
#         0.1 * K_ul
#     )
    K_cw = normalize(
        0.7 * K_ul +
        0.6 * K_up +
        0.5 * K_ur +
        0.4 * K_right +
        0.3 * K_dr +
        0.2 * K_down +
        0.1 * K_dl
    )

    # Counter-clockwise spiral
    K_ccw = normalize(
        0.7 * K_dl +
        0.6 * K_down +
        0.5 * K_dr +
        0.4 * K_right +
        0.3 * K_ur +
        0.2 * K_up +
        0.1 * K_ul
    )

    # Combine the six spatial strategies
    K = (
        w_left * K_left +
        w_right * K_right +
        w_up * K_up +
        w_down * K_down +
        w_cw * K_cw +
        w_ccw * K_ccw
    )

    return normalize(K)    

# =============================================================================
# Reward and visit-memory updates (identical in Models A and B)
# =============================================================================

def update_memory(BM, loc, decay):
    """Visit memory decays, then the current location is marked as visited."""
    BM *= decay
    BM[tuple(loc)] = 1.0
    return BM


def gaussian_kernel(loc, sigma):
    r, c = loc
    dist2 = (X - c) ** 2 + (Y - r) ** 2
    return np.exp(-dist2 / (2.0 * sigma**2))


def update_reward(BR, loc, sigma):
    """
    Persistent Gaussian reward/search-value memory.
    The exact location just visited is set to zero.
    """
    BR += gaussian_kernel(loc, sigma)
    BR[tuple(loc)] = 0.0
    return BR

def calculate_effort(start_r, start_c):
    distance = np.zeros((GRID, GRID))
    
    for r in range(GRID):
        for c in range(GRID):
            dist = cdist([(r,c)],[(start_r,start_c)])[0][0]
            distance[r][c] = dist
    
    return distance

# =============================================================================
# Choice policy
# =============================================================================

def policy_from_current(K, BR, BM, EM, weights, beta):
    """
    Shared policy for both models:
        utility = wK*K + wR*BR - wM*BM
    """
    wK, wR, wM, wE = weights

    K = normalize(K)
    BR = normalize(BR)
    BM = normalize(BM)
    EM = normalize(EM)

    utilities = wK * K + wR * BR - wM * BM - wE * EM
    return softmax(beta * utilities)


# =============================================================================
# Model definitions and likelihood
# =============================================================================

def model_spatial_kernel(current, spatial_weights, model):
    r, c = current

    if model == "A":
        return spatial_bias_A(r, c, spatial_weights)
    if model == "B":
        return spatial_bias_B(r, c, spatial_weights)
    if model == "C":
        return spatial_bias_C(r, c, spatial_weights)

    raise ValueError("model must be 'A', 'B' or 'C'")

def get_parameter_count(model):
    if model == "A":
        return 4 + 4 + 2  # spatial logits + utility weights + decay/beta
    if model == "B":
        return 8 + 4 + 2  # direction logits + utility weights + decay/beta
    if model == 'C':
        return 6 + 4 + 2
    raise ValueError("model must be 'A' or 'B' or 'C'")


def neg_log_likelihood(params, data, model="A", sigma=1.0):
    """
    Parameter order:
      Model A:
        z_row, z_column, z_spiral, wK, wR, wM, memory_decay, beta

      Model B: seven free directional logits, followed by wK, wR, wM,
        memory_decay and beta.
      Model C: five free logits for left, right, up, down and clockwise_spiral,
        followed by wK, wR, wM, memory_decay and beta. The counter-clockwise
        spiral logit is fixed to zero as the softmax reference.

    Spatial logits are transformed to mixture weights with softmax. Model A
    likewise uses two free logits and a fixed reference spiral logit.
    """
    
    #n_spatial = 3 if model == 'A' else 8
    if model == 'A':
        n_spatial = 4
    if model == 'B':
        n_spatial = 8
    if model == 'C':
        n_spatial = 6
    
    spatial_weights = softmax_weights(params[:n_spatial])
#     if model == 'A':
    
#         spatial_weights = softmax_weights(np.array([params[0], params[1], 0.0]))
#     elif model == 'B':
#         # Model B: 7 free logits
#         spatial_weights = softmax_weights(np.array([*params[:7],0.0]))
    
    wK, wR, wM,wE, decay, beta = params[n_spatial:]

    nll = 0.0
    eps = 1e-12

    for traj in data:
        if len(traj) < 2:
            continue

        # Validate and convert trajectory locations to tuples.
        traj = [tuple(map(int, loc)) for loc in traj]
        if any(loc not in coord_to_index for loc in traj):
            raise ValueError(
                f"Trajectory contains a location outside the {GRID}x{GRID} grid."
            )

        current = traj[0]
        BM = np.zeros((GRID, GRID), dtype=float)
        BR = np.ones((GRID, GRID), dtype=float)

        BM = update_memory(BM, current, decay)
        BR = update_reward(BR, current, sigma)
        EM = calculate_effort(current[0], current[1])
        for next_loc in traj[1:]:
            K = model_spatial_kernel(current, spatial_weights, model)
            EM = calculate_effort(current[0],current[1])
            probs = policy_from_current(
                K, BR, BM,EM, (wK, wR, wM,wE), beta
            )

            nll -= np.log(probs[next_loc] + eps)

            BM = update_memory(BM, next_loc, decay)
            BR = update_reward(BR, next_loc, sigma)
            current = next_loc

    return float(nll)


def get_bounds(model="A"):
    """Bounds in the parameter order documented in neg_log_likelihood."""
    if model == 'A':
        n_spatial = 4
    if model == 'B':
        n_spatial = 8
    if model == 'C':
        n_spatial = 6    
#     spatial_n = 2 if model == "A" else 7
    spatial_bounds = [(-5.0, 5.0)] * n_spatial
    utility_bounds = [(0.01, 10.0)] * 4
    memory_policy_bounds = [(0.2, 0.99), (0.1, 20.0)]

#     if model not in ("A", "B",'C'):
#         raise ValueError("model must be 'A' or 'B' or 'C'")

    return spatial_bounds + utility_bounds + memory_policy_bounds


def fit_diff_evol(data, model="A", seed=42, tol=1e-3,
                  polish=True, maxiter=1000, popsize=25):
    """Fit one model by differential evolution."""
    result = differential_evolution(
        neg_log_likelihood,
        get_bounds(model),
        args=(data, model),
        tol=tol,
        polish=polish,
        seed=seed,
        workers=1,
        maxiter=maxiter,
        popsize=popsize,
        strategy="best1bin",
        mutation=(0.5, 1.0),
        recombination=0.7,
        init="sobol",    
#         disp=True
    )
    return result


# =============================================================================
# Model comparison: likelihood, AIC and BIC
# =============================================================================

def unpack_result(result, model):
    if model == 'A':
        n_spatial = 4
    if model == 'B':
        n_spatial = 8
    if model == 'C':
        n_spatial = 6    
    
    
    spatial_weights = softmax_weights(result.x[:n_spatial])
    wK, wR, wM,wE, decay, beta = result.x[n_spatial:]

    out = {}
    if model == "A":
        out.update({
            "WS(row)": spatial_weights[0],
            "WS(column)": spatial_weights[1],
            "WS(cw_spiral)": spatial_weights[2],
            "WS(ccw_spiral)":spatial_weights[3],
        })
    elif model == "B":
        for name, weight in zip(DIRECTION_NAMES, spatial_weights):
            out[f"WS({name})"] = weight
    else:
        for name, weight in zip(MODEL_C_NAMES, spatial_weights):
            out[f"WS({name})"] = weight


    out.update({
        "W(spatial_bias)": wK,
        "W(reward)": wR,
        "W(memory)": wM,
        "W(effort)":wE,
        "memory_recall": decay,
        "beta": beta,
    })
    return out


def compare_models(data, seed=1, tol=1e-3, polish=True,
                   maxiter=1000, popsize=15):
    """
    Fit A and B and return:
      - fitted scipy OptimizeResult objects
      - a comparison DataFrame with NLL, AIC and BIC
      - a parameter DataFrame

    BIC uses the total number of observed transitions as the sample size.
    """
    results = {}
    rows = []
    parameter_rows = []

    n_choices = sum(max(0, len(traj) - 1) for traj in data)
    if n_choices == 0:
        raise ValueError("The data contain no transitions to fit.")

    for model in ("A", "B","C"):
        result = fit_diff_evol(
            data,
            model=model,
            seed=seed,
            tol=tol,
            polish=polish,
            maxiter=maxiter,
            popsize=popsize,
        )
        results[model] = result

        k = get_parameter_count(model)
        nll = result.fun
        rows.append({
            "model": model,
            "n_parameters": k,
            "n_choices": n_choices,
            "negative_log_likelihood": nll,
            "log_likelihood": -nll,
            "AIC": 2 * k + 2 * nll,
            "BIC": np.log(n_choices) * k + 2 * nll,
            "converged": result.success,
        })

        parameter_rows.append({
            "model": model,
            **unpack_result(result, model),
        })

    comparison = pd.DataFrame(rows).set_index("model")
    comparison["delta_AIC"] = comparison["AIC"] - comparison["AIC"].min()
    comparison["delta_BIC"] = comparison["BIC"] - comparison["BIC"].min()

    parameters = pd.DataFrame(parameter_rows).set_index("model")
    return results, comparison.sort_values("AIC"), parameters


# =============================================================================
# Simulation
# =============================================================================

def simulate_case(
    start_loc=(2, 2),
    T=25,
    sigma=1.0,
    decay=0.9,
    wK=1.0,
    wR=1.0,
    wM=1.0,
    wE=1.0,
    beta=1.0,
    spatial_weights=None,
    model="A",
    seed=None,
):
    """
    Simulate one trajectory.

    For Model A, spatial_weights has 3 entries:
        row, column, spiral.

    For Model B, spatial_weights has 8 entries in DIRECTION_NAMES order:
        left, right, up, down, up_left, up_right, down_left, down_right.

    The supplied spatial_weights are normalized internally.
    """
    if model not in ("A", "B","C"):
        raise ValueError("model must be 'A' or 'B' or 'C'")

    if spatial_weights is None:
        if model == 'A':
            n = 4
        elif model == 'B':
            n = 8
        elif model == 'C':
            n = 6
        spatial_weights = np.ones(n) / n
    else:
        spatial_weights = spatial_weights#normalize(spatial_weights)

    rng = np.random.default_rng(seed)
    current = tuple(start_loc)
    if current not in coord_to_index:
        raise ValueError("start_loc must be inside the grid.")

    BM = np.zeros((GRID, GRID), dtype=float)
    BR = np.ones((GRID, GRID), dtype=float)

    traj = [[current[0], current[1]]]
    BM = update_memory(BM, current, decay)
    BR = update_reward(BR, current, sigma)
    EM = calculate_effort(current[0],current[1])
    for _ in range(T):
        K = model_spatial_kernel(current, spatial_weights, model)
        EM = calculate_effort(current[0],current[1])
        probs = policy_from_current(
            K, BR, BM,EM, (wK, wR, wM,wE), beta
        )

        idx = rng.choice(N, p=probs.ravel())
        next_loc = tuple(coords[idx])
        traj.append([next_loc[0], next_loc[1]])

        BM = update_memory(BM, next_loc, decay)
        BR = update_reward(BR, next_loc, sigma)
        current = next_loc

    return traj, K, BR, BM


# =============================================================================
# Behavioral summaries (retained from the original code)
# =============================================================================

def revisit_rate(traj):
    visited = set()
    revisits = 0
    for loc in traj:
        loc = tuple(loc)
        if loc in visited:
            revisits += 1
        visited.add(loc)
    return revisits / len(traj) if traj else np.nan


def first_revisit(traj):
    visited = set()
    for i, loc in enumerate(traj):
        loc = tuple(loc)
        if loc in visited:
            return i
        visited.add(loc)
    return 0


def jump_lengths(traj):
    traj = np.asarray(traj)
    if len(traj) < 2:
        return np.nan, np.nan, np.array([])
    differences = np.abs(np.diff(traj, axis=0))
    total_jumps = differences.sum(axis=1)
    return np.mean(total_jumps), np.var(total_jumps), total_jumps


def calc_total_distance_covered(traj):
    if len(traj) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(traj), axis=0), axis=1).sum())


def behavioral_features(traj):
    mean_jump, _, _ = jump_lengths(traj)
    return {
        "revisit_rate": revisit_rate(traj),
        "jump_length": mean_jump,
        "first_revisit": first_revisit(traj),
        "total_dist": calc_total_distance_covered(traj),
    }


def calc_dist_bw_traj(traj1, traj2):
    traj1 = np.asarray(traj1)
    traj2 = np.asarray(traj2)
    if traj1.shape != traj2.shape:
        raise ValueError("Trajectories must have the same shape.")
    dist = np.linalg.norm(traj1 - traj2, axis=1).sum()
    corr = np.corrcoef(traj1.ravel(), traj2.ravel())[0, 1]
    return dist, corr


def visit_map(traj):
    V = np.zeros((GRID, GRID))
    for r, c in traj:
        V[r, c] += 1
    return V


def count_choices(data):
    return sum(max(0, len(traj) - 1) for traj in data)


# =============================================================================
# Example usage
# =============================================================================
#
# `data` should be a list of trajectories. Each trajectory is a list/array of
# [row, column] coordinates, e.g.:
#
# data = [
#     [[2, 2], [2, 3], [1, 3], [1, 2]],
#     [[0, 0], [1, 0], [1, 1], [2, 2]],
# ]
#
# Fit and compare:
#
# results, comparison, parameters = compare_models(
#     data, seed=1, maxiter=1000, popsize=15
# )
#
# print(comparison)
# print(parameters.T)
# print("Preferred by AIC:", comparison.index[0])
#
# Access fitted parameters:
# print(unpack_result(results["A"], "A"))
# print(unpack_result(results["B"], "B"))
#
# Interpretation:
#   - Lower AIC/BIC indicates better relative fit after penalizing complexity.
#   - Model B has more spatial-mixture parameters (8 instead of 3), so the
#     information criteria penalize that additional flexibility.
#   - The directional weights in Model B sum to one and describe the relative
#     contribution of each of the eight movement directions.
