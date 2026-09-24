
import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import differential_evolution

GRID = 5
N = GRID * GRID

coords = np.array([(r, c) for r in range(GRID) for c in range(GRID)])
coord_to_index = {tuple(c): i for i, c in enumerate(coords)}

dist_matrix = np.zeros((N, N))
for i, (r, c) in enumerate(coords):
    for j, (r1, c1) in enumerate(coords):
        dist_matrix[i, j] = sp.spatial.distance.pdist([(r, c), (r1, c1)])[0]

X, Y = np.meshgrid(np.arange(GRID), np.arange(GRID))


# ---------------------------------------------------------------------
# Spatial kernels
# ---------------------------------------------------------------------

def row_kernel(start_r, start_c):
    K = np.zeros((GRID, GRID))
    for r in range(GRID):
        for c in range(GRID):
            row_cost = abs(r - start_r)
            col_cost = abs(c - start_c)
            K[r, c] = np.exp(-3.0 * row_cost - 0.3 * col_cost)
    K /= K.sum()
    return K

def softmax(x):
    x = x - np.max(x)# No negative exponents
    e = np.exp(x)
    return e / e.sum()


def column_kernel(start_r, start_c):
    K = np.zeros((GRID, GRID))
    for r in range(GRID):
        for c in range(GRID):
            row_cost = abs(r - start_r)
            col_cost = abs(c - start_c)
            K[r, c] = np.exp(-3.0 * col_cost - 0.3 * row_cost)
    K /= K.sum()
    return K


def spiral_kernel(start_r, start_c):
    center = (start_r, start_c)
    K = np.zeros((GRID, GRID))

    for r in range(GRID):
        for c in range(GRID):
            dist2 = (r - center[0])**2 + (c - center[1])**2
            angle = np.arctan2(r - center[1], c - center[0])
            radial = np.sqrt(dist2)
            K[r, c] = np.exp(-radial / 2.0) * (np.cos(2 * angle) + 1.5)

    K -= K.min()
    K /= K.sum()
    return K


def spatial_bias_mixture(start_r, start_c, wRS, wCS, wSS):
    K_row = row_kernel(start_r, start_c)
    K_col = column_kernel(start_r, start_c)
    K_spi = spiral_kernel(start_r, start_c)

    K = wRS * K_row + wCS * K_col + wSS * K_spi
    K /= K.sum()
    return K


def softmax_weights(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------
# State updates
# ---------------------------------------------------------------------

def update_memory(BM, loc, decay):
    BM *= decay
    BM[tuple(loc)] = 1
    return BM


def gaussian_kernel(loc, sigma):
    r, c = loc
    dist2 = (X - c)**2 + (Y - r)**2
    return np.exp(-dist2 / (2 * sigma**2))


def update_reward_A(BR, loc, sigma):
    """
    Model A: original model.
    Reward/search-value kernels accumulate indefinitely.
    The exact rewarded location is set to zero.
    """
    BR += gaussian_kernel(loc, sigma)
    BR[tuple(loc)] = 0
    return BR


def update_reward_B(BR, loc, sigma, reward_decay):
    """
    Model B: Gaussian reward/search-value memory decays over time.
    """
    BR *= reward_decay
    BR += gaussian_kernel(loc, sigma)
    BR[tuple(loc)] = 0
    return BR


def update_reward_C(BR, available_reward, loc, sigma):
    """
    Model C:
      BR = persistent Gaussian reward/search-value memory.
      available_reward = separate binary map indicating which exact
                         slots still contain reward.

    The persistent Gaussian is NOT zeroed at the visited location.
    Depletion is represented separately by available_reward.
    """
    BR += gaussian_kernel(loc, sigma)
    available_reward[tuple(loc)] = 0
    return BR, available_reward


def normalize(u):
    s = np.sum(u)
    if s <= 0:
        return np.ones_like(u) / u.size
    return u / s


def policy_from_current(
    K, BR, BM, current, weights, beta,
    model="A", available_reward=None
):
    """
    A: utility = wK*K + wR*BR - wM*BM

    B: same policy as A, but BR has temporal decay.

    C: utility = wK*K + wR*BR + wA*available_reward - wM*BM
       where available_reward is binary and is NOT normalized.
    """
    if model in ("A", "B"):
        wK, wR, wM = weights
    elif model == "C":
        wK, wR, wA, wM = weights
    else:
        raise ValueError("model must be 'A', 'B', or 'C'")

    K = normalize(K)
    BR = normalize(BR)
    BM = normalize(BM)

    if model == "C":
        if available_reward is None:
            raise ValueError("available_reward is required for Model C")
        # Keep this as a 0/1 state rather than normalizing it.
        availability = available_reward
        utilities = (
            wK * K
            + wR * BR
            + wA * availability
            - wM * BM
        )
    else:
        utilities = wK * K + wR * BR - wM * BM

    return softmax(beta * utilities)


# ---------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------

def neg_log_likelihood(params, data, model="A", sigma=1.0):
    zRS, zCS, zSS = params[:3]
    wRS, wCS, wSS = softmax_weights([zRS, zCS, zSS])

    if model == "A":
        wK, wR, wM, decay, beta = params[3:]
        reward_decay = None

    elif model == "B":
        wK, wR, wM, decay, beta, reward_decay = params[3:]
    elif model == "C":
        wK, wR, wA, wM, decay, beta = params[3:]
    else:
        raise ValueError("model must be 'A', 'B', or 'C'")

    nll = 0.0
    eps = 1e-12

    for traj in data:
        if len(traj) < 2:
            continue

        curr = tuple(traj[0])

        BM = np.zeros((GRID, GRID))
        BR = np.ones((GRID, GRID))

        if model == "C":
            available_reward = np.ones((GRID, GRID))

        BM = update_memory(BM, curr, decay)

        if model == "A":
            BR = update_reward_A(BR, curr, sigma)
        elif model == "B":
            BR = update_reward_B(BR, curr, sigma, reward_decay)
        else:
            BR, available_reward = update_reward_C(
                BR, available_reward, curr, sigma
            )

        for next_loc in traj[1:]:
            next_loc = tuple(next_loc)

            K = spatial_bias_mixture(
                curr[0], curr[1], wRS, wCS, wSS
            )

            if model == "C":
                probs = policy_from_current(
                    K, BR, BM, curr,
                    (wK, wR, wA, wM), beta,
                    model="C",
                    available_reward=available_reward
                )
            else:
                probs = policy_from_current(
                    K, BR, BM, curr,
                    (wK, wR, wM), beta,
                    model=model
                )

            idx = coord_to_index[next_loc]
            nll -= np.log(probs.ravel()[idx] + eps)

            BM = update_memory(BM, next_loc, decay)

            if model == "A":
                BR = update_reward_A(BR, next_loc, sigma)
            elif model == "B":
                BR = update_reward_B(BR, next_loc, sigma, reward_decay)
            else:
                BR, available_reward = update_reward_C(
                    BR, available_reward, next_loc, sigma
                )

            curr = next_loc

    return nll


def get_bounds(model="A"):
    """
    Parameter order:

    A:
      zRS, zCS, zSS, wK, wR, wM, decay, beta

    B:
      zRS, zCS, zSS, wK, wR, wM, decay, beta, reward_decay

    C:
      zRS, zCS, zSS, wK, wR, wA, wM, decay, beta
    """
    spatial_bounds = [(-5, 5), (-5, 5), (-5, 5)]
    utility_bounds = [(0.01, 10), (0.01, 10), (0.01, 10)]

    if model == "A":
        return spatial_bounds + utility_bounds + [(0.2, 0.99), (0.1, 20)]

    if model == "B":
        return (
            spatial_bounds
            + utility_bounds
            + [(0.2, 0.99), (0.1, 20), (0.5, 0.999)]
        )

    if model == "C":
        # wK, wR, wA, wM
        return (
            spatial_bounds
            + [(0.01, 10), (0.01, 10), (0.01, 10), (0.01, 10)]
            + [(0.2, 0.99), (0.1, 20)]
        )

    raise ValueError("model must be 'A', 'B', or 'C'")


def fit_diff_evol(data, model="A", seed=1, tol=1e-3, polish=True):
    bounds = get_bounds(model)

    res = differential_evolution(
        neg_log_likelihood,
        bounds,
        args=(data, model),
        tol=tol,
        polish=polish,
        seed=seed,
        workers=1
    )
    return res


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def simulate_case(
    start_loc=(2, 2),
    T=25,
    sigma=1.0,
    decay=0.9,
    wK=1/3.,
    wR=1/3.,
    wM=1/3.,
    beta=1.0,
    wRS=1/3.,
    wCS=1/3.,
    wSS=1/3.,
    model="A",
    reward_decay=0.95,
    wA=1.0
):
    """
    T = number of transitions after the starting location.

    For Model C, wA controls the separate exact-slot availability term.
    """
    K = spatial_bias_mixture(
        start_loc[0], start_loc[1], wRS, wCS, wSS
    )

    BM = np.zeros((GRID, GRID))
    BR = np.ones((GRID, GRID))

    if model == "C":
        available_reward = np.ones((GRID, GRID))

    current = tuple(start_loc)
    traj = [[current[0], current[1]]]

    BM = update_memory(BM, current, decay)

    if model == "A":
        BR = update_reward_A(BR, current, sigma)
    elif model == "B":
        BR = update_reward_B(BR, current, sigma, reward_decay)
    elif model == "C":
        BR, available_reward = update_reward_C(
            BR, available_reward, current, sigma
        )
    else:
        raise ValueError("model must be 'A', 'B', or 'C'")

    for _ in range(T):
        K = spatial_bias_mixture(
            current[0], current[1], wRS, wCS, wSS
        )

        if model == "C":
            weights = (wK, wR, wA, wM)
            probs = policy_from_current(
                K, BR, BM, current, weights, beta,
                model="C",
                available_reward=available_reward
            )
        else:
            weights = (wK, wR, wM)
            probs = policy_from_current(
                K, BR, BM, current, weights, beta,
                model=model
            )

        idx = np.random.choice(N, p=probs.ravel())
        next_loc = tuple(coords[idx])

        traj.append([next_loc[0], next_loc[1]])

        BM = update_memory(BM, next_loc, decay)

        if model == "A":
            BR = update_reward_A(BR, next_loc, sigma)
        elif model == "B":
            BR = update_reward_B(BR, next_loc, sigma, reward_decay)
        else:
            BR, available_reward = update_reward_C(
                BR, available_reward, next_loc, sigma
            )

        current = next_loc

    if model == "C":
        return traj, K, BR, BM, available_reward

    return traj, K, BR, BM


# ---------------------------------------------------------------------
# Behavioral summaries
# ---------------------------------------------------------------------

def revisit_rate(traj):
    visited = set()
    revisits = 0

    for loc in traj:
        loc = tuple(loc)
        if loc in visited:
            revisits += 1
        visited.add(loc)

    return revisits / len(traj)


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
    d = np.abs(np.diff(traj, axis=0))
    total_jumps = d.sum(axis=1)
    return np.mean(total_jumps), np.var(total_jumps), total_jumps


def calc_total_distance_covered(traj):
    tot_dist = 0.0
    for i in range(len(traj) - 1):
        tot_dist += sp.spatial.distance.pdist(
            [traj[i], traj[i + 1]]
        )[0]
    return tot_dist


def behavioral_features(traj):
    mean_jump, _, _ = jump_lengths(traj)
    return {
        "revisit_rate": revisit_rate(traj),
        "jump_length": mean_jump,
        "first_revisit": first_revisit(traj),
        "total_dist": calc_total_distance_covered(traj),
    }


def unpack_result(res, model):
    zRS, zCS, zSS = res.x[:3]
    wRS, wCS, wSS = softmax_weights([zRS, zCS, zSS])

    if model in ("A", "B"):
        wK, wR, wM = res.x[3:6]
        out = {
            "W(row_scan)": wRS,
            "W(column_scan)": wCS,
            "W(spiral_scan)": wSS,
            "W(spatial_bias)": wK,
            "W(reward)": wR,
            "W(memory)": wM,
            "memory_recall": res.x[6],
            "beta": res.x[7],
        }
        if model == "B":
            out["reward_decay"] = res.x[8]
        return out

    wK, wR, wA, wM = res.x[3:7]
    return {
        "W(row_scan)": wRS,
        "W(column_scan)": wCS,
        "W(spiral_scan)": wSS,
        "W(spatial_bias)": wK,
        "W(reward)": wR,
        "W(reward_availability)": wA,
        "W(memory)": wM,
        "memory_recall": res.x[7],
        "beta": res.x[8],
    }


def count_choices(data):
#     return sum(max(0, len(traj) - 1) for traj in data)
    return sum([len(traj)-1 for traj in data])
