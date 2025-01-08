"""
A collection of key functions for the market simulation.

PV, 2024-12-27
"""
import numpy as np


def simulate_market_1d(
    Es,
    Ns,
    Caps,
    dN=0.01,
    n_steps=5000,
    n_sim=1000,
    T=0.1,
    seed=43,
    thermo=100,
    debug=False,
):
    """1d MC simulation of the market, simple cost curve at no transport costs"""
    np.random.seed(seed)
    num_E = len(Es)

    # MC simulation
    n_steps = int(n_steps)
    n_sim = int(n_sim)
    n_eq = n_steps - n_sim  # equilibration steps

    # simulation buffers
    energies = []
    prices = []
    avg_Es = np.zeros_like(Es)
    avg_Ns = np.zeros_like(Ns)

    n_swap = 0  # number of successful swaps

    print("step n_swap E dE P")
    for step in range(n_steps + 1):
        # select proper levels
        while True:
            i = np.random.randint(num_E)
            j = np.random.randint(num_E)
            if i != j and Ns[i] > 0 and Ns[j] < Caps[j]:
                break

        # swap particles
        Ns[i] -= dN
        Ns[j] += dN
        dE = Es[j] - Es[i]

        # perform Metropolis step, return to original config, prevent overflow in exp
        if np.exp(exp_bounding_fun(-dE / T)) <= np.random.rand():
            # reject
            Ns[i] += dN
            Ns[j] -= dN
        else:
            # accept
            n_swap += 1

        # compute observables
        E = Ns @ Es  # total energy
        P = compute_percentile(Ns, Es, 90, dN)
        energies.append(E)
        prices.append(P)
        if step >= n_eq:
            avg_Ns += Ns

        if step % 100 == 0:
            print(step, n_swap, E, dE, P)

    # post-process
    avg_Ns = avg_Ns / n_sim

    return energies, prices, avg_Ns


def simulate_market_2d(
    mat_Es,
    mat_Ns,
    vec_Caps,
    vec_PP_Caps,
    dN=0.01,
    n_steps=5000,
    n_sim=1000,
    T=1.0,
    seed=43,
    thermo=100,
    debug=False,
):
    """Main simulation function for the 2d system"""
    np.random.seed(seed)
    DEBUG = debug
    THERMO = thermo
    num_E, num_pp = mat_Es.shape

    # MC simulation
    n_steps = int(n_steps)
    n_sim = int(n_sim)
    n_eq = n_steps - n_sim  # equilibration steps

    # simulation buffers
    energies = []
    prices = []
    avg_Es = np.zeros_like(mat_Es)
    avg_Ns = np.zeros_like(mat_Ns)

    n_swap = 0  # number of successful swaps
    n_measure = 0  # number of measured steps

    print("step n_swap E dE P")
    for step in range(n_steps + 1):
        # select random mines and power plants
        while True:
            # transfer dN from (i1, j1) to (i2, j2)
            i1 = np.random.randint(num_E)
            j1 = np.random.randint(num_pp)
            i2 = np.random.randint(num_E)
            j2 = np.random.randint(num_pp)
            if selection_rules_2d(i1, j1, i2, j2):
                break

        # swap particles
        mat_Ns[i1, j1] -= dN
        mat_Ns[i2, j2] += dN

        dE = mat_Es[i2, j2] - mat_Es[i1, j1]
        if DEBUG:
            print(f"({i1}, {j1}) to ({i2}, {j2}), dE {dE}")

        # Metropolis step, preventing overflow in exp
        # return to original config,
        if (
            np.exp(exp_bounding_fun(-dE / T)) < np.random.rand()
            or mat_Ns[i1, j1] < 0.0
            or (mat_Ns.sum(axis=1) > vec_Caps).any()
            or (mat_Ns.sum(axis=0) > vec_PP_Caps).any()
        ):
            # reject
            mat_Ns[i1, j1] += dN
            mat_Ns[i2, j2] -= dN
        else:
            # accept
            n_swap += 1

        # compute observables
        E = (mat_Ns * mat_Es).sum()  # total energy
        P = np.zeros(num_pp)
        for j in range(num_pp):
            try:
                P[j] = compute_percentile(mat_Ns[:, j], mat_Es[:, j], 90, dN)
            except:
                if step % THERMO == 0:
                    print(f"error computing percentile for PP {j}")
                P[j] = -1.0
        energies.append(E)
        prices.append(P)

        # compute averages
        if step > n_eq:
            n_measure += 1
            avg_Ns += mat_Ns

        # print
        if step % THERMO == 0:
            print(step, n_swap, E.round(2), dE.round(2), P.round(2))

    # post-process
    avg_Ns = avg_Ns / n_sim

    return energies, prices, avg_Ns


def selection_rules_2d(i1, j1, i2, j2):
    """Rules for indices to swap particles in MC exchange"""
    if i1 == i2 and j1 == j2:
        return False
    else:
        return True


def exp_bounding_fun(x):
    """Prevent overflow in exponential"""
    return min(0, max(-100, x))


def compute_percentile(Ns, Es, perc, dN):
    """compute real percentile with weights, Numpy function only available from Numpy 2.0"""
    buf = []
    for i, _ in enumerate(Ns):
        temp_weight = int(Ns[i] / dN)
        buf.extend([Es[i]] * temp_weight)

    return np.percentile(buf, perc, method="nearest")
