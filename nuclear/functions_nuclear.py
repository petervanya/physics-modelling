"""
A collection of functions for nuclear simulations

PV
Created: 2024-06-07
"""
import numpy as np
import jax
import jax.numpy as jnp


EPS_STRONG = 3.5
ALPHA_STRONG = 8.0


def elmag_potential(r, t1, t2):
    """In MeV, r in fm"""
    return 1.44 * t1 * t2 / r

def lj_potential(r, epsilon=EPS_STRONG, sigma=1/2**(1/6)):
    """In MeV, r in fm"""
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

def morse_potential(r, D=EPS_STRONG, alpha=ALPHA_STRONG, re=1.0):
    """In MeV, r in fm"""
    return D * ((1.0 - np.exp(-alpha * (r - re))) ** 2 - 1.0)


"""
Energy functions
"""
def kinetic_energy(V, M):
    return 0.5 * M @ (V * V).sum(axis=1)

def total_energy(R, T, pot='lj', mode='bh'):
    if mode == 'bh':
        R = R.reshape(-1, 3)
    Ve, Vs= 0.0, 0.0
    for i, _ in enumerate(T):
        for j in range(i):
            r = np.linalg.norm(R[i] - R[j])
            Ve += elmag_potential(r, T[i], T[j])
            if pot == 'lj':
                Vs += lj_potential(r)
            elif pot == 'morse':
                Vs += morse_potential(r)
    V = Ve + Vs
    return V, Ve, Vs

def total_energy_min(R, T, pot='lj', mode='bh'):
    '''Same as total energy, only returning one value for minimisation'''
    if mode == 'bh':
        R = R.reshape(-1, 3)
    Ve, Vs= 0.0, 0.0
    for i, _ in enumerate(T):
        for j in range(i):
            r = np.linalg.norm(R[i] - R[j])
            Ve += elmag_potential(r, T[i], T[j])
            if pot == 'lj':
                Vs += lj_potential(r)
            elif pot == 'morse':
                Vs += morse_potential(r)
    V = Ve + Vs
    return V

def total_energy_jnp(R, T, pot='lj'):
    Ve, Vs = 0.0, 0.0
    for i, _ in enumerate(T):
        for j in range(i):
            r = jnp.linalg.norm(R[i] - R[j])
            Ve += elmag_potential(r, T[i], T[j])
            if pot == 'lj':
                Vs += lj_potential(r)
            elif pot == 'morse':
                Vs += morse_potential(r)
    V = Ve + Vs
    return V, Ve, Vs

def minimize_energy(R, T, n_steps=10, lr=1e-5, verbose=False, thermo=10, pot='lj'):
    """Gradient descent for minimisation using jax"""
    for i in range(n_steps):
        R -= lr * jax.grad(total_energy_jnp)(R, T)
        if verbose and i % thermo == 0:
            print(i, total_energy_jnp(R, T, pot=pot))[0]
    return R


"""
Helper functions
"""
def distance_vector(X):
    N = X.shape[0]
    D = []
    for i in range(N):
        for j in range(i):
            D.append(np.linalg.norm(X[i] - X[j]))
    return np.array(sorted(D))


def generate_random_labels(Z, N):
    T = np.array([1] * Z + [0] * N)
    np.random.shuffle(T)
    return T