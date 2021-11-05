# ==========================================
# Title:  Symulacja dynamiki molekularnej
# Author: 01141448
# Date:   04 Nov 2021
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
from tqdm import tqdm

# http://omega.if.pw.edu.pl/~mars/kms/

kB = 8.31 * 1e-3  # kJ / (mol * K)


def initialize_values():
    params = {}
    with open(os.path.join("data", "parameters.txt"), 'r') as f:
        lines = f.readlines()

    for line in lines:
        param_name, val = line.replace('\n', '').split('\t')
        params[param_name] = float(val)

    return params


def calculate_crystal_edges(n, a):
    N = n ** 3
    b_0 = (a, 0, 0)
    b_1 = (a / 2, a * 3 ** (1 / 2) / 2, 0)
    b_2 = (a / 2, a * 3 ** (1 / 2) / 6, a * (2 / 3) ** (1 / 2))

    return N, b_0, b_1, b_2


def calculate_energy(N, T_0):
    E = -0.5 * kB * T_0 * np.log(np.random.random((N, 3)))

    print("Mean E: {:.2e}, 1/2kT_0 = {:.2e} [kJ / mol]"
          .format(np.mean(np.mean(E)),
                  0.5 * kB * T_0))
    return E


def calculate_initial_positions(n, b_0, b_1, b_2):
    r_0 = np.zeros(shape=(n ** 3, 3))

    for i in range(0, n):  # i0
        for j in range(0, n):  # i1
            for k in range(0, n):  # i2
                iX = i + j * n + k * n ** 2
                r_0[iX, 0] = (i - (n - 1) / 2) * b_0[0] + \
                             (j - (n - 1) / 2) * b_1[0] + \
                             (k - (n - 1) / 2) * b_2[0]

                r_0[iX, 1] = (i - (n - 1) / 2) * b_0[1] + \
                             (j - (n - 1) / 2) * b_1[1] + \
                             (k - (n - 1) / 2) * b_2[1]

                r_0[iX, 2] = (i - (n - 1) / 2) * b_0[2] + \
                             (j - (n - 1) / 2) * b_1[2] + \
                             (k - (n - 1) / 2) * b_2[2]

    return r_0


def calculate_initial_momenta(N, m, E):
    random_signs = np.random.choice([-1, 1], size=(N, 3))
    p = random_signs * np.sqrt(2 * m * E)
    p_prim = p - 1 / N * np.mean(p, axis=0)

    return p_prim


@jit
def calculate_potential_forces_and_pressure(N, R, r, epsilon, L, f):
    V_P = np.zeros(shape=(N, N))
    V_S = np.zeros(shape=N)

    F_P = np.zeros(shape=(N, 3))
    F_S = np.zeros(shape=(N, 3))

    for i in range(N):
        r_i = np.linalg.norm(r[i, :])

        if r_i >= L:
            V_S[i] = 0.5 * f * (r_i - L) ** 2
            if r_i != 0:
                F_S[i] = f * (L - r_i) * r[i] / r_i

        for j in range(i):
            r_ij = np.linalg.norm(r[i, :] - r[j, :])

            if r_ij != 0:
                V_P[i, j] = epsilon * ((R / r_ij) ** 12 - 2 * (R / r_ij) ** 6)

                temp_var_for_forces = 12 * epsilon * ((R / r_ij) ** 12 - (R / r_ij) ** 6) * (r[i] - r[j]) / r_ij ** 2
                F_P[i] += temp_var_for_forces
                F_P[j] -= temp_var_for_forces

    V = np.sum(V_P) + np.sum(V_S)
    F = F_P + F_S

    # pressure calculation
    abs_F_S = np.sqrt(F_S[:, 0] ** 2 + F_S[:, 1] ** 2 + F_S[:, 2] ** 2)
    P = 1 / (4 * np.pi * L ** 2) * np.sum(abs_F_S)
    # P = calculate_pressure(L, F_S, N)

    return V, F, P


def calculate_pressure(L, F_S):
    abs_F_S = np.sqrt(F_S[:, 0] ** 2 + F_S[:, 1] ** 2 + F_S[:, 2] ** 2)
    return 1 / (4 * np.pi * L ** 2) * np.sum(abs_F_S)


def calculate_temperature(N, p, m):
    abs_p = np.linalg.norm(p, axis=1)
    return 2 / (3 * N * kB) * np.sum(abs_p ** 2 / (2 * m))


def calculate_total_energy(p, m, V):
    abs_p = np.linalg.norm(p, axis=1)
    return np.sum(abs_p ** 2 / (2 * m)) + V


def write_initial_positions_to_file(r_0):
    with open(os.path.join("data", "initial_state.dat"), 'w') as f:
        for i in range(r_0.shape[0]):
            f.write("{} {} {}\n".format(r_0[i, 0], r_0[i, 1], r_0[i, 2]))


def write_atom_params_to_file(r, p, m):
    with open(os.path.join("data","atom-positions.txt"), 'a') as f:
        for i in range(r.shape[0]):
            E = np.linalg.norm(p[i, :]) ** 2 / (2 * m)
            f.write("{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\n".format(r[i, 0], r[i, 1], r[i, 2], E))


def write_params_to_file(t, H, V, T, P):
    with open(os.path.join("data", "temp-values.txt"), 'a') as f:
        f.write("{:.3e}\t{:.3e}\t{:.6e}\t{:.3e}\t{:.3e}\n".format(t, H, V, T, P))


def main():
    params = initialize_values()

    n = int(params.get('n'))
    m = int(params.get('m'))  # u
    epsilon = int(params.get('epsilon'))  # kJ / mol
    R = params.get('R')  # nm
    f = int(params.get('f'))
    L = int(params.get('L'))  # nm
    a = params.get('a')  # nm
    T_0 = int(params.get('T_0'))  # K
    tau = params.get('tau')
    S_0 = int(params.get('S_0'))
    S_d = int(params.get('S_d'))
    S_out = int(params.get('S_out'))
    S_xyz = int(params.get('S_xyz'))

    N, b_0, b_1, b_2 = calculate_crystal_edges(n, a)
    E = calculate_energy(N, T_0)
    r_0 = calculate_initial_positions(n, b_0, b_1, b_2)
    p = calculate_initial_momenta(N, m, E)

    write_initial_positions_to_file(r_0)

    V, F, P = calculate_potential_forces_and_pressure(N, R, r_0, epsilon, L, f)

    r = r_0
    t = 0
    T_sum, P_sum, H_sum = 0, 0, 0

    with open(os.path.join("data", "temp-values.txt"), 'a') as file:
        file.write("t\tH\tV\tT\tP\n")

    for s in tqdm(range(S_0 + S_d)):
        # for s in range(3):
        t += tau
        p += 0.5 * F * tau
        r += 1 / m * p * tau
        V, F, P = calculate_potential_forces_and_pressure(N, R, r_0, epsilon, L, f)
        p += 0.5 * F * tau

        T = calculate_temperature(N, p, m)
        H = calculate_total_energy(p, m, V)

        if s % S_out == 0:
            write_params_to_file(t, H, V, T, P)

        if s % S_xyz == 0:
            write_atom_params_to_file(r, p, m)

        if s >= S_0:
            T_sum += T
            P_sum += P
            H_sum += H

    T_mean = 1 / S_d * T_sum
    P_mean = 1 / S_d * P_sum
    H_mean = 1 / S_d * H_sum

    print("T_mean = {:.2e}, P_mean = {:.2e}, H_mean = {:.2e}".format(T_mean, P_mean, H_mean))


# # Uncomment below for plot of initial positions
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(r_0[:,0], r_0[:,1], r_0[:,2])
# plt.show()

# # Uncomment below for plot of initial forces
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(r_0[:,0], r_0[:,1], r_0[:,2])
# ax.quiver(r_0[:,0], r_0[:,1], r_0[:,2], F[:,0], F[:,1], F[:,2], length=0.3, normalize=True)
# plt.show()


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    atom_positions_file = os.path.join("data", "atom-positions.txt")
    temp_values_file = os.path.join("data", "temp-values.txt")

    if os.path.exists(atom_positions_file):
        os.remove(atom_positions_file)
    if os.path.exists(temp_values_file):
        os.remove(temp_values_file)
    main()
