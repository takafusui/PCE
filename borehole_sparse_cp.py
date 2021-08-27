#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: borehole_cp.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:

"""
import numpy as np
import chaospy as cp
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import rc

# Use TeX font
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
# Font size
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.title_fontsize"] = 14


def borehole(x):
    """Borehore function."""
    rw = x[:, 0]
    r = x[:, 1]
    Tu = x[:, 2]
    Hu = x[:, 3]
    Tl = x[:, 4]
    Hl = x[:, 5]
    L = x[:, 6]
    Kw = x[:, 7]

    r_rw_log = np.log(r / rw)
    numerator = 2 * np.pi * Tu * (Hu - Hl)
    denominator = r_rw_log * (
        1 + (2 * L * Tu) / (r_rw_log * rw**2 * Kw) + (Tu / Tl))

    return numerator / denominator


rw_dist = cp.Normal(0.10, 0.0161812)
# rw_dist = cp.Uniform(0.05, 0.15)
r_dist = cp.LogNormal(7.71, 1.0056)
# r_dist = cp.Uniform(100, 50000)
Tu_dist = cp.Uniform(63070, 115600)
Hu_dist = cp.Uniform(990, 1110)
Tl_dist = cp.Uniform(63.1, 116)
Hl_dist = cp.Uniform(700, 820)
L_dist = cp.Uniform(1120, 1680)
Kw_dist = cp.Uniform(9855, 12045)

# import ipdb; ipdb.set_trace()
joint_dist = cp.J(
    rw_dist, r_dist, Tu_dist, Hu_dist, Tl_dist, Hl_dist, L_dist, Kw_dist)

poly_order = 5

phi, coeffs = cp.expansion.stieltjes(
    poly_order, joint_dist, normed=True, graded=True, reverse=True,
    retall=True, cross_truncation=1.0)

# Number of experimental design
N_EDpoints = 300  # 3 * len(phi)
# Experimental design points
EDpoints = joint_dist.sample(N_EDpoints, 'latin_hypercube')

evals = np.array(borehole(EDpoints.T))

# Ordinary information matrix
poly_evals = phi(*EDpoints).T

# Using Lasso model fit with Least Angle Regression
reg = linear_model.LassoLars(alpha=0.01, fit_intercept=False, normalize=False)
reg.fit(poly_evals, evals)
phi_active = phi[reg.active_]

# Sparse information matrix
Pinfo_sparse = phi_active(*EDpoints).T

fourier_coeffs_sparse = np.linalg.inv(Pinfo_sparse.T @ Pinfo_sparse) \
    @ Pinfo_sparse.T @ evals

M_sparse = cp.sum(phi_active * fourier_coeffs_sparse, -1)

# --------------------------------------------------------------------------- #
# Compute the leave-one-out error (errLOO)
# errLOO <= 10^{-2}
# --------------------------------------------------------------------------- #
h_diag_sparse = np.diagonal(
    Pinfo_sparse @ np.linalg.inv(Pinfo_sparse.T @ Pinfo_sparse)
    @ Pinfo_sparse.T)

errLOO = 1 / N_EDpoints * np.sum(
    ((evals - M_sparse(*EDpoints)) / (1 - h_diag_sparse))**2)

print(r'errLOO error is {}'.format(errLOO))
# import ipdb; ipdb.set_trace()

# --------------------------------------------------------------------------- #
# Estimate the total and the first-order Sobol indices based on the fourier
# coefficients
# --------------------------------------------------------------------------- #
N_param = EDpoints.shape[0]  # Number of uncertain parameters

# Coefficients but filled with zero if it is sparse
fourier_coeffs_hat = np.zeros(len(phi))
fourier_coeffs_hat[reg.active_] = fourier_coeffs_sparse

# Total variance using the Fourier coefficients
D_hat = np.sum(fourier_coeffs_hat[1:]**2, axis=0)

# Get a multi-index
alpha = cp.glexindex(
    start=0, stop=poly_order+1, dimensions=N_param, cross_truncation=1.0,
    graded=True, reverse=True).T

# Total Sobol' indices
Sens_t_hat = np.empty(N_param)
for idx in range(N_param):
    index = (alpha[idx, :] > 0)
    Sens_t_hat[idx] = np.sum(fourier_coeffs_hat[index]**2, axis=0) / D_hat

# First-order Sobol' indices
Sens_m_hat = np.empty(N_param)
for idx in range(N_param):
    index = (alpha[idx, :] > 0) & (alpha.sum(0) == alpha[idx, :])
    Sens_m_hat[idx] = np.sum(fourier_coeffs_hat[index]**2, axis=0) / D_hat

# --------------------------------------------------------------------------- #
# Plot the total and the first-order Sobol' indices
# --------------------------------------------------------------------------- #
xlabels = [r'$r_{w}$', r'$r$', r'$T_{u}$', r'$H_{u}$', r'$T_{l}$', r'$H_{l}$',
           r'$L$', r'$K_{w}$']

# Total Sobol' indices
fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(xlabels, Sens_t_hat, color='royalblue', edgecolor='black')
plt.ylim([0, None])
plt.ylabel(r"$S^{T}_{i}$")
plt.show()

# First-order Sobol' indices
fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(xlabels, Sens_m_hat, color='royalblue', edgecolor='black')
plt.ylim([0, None])
plt.ylabel(r"$S_{i}$")
plt.show()
