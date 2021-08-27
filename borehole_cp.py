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
import numpoly
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
    rw = x[0]
    r = x[1]
    Tu = x[2]
    Hu = x[3]
    Tl = x[4]
    Hl = x[5]
    L = x[6]
    Kw = x[7]

    r_rw_log = np.log(r / rw)
    numerator = 2 * np.pi * Tu * (Hu - Hl)
    denominator = r_rw_log * (
        1 + (2 * L * Tu) / (r_rw_log * rw**2 * Kw) + (Tu / Tl))

    return numerator / denominator


# rw_dist = cp.Normal(0.10, 0.0161812)
rw_dist = cp.Uniform(0.05, 0.15)
# r_dist = cp.LogNormal(7.71, 1.0056)
r_dist = cp.Uniform(100, 50000)
Tu_dist = cp.Uniform(63070, 115600)
Hu_dist = cp.Uniform(990, 1110)
Tl_dist = cp.Uniform(63.1, 116)
Hl_dist = cp.Uniform(700, 820)
L_dist = cp.Uniform(1120, 1680)
Kw_dist = cp.Uniform(9855, 12045)

# import ipdb; ipdb.set_trace()
joint_dist = cp.J(
    rw_dist, r_dist, Tu_dist, Hu_dist, Tl_dist, Hl_dist, L_dist, Kw_dist)

poly_order = 4

phi, coeffs = cp.expansion.stieltjes(
    poly_order, joint_dist, normed=True, graded=True, reverse=True,
    retall=True, cross_truncation=1.0)

N_EDpoints = 3 * len(phi)

EDpoints = joint_dist.sample(N_EDpoints, 'latin_hypercube')

evals = np.array([borehole(x) for x in EDpoints.T])

M_hat, fourier_coeffs = cp.fit_regression(phi, EDpoints, evals, retall=1)

# Total variance using the Fourier coefficients
D_hat = np.sum(fourier_coeffs[1:]**2, axis=0)

# Information matrix
Ainfo = phi(*EDpoints).T

# Based on the least square minimization
fourier_coeffs_hat = np.linalg.inv(Ainfo.T @ Ainfo) @ Ainfo.T @ evals
M_hathat = numpoly.sum((phi * fourier_coeffs_hat.T), -1)

# --------------------------------------------------------------------------- #
# Compute the leave-one-out error (errLOO)
# errLOO <= 10^{-2}
# --------------------------------------------------------------------------- #
h_diag = np.diagonal(Ainfo @ np.linalg.inv(Ainfo.T @ Ainfo) @ Ainfo.T)

errLOO = 1 / N_EDpoints * np.sum(
    ((evals - M_hathat(*EDpoints)) / (1 - h_diag))**2)

print(r'errLOO error is {}'.format(errLOO))

import ipdb; ipdb.set_trace()
N_param = EDpoints.shape[0]

# Get a multi-index
alpha = cp.glexindex(
    start=0, stop=poly_order+1, dimensions=N_param, cross_truncation=1.0,
    graded=True, reverse=True).T


Sens_t_hat = np.empty(N_param)
for idx in range(N_param):
    index = (alpha[idx, :] > 0)
    Sens_t_hat[idx] = np.sum(fourier_coeffs[index]**2, axis=0) / D_hat

Sens_m_hat = np.empty(N_param)
for idx in range(N_param):
    index = (alpha[idx, :] > 0) & (alpha.sum(0) == alpha[idx, :])
    Sens_m_hat[idx] = np.sum(fourier_coeffs[index]**2, axis=0) / D_hat

xlabels = [r'$r_{w}$', r'$r$', r'$T_{u}$', r'$H_{u}$', r'$T_{l}$', r'$H_{l}$',
           r'$L$', r'$K_{w}$']

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(xlabels, Sens_t_hat, color='royalblue', edgecolor='black')
plt.ylim([0, None])
plt.ylabel(r"$S^{T}_{i}$")
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(xlabels, Sens_m_hat, color='royalblue', edgecolor='black')
plt.ylim([0, None])
plt.ylabel(r"$S_{i}$")
plt.show()

import ipdb; ipdb.set_trace()
