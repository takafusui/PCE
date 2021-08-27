#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import numpy as np
import chaospy as cp
import numpoly
from sklearn import linear_model

print(np.__version__)  # 1.21.0
print(cp.__version__)  # 4.3.2
print(numpoly.__version__)  # 1.2.2


def ishigami(x, a=7., b=0.1):
    """Ishigami function."""
    _f = np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])
    return _f


N_param = 3  # Number of uncertain parameters
# Joint distribution
dist = cp.Iid(cp.Uniform(-np.pi, np.pi), N_param)

poly_order = 9  # Maximum maybe 10
# polynomials
phi, coeffs = cp.expansion.stieltjes(
    poly_order, dist, normed=True, graded=True, reverse=True, retall=True,
    cross_truncation=1.0)

# --------------------------------------------------------------------------- #
# Collocation method
# --------------------------------------------------------------------------- #
# Number of experimental design points
N_ED = len(phi) * 3  # or * 2

print(r"Number of ED points {}".format(N_ED))

# latin_hypercube, halton
samples = dist.sample(N_ED, 'halton')

evals = np.array([ishigami(sample) for sample in samples.T])

# Return polynomials and Fourier coefficients
M_hat, fourier_coeffs_hat = cp.fit_regression(phi, samples, evals, retall=1)

# Information matrix
Ainfo = phi(*samples).T

# Based on the least square minimization
fourier_coeffs_hat1 = np.linalg.inv(Ainfo.T @ Ainfo) @ Ainfo.T @ evals
M_hat1 = numpoly.sum((phi * fourier_coeffs_hat.T), -1)

# Using Lasso model fit with Least Angle Regression
model = linear_model.LassoLars(alpha=0.1, fit_intercept=False, normalize=False)
model.fit(Ainfo, evals)
fourier_coeffs_hat2 = model.coef_
M_hat2 = cp.sum(phi * fourier_coeffs_hat2, -1)

# --------------------------------------------------------------------------- #
# Compute the leave-one-out error (errLOO)
# errLOO <= 10^{-2}
# --------------------------------------------------------------------------- #
h_diag = np.diagonal(Ainfo @ np.linalg.inv(Ainfo.T @ Ainfo) @ Ainfo.T)

errLOO = 1 / N_ED * np.sum(((evals - M_hat1(*samples)) / (1 - h_diag))**2)

print(r'errLOO error is {}'.format(errLOO))

import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
# Return Sobol' indices
length_ = 50
# --------------------------------------------------------------------------- #
# From operators:
print(r'-' * length_)
print(r'Expectation and variance using the Chaospy built-in functions')
print(r'-' * length_)
print(cp.E(M_hat, dist))
print(cp.Var(M_hat, dist))

# Total variance using the Fourier coefficients
D_hat = np.sum(fourier_coeffs[1:]**2, axis=0)

print(r'-' * length_)
print(r'Expectation and variance using the coefficients')
print(r'-' * length_)
print(fourier_coeffs[0])
print(D_hat)

# Get a multi-index
alpha = cp.glexindex(
    start=0, stop=poly_order+1, dimensions=3, cross_truncation=1.0,
    graded=True, reverse=True).T

# --------------------------------------------------------------------------- #
# Total Sobol' indices based on PCE
# --------------------------------------------------------------------------- #
# Collocation
Sens_t_hat = np.empty(N_param)
for idx in range(N_param):
    index = (alpha[idx, :] > 0)
    Sens_t_hat[idx] = np.sum(fourier_coeffs[index]**2, axis=0) / D_hat

print(r'-' * length_)
print(r'Total Sobol indices')
print(r'-' * length_)
print(Sens_t_hat)
print(r'cp.Sens_t: {}'.format(cp.Sens_t(M_hat, dist)))

# --------------------------------------------------------------------------- #
# First-order Sobol' indices based on PCE
# --------------------------------------------------------------------------- #
# Collocation
Sens_m_hat = np.empty(N_param)
for idx in range(N_param):
    index = (alpha[idx, :] > 0) & (alpha.sum(0) == alpha[idx, :])
    Sens_m_hat[idx] = np.sum(fourier_coeffs[index]**2, axis=0) / D_hat

print(r'-' * length_)
print(r'First-order Sobol indices')
print(r'-' * length_)
print(Sens_m_hat)
print(r'cp.Sens_m: {}'.format(cp.Sens_m(M_hat, dist)))

# --------------------------------------------------------------------------- #
# Second-order Sobol' indices based on PCE
# --------------------------------------------------------------------------- #
# Collocation
Sens_m2_hat = np.empty([N_param, N_param])
for idx in range(N_param):
    for jdx in range(N_param):
        index = (idx != jdx) & (alpha[idx, :] > 0) & (alpha[jdx, :] > 0) \
            & (alpha.sum(0) == alpha[idx, :]+alpha[jdx, :])
        Sens_m2_hat[idx, jdx] = np.sum(
            fourier_coeffs[index]**2, axis=0) / D_hat

print(r'-' * length_)
print(r'Second-order Sobol indices')
print(r'-' * length_)
print(Sens_m2_hat)
print(r'cp.Sens_m2: {}'.format(cp.Sens_m2(M_hat, dist)))

# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
print(r'-' * length_)
print(r'Plot Univariate effects')
print(r'-' * length_)
# --------------------------------------------------------------------------- #
N_linspace = 250
M1_linspace = np.linspace(
    [-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi], N_linspace)

M1 = np.empty([N_param, N_linspace])
for idx in range(N_param):
    theta_idx = np.zeros_like(M1_linspace)
    index = (alpha[idx, :] > 0) & (alpha.sum(0) == alpha[idx, :])
    theta_idx[:, idx] = M1_linspace[:, idx]
    evals = (fourier_coeffs * phi)[index](**{f"q{idx}": M1_linspace[:, idx]})
    # evals = (fourier_coeffs * phi)[index](*theta_idx.T)
    M1[idx, :] = fourier_coeffs[0] + np.sum(evals, axis=0)

# Plot with matplotlib
# mport ipdb; ipdb.set_trace()

import matplotlib.pyplot as plt
idx = 0
plt.plot(M1_linspace[:, idx], M1[idx, :], label=r'$M^1_0$')
idx = 1
plt.plot(M1_linspace[:, idx], M1[idx, :], label=r'$M^1_1$')
idx = 2
plt.plot(M1_linspace[:, idx], M1[idx, :], label=r'$M^1_2$')
plt.xlim([-np.pi, np.pi])
plt.xlabel(r'$\theta_i$')
plt.ylabel(r'$M^1_i$')
plt.legend(loc='lower left')
plt.show()
