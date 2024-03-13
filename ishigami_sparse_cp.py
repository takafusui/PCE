#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import numpy as np
import chaospy as cp
from sklearn import linear_model

print(np.__version__)  # 1.21.0
print(cp.__version__)  # 4.3.2


def ishigami(x, a=7., b=0.1):
    """Ishigami function."""
    _f = np.sin(x[0, :]) + a * np.sin(x[1, :])**2 \
        + b * x[2, :]**4 * np.sin(x[0, :])
    return _f


N_param = 3  # Number of uncertain parameters

# Joint distribution
dist = cp.Iid(cp.Uniform(-np.pi, np.pi), N_param)

poly_order = 12  # Maximum maybe 20

# Ordinary polynomials
phi, coeffs = cp.expansion.stieltjes(
    poly_order, dist, normed=True, graded=True, reverse=True, retall=True,
    cross_truncation=1.0)

# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
# Sparse PCE
# --------------------------------------------------------------------------- #
# Number of experimental design points
N_ED = 200  # Can be smaller that 2-3 * len(phi)

print(r"Number of ED points {}".format(N_ED))

# Experimental design points drawing from latin_hypercube, halton etc.
samples = dist.sample(N_ED, 'latin_hypercube')

evals = np.array(ishigami(samples))

# Ordinary information matrix
poly_evals = phi(*samples).T

# Using Lasso model fit with Least Angle Regression
reg = linear_model.LassoLars(alpha=0.01, fit_intercept=False, normalize=False)
reg.fit(poly_evals, evals)
phi_active = phi[reg.active_]

# Sparse information matrix
Pinfo_sparse = phi_active(*samples).T

fourier_coeffs_sparse = np.linalg.inv(Pinfo_sparse.T @ Pinfo_sparse) \
    @ Pinfo_sparse.T @ evals

# Sparse PCE based surrogate model
M_sparse = cp.sum(phi_active * fourier_coeffs_sparse, -1)

# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
# Compute the leave-one-out error (errLOO)
# errLOO <= 10^{-2}
# --------------------------------------------------------------------------- #
h_diag_sparse = np.diagonal(
    Pinfo_sparse @ np.linalg.inv(Pinfo_sparse.T @ Pinfo_sparse)
    @ Pinfo_sparse.T)

# Compute the leave-one-out error
errLOO = 1 / N_ED * np.sum(
    ((evals - M_sparse(*samples)) / (1 - h_diag_sparse))**2)

print(r'errLOO error is {}'.format(errLOO))
# import ipdb; ipdb.set_trace()

# --------------------------------------------------------------------------- #
# Estimate the total and the first-order Sobol indices based on the fourier
# coefficients
# --------------------------------------------------------------------------- #
N_param = samples.shape[0]  # Number of uncertain parameters

# Coefficients but filled with zero if it is sparse
fourier_coeffs_hat = np.zeros(len(phi))
fourier_coeffs_hat[reg.active_] = fourier_coeffs_sparse

# Total variance using the Fourier coefficients
D_hat = np.sum(fourier_coeffs_hat[1:]**2, axis=0)

# Get a multi-index
alpha = cp.glexindex(
    start=0, stop=poly_order+1, dimensions=3, cross_truncation=1.0,
    graded=True, reverse=True).T

# --------------------------------------------------------------------------- #
# Return Sobol' indices
length_ = 50
# --------------------------------------------------------------------------- #
# Total Sobol' indices based on sparse PCE
Sens_t_hat = np.empty(N_param)
for idx in range(N_param):
    index = (alpha[idx, :] > 0)
    Sens_t_hat[idx] = np.sum(fourier_coeffs_hat[index]**2, axis=0) / D_hat

print(r'-' * length_)
print(r'Total Sobol indices')
print(r'-' * length_)
print(Sens_t_hat)

# First-order Sobol' indices based on sparse PCE
Sens_m_hat = np.empty(N_param)
for idx in range(N_param):
    index = (alpha[idx, :] > 0) & (alpha.sum(0) == alpha[idx, :])
    Sens_m_hat[idx] = np.sum(fourier_coeffs_hat[index]**2, axis=0) / D_hat

print(r'-' * length_)
print(r'First-order Sobol indices')
print(r'-' * length_)
print(Sens_m_hat)

# Second-order Sobol' indices based on PCE
Sens_m2_hat = np.empty([N_param, N_param])
for idx in range(N_param):
    for jdx in range(N_param):
        index = (idx != jdx) & (alpha[idx, :] > 0) & (alpha[jdx, :] > 0) \
            & (alpha.sum(0) == alpha[idx, :]+alpha[jdx, :])
        Sens_m2_hat[idx, jdx] = np.sum(
            fourier_coeffs_hat[index]**2, axis=0) / D_hat

print(r'-' * length_)
print(r'Second-order Sobol indices')
print(r'-' * length_)
print(Sens_m2_hat)

import ipdb; ipdb.set_trace()
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
    # If Sens_m_hat[idx]=0, univariate effects are zero
    if Sens_m_hat[idx] != 0:
        theta_idx = np.zeros_like(M1_linspace)
        index = (alpha[idx, :] > 0) & (alpha.sum(0) == alpha[idx, :])
        theta_idx[:, idx] = M1_linspace[:, idx]
        evals = (fourier_coeffs_hat * phi)[index](
            **{f"q{idx}": theta_idx[:, idx]})
    else:  # Univariate effects are zero
        evals = np.zeros_like(M1_linspace)
    M1[idx, :] = fourier_coeffs_hat[0] + np.sum(evals, axis=0)


# Plot with matplotlib
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
plt.close()
