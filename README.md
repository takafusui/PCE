# Uncertainty Quantification based on Polynomial Chaos Expansion

Polynomial Chaos Expansion (PCE) is one of the most popular surrogate modeling algorithms, especially for global sensitivity analysis. Using PCE, the first-order and the total Sobol' indices and univariate effects are obtained analytically. [Harenberg et al. (2019)](https://doi.org/10.3982/QE866) is the first example in economics to present global sensitivity analysis based on PCE.

We study PCE using the `chaospy` library and apply the technique to some test functions. In `python`, there are some implementations of the PCE-based global sensitivity analysis, such as [`uqpylab`](https://uqpylab.uq-cloud.io/). We verify our implementations by comparing them with the results from `uqpylab`.