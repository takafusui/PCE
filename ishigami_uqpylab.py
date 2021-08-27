#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from uqpylab import sessions
import numpy as np
import matplotlib.pyplot as plt

# The user's token to access the UQCloud API
myToken = '217ac4e1da9bbadecdd5c9f326637f6fdce1f272'
# The UQCloud instance to use
UQCloud_instance = 'https://beta.uq-cloud.io'
plt.rcParams["figure.figsize"] = (10, 10)

# Start the session
mySession = sessions.cloud(host=UQCloud_instance, token=myToken)
# (Optional) Get a convenient handle to the command line interface
uq = mySession.cli
# Reset the session
mySession.reset()

# Set the random seed for reproducibility
uq.rng(100, 'twister')


def ishigami(x, a=7., b=0.1):
    """Ishigami function."""
    x = np.array(x, ndmin=2)
    _f = np.sin(x[:, 0]) + a * np.sin(x[:, 1])**2 \
        + b * x[:, 2]**4 * np.sin(x[:, 0])
    return _f


# Computational model
ModelOpts = {'Type': 'Model',
             'ModelFun': 'ishigami_uqpylab.ishigami'}

myModel = uq.createModel(ModelOpts)

# Probabilistic input model
InputOpts = {
    'Marginals': [
        {
            'Name': 'x0',
            'Type': 'Uniform',
            'Parameters': [-np.pi, np.pi]
        },
        {
            'Name': 'x1',
            'Type': 'Uniform',
            'Parameters': [-np.pi, np.pi]
        },
        {
            'Name': 'x2',
            'Type': 'Uniform',
            'Parameters': [-np.pi, np.pi]
        },
    ]
}

myInput = uq.createInput(InputOpts)

# --------------------------------------------------------------------------- #
# MC-based Sobol' indices
# --------------------------------------------------------------------------- #
SobolOpts = {
    'Type': 'Sensitivity',
    'Method': 'Sobol'
}

# The maximum order of the Sobol' indices
SobolOpts['Sobol'] = {
    'Order': 2
}

# Sample size for the MC simulation
SobolOpts['Sobol']['SampleSize'] = 1e5  # 1e4

# Run the MC based sensitivity analysis
mySobolAnalysisMC = uq.createAnalysis(SobolOpts)
# Retrieve the MC sensitivity results
mySobolResultsMC = mySobolAnalysisMC['Results']

# --------------------------------------------------------------------------- #
# PCE-based Sobol' indices
# --------------------------------------------------------------------------- #
PCEOpts = {
    'Type': 'Metamodel',
    'MetaType': 'PCE',
    'Method': 'LARS'
}

# Assign the full computational model
PCEOpts['FullModel'] = myModel['Name']

# Maximum polynomial degree (sparse PCE)
PCEOpts['Degree'] = 5

# Size of the experimental design
PCEOpts['ExpDesign'] = {
    'NSamples': 200
}

# Calculate the PCE
myPCE = uq.createModel(PCEOpts)

mySobolAnalysisPCE = uq.createAnalysis(SobolOpts)
# Retrieve the results for comparison
mySobolResultsPCE = mySobolAnalysisPCE['Results']

# --------------------------------------------------------------------------- #
# Print the sensitivity results
# --------------------------------------------------------------------------- #
uq.print(mySobolAnalysisMC)
uq.print(mySobolAnalysisPCE)

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
# setup plot
barWidth = 0.2
labels = [
    'MC-based ({:.2E} simulations)'.format(mySobolResultsMC['Cost']),
    'PCE-based ({} simulations)'.format(myPCE['ExpDesign']['NSamples'])
]

# Plot the Total Sobol' indices
plt.bar(np.arange(0, len(myInput['Marginals'])), mySobolResultsMC['Total'],
        align='edge', width=-barWidth, color='b', label=labels[0], alpha=0.5)
plt.bar(np.arange(0, len(myInput['Marginals'])), mySobolResultsPCE['Total'],
        align='edge', width=barWidth, color='r', label=labels[1],
        alpha=0.5, tick_label=mySobolAnalysisMC['Results']['VariableNames']
        )
plt.legend(fontsize=20)
plt.tick_params(axis='both', labelsize=28)
plt.grid(True)
plt.xlabel('Variable name', fontsize=28)
plt.ylabel('Total Sobol\' indices', fontsize=28)
plt.show()


# Plot the first-order Sobol' indices
plt.bar(
    np.arange(0, len(myInput['Marginals'])), mySobolResultsMC['FirstOrder'],
    align='edge', width=-barWidth, color='b', label=labels[0], alpha=0.5)
plt.bar(
    np.arange(0, len(myInput['Marginals'])), mySobolResultsPCE['FirstOrder'],
    align='edge', width=barWidth, color='r', label=labels[1], alpha=0.5,
    tick_label=mySobolAnalysisMC['Results']['VariableNames'])
plt.legend(fontsize=20)
plt.tick_params(axis='both', labelsize=28)
plt.grid(True)
plt.xlabel('Variable name', fontsize=28)
plt.ylabel('First-order Sobol\' indices', fontsize=28)
plt.show()

import ipdb; ipdb.set_trace()
