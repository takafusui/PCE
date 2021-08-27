#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: sobol_uqpylab.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:

"""
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

# Computational model
ModelOpts = {'Type': 'Model',
             'ModelFun': 'borehole.model'}

myModel = uq.createModel(ModelOpts)

# Probabilistic input model
InputOpts = {
    'Marginals': [
        {
            'Name': 'rw',  # Radius of the borehole
            'Type': 'Gaussian',
            'Parameters': [0.10, 0.0161812]  # % (m)
        },
        {
            'Name': 'r',  # Radius of influence
            'Type': 'Lognormal',
            'Parameters': [7.71, 1.0056]  # % (m)
        },
        {
            'Name': 'Tu',  # Transmissivity, upper aquifer
            'Type': 'Uniform',
            'Parameters': [63070, 115600]  # % (m^2/yr)
        },
        {
            'Name': 'Hu',  # Potentiometric head, upper aquifer
            'Type': 'Uniform',
            'Parameters': [990, 1110]  # % (m)
        },
        {
            'Name': 'Tl',  # Transmissivity, lower aquifer
            'Type': 'Uniform',
            'Parameters': [63.1, 116]  # % (m^2/yr)
        },
        {
            'Name': 'Hl',  # Potentiometric head , lower aquifer
            'Type': 'Uniform',
            'Parameters': [700, 820]  # % (m)
        },
        {
            'Name': 'L',  # Length of the borehole
            'Type': 'Uniform',
            'Parameters': [1120, 1680]  # % (m)
        },
        {
            'Name': 'Kw',  # Borehole hydraulic conductivity
            'Type': 'Uniform',
            'Parameters': [9855, 12045]  # % (m/yr)
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
    'Order': 1
}

# Sample size for the MC simulation
SobolOpts['Sobol']['SampleSize'] = 1e4  # 1e5

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
    'PCE-based ({} simulations)'.format(myPCE['ExpDesign']['NSamples']),
    # 'LRA-based ({} simulations)'.format(myLRA['ExpDesign']['NSamples']),
]


# Plot the Total Sobol' indices
plt.bar(np.arange(0, len(myInput['Marginals'])) - barWidth,
        mySobolResultsMC['Total'], width=barWidth, color='b', label=labels[0],
        alpha=0.5)
plt.bar(np.arange(0, len(myInput['Marginals'])), mySobolResultsPCE['Total'],
        width=barWidth, color='r', label=labels[1], alpha=0.5,
        tick_label=mySobolAnalysisMC['Results']['VariableNames'])
# plt.bar(np.arange(0,len(myInput['Marginals']))+barWidth, mySobolResultsLRA['Total'], width=barWidth, 
#         color='g', label=labels[2], alpha=0.5)
plt.legend(fontsize=20)
plt.tick_params(axis='both', labelsize=28)
plt.grid(True)
plt.xlabel('Variable name', fontsize=28)
plt.ylabel('Total Sobol\' indices', fontsize=28)
plt.show()


# Plot the first-order Sobol' indices
plt.bar(np.arange(0, len(myInput['Marginals'])) - barWidth,
        mySobolResultsMC['FirstOrder'], width=barWidth, color='b',
        label=labels[0], alpha=0.5)
plt.bar(np.arange(0, len(myInput['Marginals'])),
        mySobolResultsPCE['FirstOrder'], width=barWidth, color='r',
        label=labels[1], alpha=0.5,
        tick_label=mySobolAnalysisMC['Results']['VariableNames'])
# plt.bar(np.arange(0,len(myInput['Marginals']))+barWidth, mySobolResultsLRA['FirstOrder'], width=barWidth, 
#         color='g', label=labels[2], alpha=0.5)
plt.legend(fontsize=20)
plt.tick_params(axis='both', labelsize=28)
plt.grid(True)
plt.xlabel('Variable name', fontsize=28)
plt.ylabel('First-order Sobol\' indices', fontsize=28)
plt.show()
