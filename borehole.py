import numpy as np

# Y = uq_borehole(X) returns the value of the water flow Y through a borehole, described by 8 variables given in X = [rw, r, Tu, Hu, Tl, Hl, L, Kw]

# rw - radius of borehole (m)
# r  - radius of influence (m)
# Tu - transmissivity of upper aquifer (m^2/yr)
# Hu - potentiometric head of upper aquifer (m)
# Tl - transmissivity of lower aquifer (m^2/yr)
# Hl - potentiometric head of lower aquifer (m)
# L  - length of borehole (m)
# Kw - hydraulic conductivity of borehole (m/yr)

# For more info, see: http://www.sfu.ca/~ssurjano/borehole.html

def model(X):
    X = np.array(X, ndmin=2)

    rw = X[:, 0]
    r = X[:, 1]
    Tu = X[:, 2]
    Hu = X[:, 3]
    Tl = X[:, 4]
    Hl = X[:, 5]
    L = X[:, 6]
    Kw = X[:, 7]

    # Precalculate the logarithm:
    Logrrw = np.log(np.divide(r, rw))

    Numerator = 2*np.pi*Tu*(Hu - Hl)
    Denominator = Logrrw*(1 + np.divide(
        (2 * L * Tu), (Logrrw * (rw**2) * Kw)) + np.divide(Tu, Tl))

    return np.divide(Numerator, Denominator)
