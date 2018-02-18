import numpy as np

# This function should implement the equality constraint of (10).
# Also here, you can make use of numpy.dot to be efficient.


def zerofun(alpha, target):
    """
    Calculates the value which should be constrained to zero.
    Like objective, zerofun takes a vector alpha and a target
    as argument and returns a scalar value.
    """
    return np.dot(np.transpose(alpha), target)

