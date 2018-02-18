import numpy as np

"""
A suitable kernel function.

The kernel function takes two data points as arguments and returns a
“scalar product-like” similarity measure; a scalar value. Start with the
linear kernel which is the same as an ordinary scalar product, but also
explore the other kernels in section 3.3. """


def ker_lin(v_x, v_y):
    """"
    This kernel simply returns the scalar product between the two points.
    This results in a linear separation.
    """
    return np.dot(np.transpose(v_x),  v_y)


def ker_pol(v_x, v_y, p):
    """
    This kernel allows for curved decision boundaries. The exponent p (a
    positive integer) controls the degree of the polynomials. p = 2 will make
    quadratic shapes (ellipses, parabolas, hyperbolas). Setting p = 3 or higher
    will result in more complex shapes.
    """
    return (np.dot(np.transpose(v_x),  v_y) + 1)**p


def ker_rbf(v_x, v_y, sigma):
    """
    This kernel uses the explicit euclidian distance between the two datapoints,
    and often results in very good boundaries. The parameter σ is used to
    control the smoothness of the boundary.
    """
    return np.exp(-np.linalg.norm(v_x-v_y)**2 / 2*sigma**2)

