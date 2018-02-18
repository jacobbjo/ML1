import numpy as np
from kernel import *
import matplotlib as plt
from scipy.optimize import minimize


def pre_comp_matrix(m_inputs, v_target, kernel):
    """ Helpfunction to compute the values in the help matrix M_P
        ti*tj*kernel()
    """
    num_inputs = m_inputs.shape[0]
    m_result = np.zeros([num_inputs, num_inputs])

    for i in range(num_inputs):
        for j in range(num_inputs):
            m_result[i, j] = kernel(m_inputs[:, i], m_inputs[:, j]) * v_target[i] * v_target[j]
    return m_result


def objective(v_alpha):
    """
    Computes 1/2(sum(sum(alphai*alphaj*targeti*targetj*Kerner(xi, xj))) - sum(alphai)
    Use global values for kernel and values and targets
    """
    # Matrix for combinations of the alphas
    m_alpha = np.dot(v_alpha, np.transpose(v_alpha))
    # alpha values multiplied with the target and kernel values
    m_mult = m_alpha * M_P
    # sum the columns and the rows together
    sum = np.sum(m_mult)/2

    alphaSum = np.sum(v_alpha)

    return sum - alphaSum


M_P = pre_comp_matrix(m_inputs, v_target, ker_lin)

# Looking for trouble

x = np.array([[1,2], [3,4]])
y = np.array([2,3])
z = np.array([1,2])
sum = np.sum(x)
#sumsum = np.sum(sum)
print(sum)
#print(sumsum)


#print(x[:, 0])
#print("* ", str(x*y))
#print("* ", str(x*y*np.transpose(z)))
#print("Multiply: ",str(np.multiply(x, y)))
#print("dot: ", str(np.dot(x, y)))

#Hipp hurra för här kommer bumbibjörnarna