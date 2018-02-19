import random
import numpy as np
import matplotlib.pyplot as plt
import kernel as ker


def generate_input(num):
    class1 = np.concatenate((np.random.randn(num//2, 2)*0.2 + [1.5, 0.5],
                            np.random.randn(num//2,2) * 0.2 + [-1.5, 0.5]))
    class2 = np.random.randn(num, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((class1, class2))
    targets = np.concatenate((np.ones((class1.shape[0], 1)), -np.ones((class2.shape[0], 1))))
    # Shuffles the inputs with permutations
    num_inputs = inputs.shape[0]
    permute = list(range(num_inputs))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute, :]

    plt.plot([p[0] for p in class1], [p[1] for p in class1], "o", c= "b")
    plt.plot([p[0] for p in class2], [p[1] for p in class2], "o", c= "r")

    return inputs, targets


def pre_comp_matrix(m_inputs, v_target, kernel):
    """
    Helpfunction to compute the values in the help matrix M_P
    ti*tj*kernel()
    """
    num_inputs = m_inputs.shape[0]
    m_result = np.zeros([num_inputs, num_inputs])

    for i in range(num_inputs):
        for j in range(num_inputs):
            #print(kernel(m_inputs[i, :], m_inputs[j, :]))
            #print(v_target[i])
            #print(v_target[j])
            m_result[i, j] = kernel(m_inputs[i, :], m_inputs[j, :]) * v_target[i] * v_target[j]
    return m_result


def extract_non_zeros(alphas, thresh=1e-5):
    """ Extracts nonzero alpha values and their index from alphas """

    extracted_alphas = []
    indices = []

    for ind, alpha in enumerate(alphas):
        if alpha > thresh:
            extracted_alphas.append(alpha)
            indices.append(ind)

    return extracted_alphas, indices


def indicator(alphas, targets, s, x, b):
    """
    The indicator function which uses the non-zero
    α i ’s together with their ⃗x i ’s and t i ’s to classify new points.
    """
    print(s)
    ind_out = 0
    for i in range(len(alphas)):
        ind_out += alphas[i]*targets[i]*ker.lin(s, x[i]) - b
    print(ind_out)
    return ind_out


def calc_b(alphas, targets, inputs, kernel):
    s = inputs[0]
    sum = 0
    for i in range(len(alphas)):
        sum += alphas[i]*targets[i]*kernel(s, inputs[i])
    return sum - targets[0]
