import numpy as np
import random
import kernel as ker
import matplotlib.pyplot as plt
from scipy.optimize import minimize



def pre_comp_matrix(m_inputs, v_target, kernel):
    """ Helpfunction to compute the values in the help matrix M_P
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

def generateInput():
    class1 = np.concatenate((np.random.randn(10, 2)*0.2 + [1.5, 0.5],
                            np.random.randn(10,2) * 0.2 + [-1.5, 0.5]))
    class2 = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

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


m_inputs, TARGET = generateInput()
M_P = pre_comp_matrix(m_inputs, TARGET, ker.lin)

def main():

    C = None
    N = m_inputs.shape[0]

    #v_alpha_init = np.random.randn(N, 1)
    v_alpha_init = np.zeros([N, 1])

    bounds = [(0,C) for b in range(N)]

    ret = minimize(objective, v_alpha_init, bounds=bounds, constraints={"type":"eq", "fun": zerofun})#,options={'xtol': 1e-8, 'disp': True})
    print(ret["x"])
    print(ret["success"])

    plt.axis("equal")
    plt.show()



def zerofun(alpha):
    """
    Calculates the value which should be constrained to zero.
    Like objective, zerofun takes a vector alpha and a target
    as argument and returns a scalar value.
    """
    result = np.dot(alpha, TARGET)
    return result




if __name__ == "__main__":
    main()




#x = np.array([[1,2], [3,4]])
#y = np.array([2,3])
#z = np.array([1,2])

#print(x[:, 0])
#print("* ", str(x*y))
#print("* ", str(x*y*np.transpose(z)))
#print("Multiply: ",str(np.multiply(x, y)))
#print("dot: ", str(np.dot(x, y)))

