import kernel as ker
from funhelp import *
from scipy.optimize import minimize

# GLOBAL VARIABLES
INPUTS, TARGET = generate_input(20)
M_P = pre_comp_matrix(INPUTS, TARGET, ker.lin)


def objective(v_alpha):
    """
    Computes 1/2(sum(sum(alphai*alphaj*targeti*targetj*Kerner(xi, xj))) - sum(alphai)
    Use global values for kernel and values and targets
    """
    # reshape because a_alpha has shape (N,)
    v_alpha = np.reshape(v_alpha, [v_alpha.shape[0], 1])
    # Matrix for combinations of the alphas
    m_alpha = np.dot(v_alpha, np.transpose(v_alpha))
    # alpha values multiplied with the target and kernel values
    m_mult = m_alpha * M_P
    # sum the columns and the rows together
    sum = np.sum(m_mult)/2

    alphaSum = np.sum(v_alpha)
    return sum - alphaSum


def zerofun(alpha):
    """
    Calculates the value which should be constrained to zero.
    Like objective, zerofun takes a vector alpha and a target
    as argument and returns a scalar value.
    """
    result = np.dot(np.transpose(alpha), TARGET)
    return result


def main():
    C = 1
    N = INPUTS.shape[0]

    v_alpha_init = np.zeros(N)

    bounds = [(0, C) for b in range(N)]

    ret = minimize(objective, v_alpha_init, bounds=bounds, constraints={"type": "eq", "fun": zerofun})
    print(ret["x"])
    print(ret["success"])

    new_alphas, indices = extract_non_zeros(ret["x"])
    new_targets = [TARGET[i, 0] for i in indices]

    print(new_alphas)
    print(new_targets)

    plt.axis("equal")
    plt.show()


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

