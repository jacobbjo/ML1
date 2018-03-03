import kernel as ker
from funhelp import *
from scipy.optimize import minimize

# GLOBAL VARIABLES
INPUTS, TARGET = generate_lin_input(20, 0.4)
#INPUTS, TARGET = generate_nonlin_input(40, 0.2)
#INPUTS, TARGET = generate_circle_input(20, 0.2)

KERNEL = ker.pol
M_P = pre_comp_matrix(INPUTS, TARGET, KERNEL)



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
    C = None
    N = INPUTS.shape[0]

    v_alpha_init = np.zeros(N)

    bounds = [(0, C) for b in range(N)]

    ret = minimize(objective, v_alpha_init, bounds=bounds, constraints={"type": "eq", "fun": zerofun})
    print(ret["success"])
    if ret["success"]:
        new_alphas, indices = extract_non_zeros(ret["x"])
        new_targets = np.array([TARGET[i, 0] for i in indices])
        new_inputs = np.array([INPUTS[i, :] for i in indices])

        #print(new_alphas)
        #print(new_targets)
        b = calc_b(new_alphas, new_targets, new_inputs, KERNEL)
        print(b)

        #plt.plot([x[0] for x in new_inputs], [y[1] for y in new_inputs], "o", c="g")

        xgrid = np.linspace(-5, 5)
        #print(xgrid)
        ygrid = np.linspace(-4, 4)
        X, Y = np.meshgrid(xgrid, ygrid)
        grid = np.array([[indicator(new_alphas, new_targets, [x, y], new_inputs, b, KERNEL) for x in xgrid] for y in ygrid])

        #plt.plot(X, grid, "o")
        #plt.plot(Y, grid, "o")


        plt.contour(X, Y, grid, (-1.0, 0.0, 1.0), colors = ("red", "black", "blue"), linewidths = (1, 3, 1))
        #plt.contour(X, (1.0))


        plt.axis("equal")
    else:
        print("Could not minimize")
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

