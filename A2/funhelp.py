import random
import numpy as np
import matplotlib.pyplot as plt


def generate_lin_input(num_points, std):
    class1 = np.concatenate((np.random.randn(num_points // 2, 2) * std + [1.5, 0.5],
                             np.random.randn(num_points // 2, 2) * std + [-1.5, 0.5]))
    #class1 = np.random.randn(num_points, 2) * std + [1.5, 0.5]
    class2 = np.random.randn(num_points, 2) * std + [0.0, -0.5]

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

def generate_nonlin_input(num_points, std):
    class1 = np.concatenate((np.random.randn(num_points // 4, 2) * std + [3.5, 1.5],
                             np.random.randn(num_points // 4, 2) * std + [-3.5, 1.5],
                             np.random.randn(num_points // 4, 2) * std + [0.0, 1.5],
                             np.random.randn(num_points // 4, 2) * std + [-3.5, -1.5]
                             ))

    class2 = np.random.randn(num_points, 2) * std + [0.0, -0.5]

    return organize_data(class1, class2)

def generate_circle_input(num_points, std):
    circle_points1 = generate_circle(10, 1)
    class1 = np.concatenate((np.random.randn(num_points, 2) * std + circle_points1[0],
                             np.random.randn(num_points, 2) * std + circle_points1[1],
                             np.random.randn(num_points, 2) * std + circle_points1[2],
                             np.random.randn(num_points, 2) * std + circle_points1[3],
                             np.random.randn(num_points, 2) * std + circle_points1[4],
                             np.random.randn(num_points, 2) * std + circle_points1[5],
                             np.random.randn(num_points, 2) * std + circle_points1[6],
                             np.random.randn(num_points, 2) * std + circle_points1[7],
                             np.random.randn(num_points, 2) * std + circle_points1[8],
                             np.random.randn(num_points, 2) * std + circle_points1[9]))

    class2 = np.random.randn(num_points, 2) * std + [0.0, 0.0]

    return organize_data(class1, class2)




def generate_circle(num_points, radius):
    angles = np.linspace(0, 2* np.pi, num_points)
    x = [radius * np.cos(ang) for ang in angles]
    y = [radius * np.sin(ang) for ang in angles]
    #test = np.array([[xx, yy] for xx, yy in zip(x, y)])
    #plt.plot([p[0] for p in test], [p[1] for p in test])
    #plt.axis("equal")
    #plt.show()
    return np.array([[xx, yy] for xx, yy in zip(x, y)])


def organize_data(class1, class2):
    inputs = np.concatenate((class1, class2))
    targets = np.concatenate((np.ones((class1.shape[0], 1)), -np.ones((class2.shape[0], 1))))
    # Shuffles the inputs with permutations
    num_inputs = inputs.shape[0]
    permute = list(range(num_inputs))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute, :]

    plt.plot([p[0] for p in class1], [p[1] for p in class1], "o", c="b")
    plt.plot([p[0] for p in class2], [p[1] for p in class2], "o", c="r")

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


def indicator(alphas, targets, s, x, b, kernel):
    """
    The indicator function which uses the non-zero
    α i ’s together with their ⃗x i ’s and t i ’s to classify new points.
    """
    ind_out = 0
    for i in range(len(alphas)):
        ind_out += alphas[i]*targets[i]*kernel(s, x[i])


    return ind_out -b


def calc_b(alphas, targets, inputs, kernel):
    """ Calculate the b value using equation (7) """
    s = inputs[0]
    sum = 0
    for i in range(len(alphas)):
        sum += alphas[i]*targets[i]*kernel(s, inputs[i])
    return sum - targets[0]