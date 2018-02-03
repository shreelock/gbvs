import numpy as np
import scipy.io
import sklearn.preprocessing
import markovChain

def loadGraphDistanceMatrixFor28x32():
    f = scipy.io.loadmat("./28__32__m__2.mat")
    distanceMat = np.array(f['grframe'])[0][0][0]
    lx = np.array(f['grframe'])[0][0][1]
    dim = np.array(f['grframe'])[0][0][2]
    return [distanceMat, lx, dim]

def calculate(map, sigma):
    [distanceMat, _, _] = loadGraphDistanceMatrixFor28x32()
    denom = 2 * pow(sigma, 2)
    expr = -np.divide(distanceMat, denom)
    Fab = np.exp(expr)

    map_linear = np.ravel(map, order='F')  # column major
    state_transition_matrix = np.zeros_like(distanceMat, dtype=np.float32)

    # calculating STM : w = d*Fab
    for i in xrange(distanceMat.shape[0]):
        for j in xrange(distanceMat.shape[1]):
            state_transition_matrix[i][j] = Fab[i][j] * abs(map_linear[i] - map_linear[j])

    # normalising outgoing weights of each node to sum to 1, using scikit normalize
    norm_STM = sklearn.preprocessing.normalize(state_transition_matrix, axis=0, norm='l1')

    # caomputing equilibrium state of a markv chain is same as computing eigen vector of its weight matrix
    # https://lps.lexingtonma.org/cms/lib2/MA01001631/Centricity/Domain/955/EigenApplications%20to%20Markov%20Chains.pdf
    eVec = markovChain.solve(norm_STM, 0.0001)
    processed_reshaped = np.reshape(eVec, map.shape, order='F')

    return processed_reshaped

def normalize(map, sigma):
    [distanceMat, _, _] = loadGraphDistanceMatrixFor28x32()
    denom = 2 * pow(sigma, 2)
    expr = -np.divide(distanceMat, denom)
    Fab = np.exp(expr)

    map_linear = np.ravel(map, order='F')  # column major
    state_transition_matrix = np.zeros_like(distanceMat, dtype=np.float32)

    # calculating STM : w = d*Fab
    for i in xrange(distanceMat.shape[0]):
        for j in xrange(distanceMat.shape[1]):
            state_transition_matrix[i][j] = Fab[i][j] * abs(map_linear[i])

    # normalising outgoing weights of each node to sum to 1, using scikit normalize
    norm_STM = sklearn.preprocessing.normalize(state_transition_matrix, axis=0, norm='l1')

    # caomputing equilibrium state of a markv chain is same as computing eigen vector of its weight matrix
    # https://lps.lexingtonma.org/cms/lib2/MA01001631/Centricity/Domain/955/EigenApplications%20to%20Markov%20Chains.pdf
    eVec = markovChain.solve(norm_STM, 0.0001)
    processed_reshaped = np.reshape(eVec, map.shape, order='F')

    return processed_reshaped