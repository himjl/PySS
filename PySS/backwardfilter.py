import numpy as np

def backwardfilter(x, xold, sigsq, sigsqold):
    '''
    backwardfilter is a helper function that implements the backward filter
    smoothing algorithm to estimate the learning state at trial k, given all
    the data, as the Gaussian random variable with mean x{k|K} (xnew) and
    SIG^2{k|K} (signewsq).

    :param x:
    :param xold:
    :param sigsq:
    :param sigsqold:
    :return:
    '''

    T = x.shape[1] # total number of posterior mode estimates (K + 1)

    # Initial conditions: use values of posterior mode and posterior variance
    xnew = list(np.zeros(T-1)) + [x[T-1]]
    signewsq = list(np.zeros(T-1)) + [sigsq[T-1]]
    xnew = np.array(xnew)[None, :]
    signewsq = np.array(signewsq)[None, :]

    A = np.zeros((1, T-1))
    for i in range(T-1, 1):
        A[0, i - 1] = np.true_divide(sigsq[i - 1], sigsqold[i])
        xnew[0, i - 1] = x[i - 1] + A[0, i - 1] * (xnew[0, i] - xold[i])
        signewsq[0, i - 1] = sigsq[i - 1] + A[0, i - 1] * A[0, i - 1] * (signewsq[0, i] - sigsqold[i])

    return xnew, signewsq, A
