# Adapted from code written by Anne Smith in 2003 from the Brown Lab at MIT
import numpy as np


import forwardfilter
reload(forwardfilter)
from forwardfilter import forwardfilter

import backwardfilter
reload(backwardfilter)
from backwardfilter import backwardfilter

import em_bino
reload(em_bino)
from em_bino import em_bino

import pdistnv2
reload(pdistnv2)
from pdistnv2 import pdistnv2


def fit_bernoulli(Responses, MaxResponse=1, BackgroundProb=0.5):
    '''

    :param Responses: sequence of 0s or 1s. dtype = list, np.array. shape = (1, N)
    :param MaxResponse: ()
    :param BackgroundProb: ()
    :return: pmid, p025, p075
    '''

    SigE = 0.5  # default variance of learning state process is sqrt(0.5)
    UpdaterFlag = 0 # default fixed i.c.

    I = np.stack([Responses, MaxResponse * np.ones((1, Responses.shape[1]))], axis=0)

    SigsqGuess = np.square(SigE)

    # set the value of mu from the chance of correct
    mu = np.log(np.true_divide(BackgroundProb, 1 - BackgroundProb))

    # convergence criterion for SIG_EPSILON^2
    CvgceCrit = 1e-5

    ''' Start '''
    xguess = 0
    NumberSteps = 3000

    newsigsq = []
    xnew1save = []
    for i in range(NumberSteps):
        p, x, s, xold, sold = forwardfilter(I, SigE, xguess, SigsqGuess, mu)

        # Compute the backward (smoothing algorithm) estimates of the learning
        # state and its variance: x{k|K} and sigsq{k|K}
        xnew, signewsq, A = backwardfilter(x, xold, s, sold)

        if UpdaterFlag == 1:
            xnew[0] = 0.5 * xnew[1] # updates the initial value of the latent process
            signewsq[0] = np.square(SigE)
        elif UpdaterFlag == 0:
            xnew[0] = 0 # fixes initial value (no bias at all)
            signewsq[0] = np.square(SigE)
        elif UpdaterFlag == 2:
            xnew[0] = xnew[1] # x[0] = x[1] means no prior chance probability
            signewsq[0] = signewsq[1]

        # Compute the EM estimate of the learning state process variance
        newsigsq.append(em_bino(I, xnew, signewsq, A, UpdaterFlag))

        xnew1save.append(xnew[0])

        # Check for convergence
        if i > 0:
            a1 = np.abs(newsigsq[i] - newsigsq[i-1])
            a2 = np.abs(xnew1save[i] - xnew1save[i-1])

            if (a1 < CvgceCrit) and (a2 < CvgceCrit) and (UpdaterFlag >= 1):
                print 'EM estimates of learning state process variance and start point converged after %d steps'%(i+1)
                break
            elif (a1 < CvgceCrit) and (UpdaterFlag == 0):
                print 'EM estimates of learning state process variance converged after %d steps'%(i+1)

        SigE = np.sqrt(newsigsq[i])
        xguess = xnew[0]
        SigsqGuess = signewsq[0]

    if i == (NumberSteps-1):
        print 'Failed to converge after %d steps; convergence criterion was %f'%(i+1, CvgceCrit)

    # Use sampling to convert from state to a probability

    # Can used smoothed or filtered estimates - here used smoothed
    p025, p975, pmid, _, pcert = pdistnv2(xnew, signewsq, mu, BackgroundProb);

    # Remove the prior
    p025 = p025[1:]
    p975 = p975[1:]
    pmid = pmid[1:]

    return pmid, p025, p975

# Helper functions
def _check_input(Responses):
    Responses = np.array(Responses)

    assert len(Responses.shape) == 2
    if Responses.shape[0] != 1:
        Responses = np.transpose(Responses)

    assert Responses.shape[0] == 1
    return Responses
