# Adapted from code written by Anne Smith in 2003 from the Brown Lab at MIT
import numpy as np


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


def forwardfilter(I, sigE, xguess, sigsqguess, mu):
    '''
    forwardfilter is a helper function that implements the forward recursive
    filtering algorithm to estimate the learning state (hidden process) at
    trial k as the Gaussian random variable with mean x{k|k} (xhat) and
    SIG^2{k|k} (sigsq).

    :param I: [2xN]. dtype = int. [num_successes_v_time, total_possible]
    :param sigE: (). dtype = float. default variance of learning state process
    :param xguess: ().
    :param sigsqguess: ().
    :param mu: ().
    '''

    '''
    :param xhatold: x{k|k-1}, one-step prediction (equation A.6)*
    :param sigsqold: SIG^2{k|k-1}, one-step prediction variance (equation A.7)*
    :param xhat         x{k|k}, posterior mode (equation A.8)*
    :param sigsq        SIG^2{k|k}, posterior variance (equation A.9)*
    :param p            p{k|k}, observation model probability (equation 2.2)*
    :param N            vector of number correct at each trial
    :param Nmax         total number that could be correct at each trial
    :param K            total number of trials
    :param number_fail  saves the time steps if Newton's Method fails
    '''

    K = I.shape[1] # total number of trials
    N = I[0, :] # vector of number correct at each trial
    Nmax = I[1, :] # total number that could be correct at each trial

    # Initial conditions: use values from previous iteration
    xhat = [xguess]
    sigsq = [sigsqguess]
    number_fail = []

    xhatold = [0]
    sigsqold = [0]

    for k in range(2, K+1+1):
        # for each trial, compute estimates of the one-step prediction, the
        # posterior mode (using Newton's Method), and the posterior variance
        # (estimates from subject's POV)

        # Compute the one-step prediction estimate of mean and variance
        xhatold.append(xhat[k-1-1])
        sigsqold.append(sigsq[k-1-1] + np.square(sigE))

        # Use Newton's Method to compute the nonlinear posterior mode estimate
        xhat_next, flagfail = newtonsolve(mu, xhatold[k-1], sigsqold[k-1], N[k-1-1], Nmax[k-1-1])
        xhat.append(xhat_next)

        # if Newton's Method fails, number_fail saves the time step
        if flagfail > 0:
            number_fail.append(k-1)

        # Compute the posterior variance estimate
        denom = np.true_divide(-1, sigsqold[k-1]) - Nmax[k-1-1]*np.exp(mu)*np.exp(xhat[k-1])/np.square(1 + np.exp(mu) * np.exp(xhat[k-1]))
        sigsq.append(np.true_divide(-1, denom))

    if len(number_fail) > 0:
        print 'Newton convergence failed at times', number_fail

    # Compute the observation model probability estimate
    p = np.true_divide(np.exp(mu) * np.exp(xhat), 1 + np.exp(mu) * np.exp(xhat))

    xhat = np.array(xhat)[None, :]
    return p, xhat, sigsq, xhatold, sigsqold


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


def pdistnv2(x, s, mu, background_prob):
    '''
    pdist is a helper function that calculates the confidence limits of the EM
    estimate of the learning state process.  For each trial, the function
    constructs the probability density for a correct response.  It then builds
    the cumulative density function from this and computes the p values of
    the confidence limits

    :param x: (1 x ntrials)
    :param s:
    :param mu: ()
    :param background_prob: ()
    :return:
    '''

    samps = []
    num_samps = 10000
    p025 = np.zeros(x.shape)
    p975 = np.zeros(x.shape)
    pmid = np.zeros(x.shape)
    pmodnew = np.zeros(x.shape) # not computed
    pcert = np.zeros(x.shape)

    for ov in range(1, x.shape[1]+1):

        xx = x[0, ov-1]
        ss = s[0, ov-1]
        samps = xx + np.sqrt(ss)*np.random.randn(num_samps)
        pr_samps = np.true_divide(np.exp(mu + samps), 1 + np.exp(mu + samps))

        order_pr_samps = np.sort(pr_samps)

        p025[ov - 1] = order_pr_samps[int(np.floor(0.025 * num_samps))]
        p975[ov - 1] = order_pr_samps[int(np.ceil(0.975 * num_samps))]
        pmid[ov - 1] = order_pr_samps[int(np.round(0.5 * num_samps))]
        pcert[ov - 1] = np.mean(pr_samps > background_prob)

    return p025, p975, pmid, pmodnew, pcert


def em_bino(I, xnew, signewsq, A, startflag):
    '''
    em_bino is a helper function that computes sigma_eps squared (estimated
    learning process variance).
    :param I:
    :param xnew:
    :param signewsq:
    :param A:
    :param startflag:
    :return:
    '''

    newsigsq = None

    M = xnew.shape[1]
    xnewt = xnew[0, 2:M]
    xnewtm1 = xnew[0, 1:M-1]
    signewsqt = signewsq[2:M]
    A = A[1:]

    covcalc = signewsqt * A

    term1 = np.sum(np.square(xnewt)) + np.sum(signewsqt)
    term2 = np.sum(covcalc) + np.sum(xnewt * xnewtm1)

    if startflag == 1:
        term3 = 1.5 * xnew[1] * xnew[1] + 2 * signewsq[1]
        term4 = np.square(xnew[-1]) + signewsq[-1]
    elif startflag == 0:
        term3 = 2 * xnew[1] * xnew[1] + 2 * signewsq[1]
        term4 = np.square(xnew[-1]) + signewsq[-1]
    elif startflag == 2:
        term3 = 1 * xnew[1]*xnew[1] + 2*signewsq[1]
        term4 = np.square(xnew[-1]) + signewsq[-1]
        M = M - 1
    else:
        raise ValueError('startflag of %d not supported'%startflag)
    newsigsq = (2 * (term1 - term2) + term3 - term4) / float(M)

    return newsigsq


def newtonsolve(mu,  xold, sigoldsq, N, Nmax):

    '''

    newtonsolve is a helper function that implements Newton's Method in order
    to recursively estimate the posterior mode (x).  Once the subsequent estimates
    sufficiently converge, the function returns the last estimate.  If, having
    never met this convergence condition, the function goes through all of the
    recursions, then a special flag (timefail) - indicating the convergence
    failure - is returned along with the last posterior mode estimate.

    :param mu:
    :param xold:
    :param sigoldsq:
    :param N:
    :param Nmax:
    :return:
    '''

    it = xold + sigoldsq * (N - Nmax * np.exp(mu) * np.exp(xold) / (1 + np.exp(mu) * np.exp(xold)))
    it = [it]

    g = []
    gprime = []
    niter = 100

    for i in range(niter):
        gupdate = xold + sigoldsq*(N - Nmax*np.exp(mu)*np.exp(it[i])/(1+np.exp(mu)*np.exp(it[i]))) - it[i]
        g.append(gupdate)

        gprime_update = -Nmax*sigoldsq*np.exp(mu)*np.exp(it[i])/np.square(1+np.exp(mu)*np.exp(it[i])) - 1
        gprime.append(gprime_update)

        it_update = it[i] - g[i]/gprime[i];
        it.append(it_update)

        x = it[-1]

        if np.abs(x - it[-2]) < 1e-8:
            timefail = 0
            return x, timefail

    # This tries a new initial condition if first Newtons doesn't work
    if i == (niter - 1):
        it = [-1]
        g = []
        gprime = []

        for i in range(niter):
            gupdate = xold + sigoldsq*(N - Nmax*np.exp(mu)*np.exp(it[i])/(1+np.exp(mu)*np.exp(it[i]))) - it[i]
            g.append(gupdate)

            gprime_update = -Nmax*sigoldsq*np.exp(mu)*np.exp(it[i])/np.square(1+np.exp(mu)*np.exp(it[i]))- 1
            gprime.append(gprime_update)

            it_update = it[i] - g[i] / gprime[i]
            it.append(it_update)

            x = it[-1]

            if np.abs(x - it[i]) < 1e-8:
                timefail = 0
                return x, timefail

    # This tries a new initial condition if second Newtons doesn't work
    if i == (niter - 1):
        it = [1]
        g = []
        gprime = []

        for i in range(niter):
            gupdate = xold + sigoldsq*(N - Nmax*np.exp(mu)*np.exp(it[i])/(1+np.exp(mu)*np.exp(it[i]))) - it[i]
            g.append(gupdate)

            gprime_update = -Nmax*sigoldsq*np.exp(mu)*np.exp(it[i])/np.square(1+np.exp(mu)*np.exp(it[i]))- 1
            gprime.append(gprime_update)

            it_update = it[i] - g[i] / gprime[i]
            it.append(it_update)

            x = it[-1]

            if np.abs(x - it[i]) < 1e-8:
                timefail = 0
                return x, timefail

    timefail = 1
    return x, timefail


# Helper functions
def _check_input(Responses):
    Responses = np.array(Responses)

    assert len(Responses.shape) == 2
    if Responses.shape[0] != 1:
        Responses = np.transpose(Responses)

    assert Responses.shape[0] == 1
    return Responses
