import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss

import fit_bernoulli
from fit_bernoulli import fit_bernoulli


ntrials = 50
niter = 5
ptrue = np.linspace(0, 1, ntrials)
ptrue = (1 + np.cos(np.linspace(0, 2*np.pi, ntrials)))/2.
np.random.seed(0)
x = np.random.rand(ntrials) < ptrue
#x = [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,]


for chance_prior_type in ['fixed', 'update', 'none']:

    pcur = np.zeros((ntrials, niter))
    for i in range(niter):
        p = fit_bernoulli(x, chance_prior_type=chance_prior_type)
        pcur[:, i] = np.array(p[0])

    plt.errorbar(np.arange(ntrials), pcur.mean(1), yerr=np.std(pcur, axis=1, ddof = 1), label=chance_prior_type)

plt.plot(ptrue, color='red', alpha=0.4, label='true')
plt.plot(x, '.', label='data')
plt.legend(loc=(1, 0))
plt.tight_layout()
plt.savefig('test3.png')