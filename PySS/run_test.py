import fit_bernoulli
reload(fit_bernoulli)
from fit_bernoulli import fit_bernoulli

import numpy as np

ntrials = 400
ptrue = np.linspace(0, 1, ntrials)
x = np.random.rand(ntrials) < ptrue

p = fit_bernoulli(x, MaxResponse=1, BackgroundProb=0.5)
print p[0]

import matplotlib.pyplot as plt
import tasso as ts
plt.plot(p[0])
plt.plot(p[1], lw=1, color='gray')
plt.plot(p[2], lw=1, color='gray')
plt.plot(x, '.')
ts.savefig('test.png')