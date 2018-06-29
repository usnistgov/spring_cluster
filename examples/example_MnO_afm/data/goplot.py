import numpy as np
import matplotlib as plt



def goplot(a,b):

    amax = max(np.max(a), np.max(b))
    amin = min(np.max(a), np.min(b))

    plt.clf()
    plt.plot([amin,amax],[amin,amax], 'k')
    plt.plot(t[:,0], t[:,1], '.',color='green',markersize=5)
    plt.plot(t[:,0], t[:,1], '.',color='orange',markersize=8)

    plt.show()
