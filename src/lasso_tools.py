import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC


# a function that might be useful for doing cross-validation of fitting data
# this isn't actually used in the code currently

def lassoplt(U,F):

    alphas = np.logspace(-10, -3, num=50)
    print 'alphas'
    print alphas

    model = LassoCV(cv=10, alphas=alphas,  fit_intercept=True, normalize=True,max_iter=15000, tol=1.0e-6, selection='random').fit(U, F)
    
    m_log_alphas = -np.log10(model.alphas_)

    plt.figure()
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                 label='alpha: CV estimate')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent ')
    plt.axis('tight')

    plt.show()

    return model

