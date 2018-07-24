import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.linear_model import lasso_path

#some basic tools for testing fits of lasso model
#U has dependent variables (distortions), F the independent (the forces + energies + stresses)

def lasso_path(U,F):

    alphas = np.logspace(-10, -3, num=50)
    print 'alphas'
    print alphas

    a,b,c = lasso_path(U,F,alphas=alphas,  fit_intercept=False, normalize=False,max_iter=1000, tol=0.000001, selection='random').fit(U, F)

    return a,b,c



def lassoplt(U,F):

    alphas = np.logspace(-10, -3, num=50)
    print 'alphas'
    print alphas

    model = LassoCV(cv=10, alphas=alphas,  fit_intercept=False, normalize=False,max_iter=1000, tol=0.000001, selection='random').fit(U, F)
    
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

def my_lasso_path(U,F):

    alphas = np.logspace(-10, -3, num=50)
    print 'alphas'
    print alphas

    a,b,c = lasso_path(U,F,alphas=alphas,  fit_intercept=False, normalize=False,max_iter=1000, tol=0.000001, selection='random')
    
    print 'alpha n_nonzero'
    t = np.sum(b!=0,0)
    for aa,tt in zip(a.tolist(),t.tolist()):
        print str(aa) + '\t'+str(tt)

    print

    a_st,b_st,c_st = lasso_path(U,F,alphas=alphas,  fit_intercept=True, normalize=True,max_iter=1000, tol=0.000001, selection='random')
    
    print 'alpha n_nonzero'
    t_st = np.sum(b_st!=0,0)
    for aa,tt,tt_st in zip(a.tolist(),t.tolist(), t_st.tolist()):
        print [aa,tt,tt_st]

    print

    return a,b,c

