'''
Author : Renzhuo Huang
Date   : 2023-05-27

This is a code to realize Logistic Mixture Auto-regressive 
Model under co-integration assumptions and manage to give 
parameters estimation with LSE-EM method. 

Orginal paper link:
https://doi.org/10.1080/14697688.2010.506445
Paper author:
Xixin Cheng , Philip L.H. Yu & W.K. Li
'''

import numpy as np
import scipy.stats


def series_padding(a_t_arr, lags, method='sym'):
    if method == 'sym':
        a_t_arr = np.hstack((a_t_arr[lags:0:-1],
                             a_t_arr))
    elif method == 'mean':
        a_t_arr = np.hstack((np.full(lags, np.mean(a_t_arr[0:lags])),
                             a_t_arr))
    elif method == 'first':
        a_t_arr = np.hstack((np.full(lags, a_t_arr[0]),
                             a_t_arr))
    a_window_arr = np.lib.stride_tricks.sliding_window_view(a_t_arr[0:-1], lags)
    a_window_arr = a_window_arr.astype(np.float32)
    return a_window_arr


def parameter_estimation(a_t_arr:np.ndarray, 
                         ksai:np.ndarray, 
                         zeta1:np.ndarray, zeta2:np.ndarray, 
                         sigma1:float, sigma2:float, *args, **kwargs):
    '''
    Optimizing likelihood function Q(.) by implementing LSE-EM method,
    parameter set defined as Theta:(ksai, zeta1, zeta2, sigma1, sigma2)
    representing the probability of residual returns array to fall into 
    regime 2 and regime 1 of stochastic process (random walk with sigma2
    / mean-reverting process with sigma1).

    Parameters
    ----------
    a_t_arr : np.ndarray, shape of (T,);
        Residuals of OLS regression conducted from specific "Basket" of
        co-integration assets ( i.e. I(1) series' linear combinations ).
    ksai    : np.ndarray, shape of (n,);
        parameters to be estimated.
    zeta1   : np.ndarray, shape of (mk+1,);
        parameters to be estimated.
    zeta2   : np.ndarray, shape of (mk+2,);
        parameters to be estimated.
    sigma1  : float;
        parameters to be estimated.
    sigma2  : float;
        parameters to be estimated.
    **kwargs: method to fill NAN when implementing functions
        rolling forward, 
        "sym": symmetric values by the first to fill NAN;
        "mean": a series of mean values to fill NAN;
        "first": use first value to fulfill all NAN.
    
    Returns
    -------
    ksai_iter  : np.ndarray, shape of (n,);
        parameters estimated after this iteration.
    zeta1_iter : np.ndarray, shape of (m1,);
        parameters estimated after this iteration.
    zeta2_iter : np.ndarray, shape of (m2,);
        parameters estimated after this iteration.
    sigma1_iter: float;
        parameters estimated after this iteration.
    sigma2_iter: float;
        parameters estimated after this iteration.
    '''
    T = a_t_arr.shape[0]
    n, m1, m2 = ksai.shape[0]-1, zeta1.shape[0]-1, zeta2.shape[0]-1
    ksai   = ksai.reshape(n+1,1)
    zeta1  = zeta1.reshape(m1+1,1)
    zeta2  = zeta2.reshape(m2+1,1)
    
    if sigma1<0 or sigma2<0:
        raise ValueError("sigma should not be a nagative value!")
    sigma1 = float(max(1e-10,sigma1))
    sigma2 = float(max(1e-10,sigma2))

    # Generate three main series in shape of (T,n+1), (T,m1+1), (T,m2+1) 
    # respectively. Compatible for arbitary {a_t} series, and therefore
    # is convinient to change "Baskets" and apply parameter estimations.
    d_r_d_ksai_tN   =  np.hstack((np.ones(T).reshape(T,1),
                          np.abs(series_padding(a_t_arr, n, **kwargs))))
    
    d_e1_d_zeta1_tM = -np.hstack((np.ones(T).reshape(T,1),
                                 series_padding(a_t_arr, m1, **kwargs)))
    d_e2_d_zeta2_tM = -np.hstack((np.ones(T).reshape(T,1),
                                 series_padding(a_t_arr, m2, **kwargs)))
    
    ####################################################################
    ##########     Here is the START point of iterations     ###########
    ####################################################################

    r  = np.einsum('tN,Nl->tl',
                   d_r_d_ksai_tN,
                   ksai)
    # e_k(k=1,2) here is in shape of (T,1).
    e1 = a_t_arr.reshape(T,1) - np.einsum('tM,Ml->tl',
                             d_e1_d_zeta1_tM,
                             zeta1)
    e2 = a_t_arr.reshape(T,1) - np.einsum('tM,Ml->tl',
                             d_e2_d_zeta2_tM,
                             zeta2)
    
    # p_k(k=1,2) here is in shape of (T,1).
    p2 = np.reciprocal(np.exp(r)+1)
    p2 = np.clip(0,1,p2)
    p1 = 1 - p2

    # these are temporary steps to calculate out tau_k.
    numerator   = (p1/sigma1)*scipy.stats.norm.cdf(e1/sigma1)
    denominator = (p1/sigma1)*scipy.stats.norm.cdf(e1/sigma1)\
                 +(p2/sigma2)*scipy.stats.norm.cdf(e2/sigma2)
    denominator = np.where(denominator<1e-10,1e-10,denominator)
    # tau_k(k=1,2) here is in shape of (T,1).
    tau1 = numerator * np.reciprocal(denominator)
    tau1 = np.clip(0, 1, tau1)
    tau2 = 1 - tau1

    # New iteration for ksai: ksai -> ksai_iter
    d_Q_d_ksai_first_order  = np.einsum('tl,tN->Nl',
                                        tau1-p1,
                                        d_r_d_ksai_tN)
    # Eliminate "ValueError: cannot reshape array of size 9000 into shape (3,)"
    d_Q_d_ksai_second_order = -np.einsum('tl,tNn->Nn',
                                         p1*p2,
                                         np.einsum('LtN,ltn->tNn',
                                                   d_r_d_ksai_tN[np.newaxis,:],
                                                   d_r_d_ksai_tN[np.newaxis,:])
                                        )
    
    ## Error test output:
    #print(d_Q_d_ksai_second_order.shape, d_Q_d_ksai_first_order.shape, 
    #      d_r_d_ksai_tN.shape, (tau1-p1).shape, 
    #      tau1.shape, p1.shape, numerator.shape, denominator.shape,
    #      (p1/sigma1).shape,scipy.stats.norm.cdf(e1/sigma1).shape)
    
    du = np.einsum('nN,Nl->nl',
                   np.linalg.inv(d_Q_d_ksai_second_order),
                   d_Q_d_ksai_first_order)
    ksai_iter = (ksai - du).reshape(n+1)


    # New iteration for zeta1: zeta1 -> zeta1_iter
    l1 = np.linalg.inv(np.einsum('tl,tmM->mM',
                                 tau1,
                                 np.einsum('tm,tM->tmM',
                                           d_e1_d_zeta1_tM,
                                           d_e1_d_zeta1_tM)))
    r1 = -np.einsum('tl,tM->Ml',
                    tau1*a_t_arr.reshape(T,1),
                    d_e1_d_zeta1_tM)
    zeta1_iter = np.einsum('mM,Ml->ml',l1,r1)
    # New iteration for zeta2: zeta2 -> zeta2_iter
    l2 = np.linalg.inv(np.einsum('tl,tmM->mM',
                                 tau2,
                                 np.einsum('tm,tM->tmM',
                                           d_e2_d_zeta2_tM,
                                           d_e2_d_zeta2_tM)))
    r2 = -np.einsum('tl,tM->Ml',
                    tau2*a_t_arr.reshape(T,1),
                    d_e2_d_zeta2_tM)
    zeta2_iter = np.einsum('mM,Ml->ml',l2,r2)

    #####################################################################
    ####   Paper got some mistakes!!! Here we should squared staff   ####
    ####   (a_t_arr+np.einsum(...)) to get STABLE sigma numerator!   ####
    ####   Otherwise r1, r2 might be nagative values and sqrt(r/s)   ####
    ####   will result in "complex numbers", i.e.sigma_k as np.nan.  ####
    ##################################################################### 

    # New iteration for sigma1: sigma1 -> sigma1_iter
    r1 = np.einsum('tL,tl->Ll',
                   tau1,
                   (a_t_arr.reshape(T,1)+np.einsum('tm,ml->tl',
                                                  d_e1_d_zeta1_tM,
                                                  zeta1_iter))**2)
    # New iteration for sigma2: sigma2 -> sigma2_iter
    r2 = np.einsum('tL,tl->Ll',
                   tau2,
                   (a_t_arr.reshape(T,1)+np.einsum('tm,ml->tl',
                                                  d_e2_d_zeta2_tM,
                                                  zeta2_iter))**2)
    s1,s2 = np.sum(tau1),np.sum(tau2)
    s1 = np.where(s1<1e-10,1e-10,s1)
    s2 = np.where(s2<1e-10,1e-10,s2)
    # should not forget to return SQUARE-ROOT of estimated result!
    # HOW TO HANDLE SUCH CASE: nagative result of r1 or r2 (?)
    sigma1_iter = float(np.sqrt(r1 * np.reciprocal(s1)))
    sigma2_iter = float(np.sqrt(r2 * np.reciprocal(s2)))

    # should not forget to re format the zeta1, zeta2.
    zeta1_iter = zeta1_iter.reshape(m1+1)
    zeta2_iter = zeta2_iter.reshape(m2+1)

    return ksai_iter, zeta1_iter, zeta2_iter, sigma1_iter, sigma2_iter



def parameter_distance(para_arr:tuple, est_para_arr:tuple, threshold:float):
    """
    Parameters
    ----------
    para_arr     : tuple,
        ksai, zeta1, zeta2, sigma1, sigma2 for (i-1) th iteration,
        where sigma1, sigma2 are "float" and should EXCLUDE np.nan!  
    est_para_arr : tuple,
        ksai, zeta1, zeta2, sigma1, sigma2 for  (i)  th iteration,
        where sigma1, sigma2 are "float" and should EXCLUDE np.nan!
    threshold    : float
        Hyper-parameter, self set level to recognize convergence.

    Returns
    -------
    flag : bool,
        If True, then the estimation is convergent, stop iterates;
        If False, continue to iterate.
    """
    arr = np.concatenate((para_arr[0], para_arr[1], para_arr[2],
                          np.array([para_arr[3],para_arr[4]])))
    est_arr = np.concatenate((est_para_arr[0], est_para_arr[1], est_para_arr[2],
                          np.array([est_para_arr[3],est_para_arr[4]])))
    distance = np.linalg.norm(arr-est_arr)
    flag = distance < threshold
    return flag




if __name__=="__main__":
    da=np.random.randint(-100,100,size=3000)/100
    ksai=np.array([1.1,0.3,-0.4])
    zeta1=np.array([0.5,0.5])
    zeta2=np.array([0.1,-0.7,0.2])
    n,m1,m2=2,1,2
    sigma1=2.5
    sigma2=1.2
    iterations=1000
    threshold=1e-4

    s=time.time()
    for t in range(iterations):
        try:
            parameters = (ksai,zeta1,zeta2,sigma1,sigma2)
            est_parameters = parameter_estimation(da,ksai,zeta1,zeta2,sigma1,sigma2)
            if np.isnan(est_parameters[-2:]).any():
                print('NANs occured in estimating sigmas.') 
                break
            if parameter_distance(parameters, est_parameters, threshold)==True:
                print('Estimation successfully convergent.')
                break
            elif parameter_distance(parameters, est_parameters, threshold)==False:
                ksai,zeta1,zeta2,sigma1,sigma2 = est_parameters
                print(ksai)
        except:
            continue
    e=time.time()
    print(f'{round(e-s,8)} s per {iterations} iters for module \
          "parameter_estimation".')

