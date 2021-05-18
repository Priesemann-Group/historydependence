import numpy as np
import mpmath as mp
from scipy.integrate import quad, simps, dblquad
from scipy.optimize import newton, minimize
#from scipy import weave

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
-----------------Functions to compute counts and multiplicities------------------
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def make_cts(data):  # Gives vector of counts for any sample consisting of numbers
    cts = []
    data_sorted = np.sort(data)
    uniq = np.unique(data_sorted)
    dat_old = data_sorted[0]
    count = 0
    for dat in data_sorted:
        if dat == dat_old:
            count += 1
        else:
            dat_old = dat
            cts += [count]
            count = 1
    cts += [count]
    return cts


def make_cts_dict(data):  # Gives vector of counts for any sample consisting of numbers
    cts = []
    data_sorted = np.sort(data)
    uniq = np.unique(data_sorted)
    dat_old = data_sorted[0]
    count = 0
    for dat in data_sorted:
        if dat == dat_old:
            count += 1
        else:
            dat_old = dat
            cts += [count]
            count = 1
    cts += [count]
    return dict(zip(uniq, cts))


def make_cts_dict_fast(datain):  # same as above but faster (but requires weave)
    """
    Similar to numpy.unique function for returning unique members of
    data, but also returns their counts
    """
    data = np.sort(datain)
    uniq = np.unique(data)
    nums = np.zeros(uniq.shape, dtype='int')

    code = """
	int i,count,j;
	j=0;
	count=0;
	for(i=1; i<Ndata[0]; i++){
		count++;
		if(data(i) > data(i-1)){
			nums(j) = count;
			count = 0;
			j++;
		}
	}
	// Handle last value
	nums(j) = count+1;
	"""
    weave.inline(code,
                 ['data', 'nums'],

                 extra_compile_args=['-O2'],
                 type_converters=weave.converters.blitz)
    return dict(zip(uniq, nums))


def make_mk(n, K):
    mk = {}
    un = np.unique(n)
    K1 = np.count_nonzero(n)
    for x in un:
        mk[x] = (n == x).sum()
    mk[0] = K - K1
    if K < K1:
        print("Choice of K not compatible with empirical histogram!")
    return mk


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
---------------------Likelihood functions for Dirichlet priors--------------------
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def _rhoi(n, beta, mk, K, N):  # Computes the ratio of gamma functions in the product
    return mp.power(mp.rf(beta, np.double(n)), mk[n])


def _rho(beta, mk, K, N):  # Computes the Dirichlet-multinomial likelihood from eq. (68)
    rn = np.array([_rhoi(n, beta, mk, K, N) for n in mk])
    return rn.prod() / mp.rf(K * beta, np.double(N))


def _dlogrho(beta, mk, K, N):  # Derivative of the logarithm of the Dirichlet-multinomial likelihood
    db = K * (mp.psi(0, K * beta) - mp.psi(0, K * beta + N)) - \
        K * mp.psi(0, beta)
    db += np.array([mk[n] * mp.psi(0, n + beta) for n in mk]).sum()
    return db


# 2nd Derivative of the logarithm of the Dirichlet-multinomial likelihood
def _d2logrho(beta, mk, K, N):
    db = K**2 * (mp.psi(1, K * beta) - mp.psi(1, K * beta + N)) - \
        K * mp.psi(1, beta)
    db += np.array([mk[n] * mp.psi(1, n + beta) for n in mk]).sum()
    return db

# NSB prior and its derivatives


def _dxi(beta, K):
    return K * mp.psi(1, K * beta + 1.) - mp.psi(1, beta + 1.)


def _d2xi(beta, K):
    return K**2 * mp.psi(2, K * beta + 1) - mp.psi(2, beta + 1)


def _d3xi(beta, K):
    return K**3 * mp.psi(3, K * beta + 1) - mp.psi(3, beta + 1)

# Posterior likelihood for the NSB estimator


def _rho_xi(beta, mk, K, N):
    return _rho(beta, mk, K, N) * _dxi(beta, K)


def _dlogrho_xi(beta, mk, K, N):
    return _dlogrho(beta, mk, K, N) + _d2xi(beta, K) / _dxi(beta, K)


def _d2logrho_xi(beta, mk, K, N):
    return _d2logrho(beta, mk, K, N) + (_d3xi(beta, K) * _dxi(beta, K) - _d2xi(beta, K)**2) / _dxi(beta, K)**2


def _beta_ML(mk, K, K1, N):  # gives ML beta from eq. (89)
    # Case where not enough coincidences were observed to find maximum likely beta
    if _dlogrho(10**1, mk, K, N) > 0:
        print("Warning, no ML parameter can be found!")
        beta_ML = 1.  # Arbitrary choice of beta, in this case the data does not allow sensible estimation anyway
    else:
        # first guess computed via likelihood of Dirichlet process simply because it's more robust. This is a different first guess as in the thesis.
        DP_est = _alpha_ML(mk, K1, N) / K
        beta_ML = newton(lambda beta: float(_dlogrho(beta, mk, K, N)), DP_est,
                         lambda beta: float(_d2logrho(beta, mk, K, N)), maxiter=250)  # Compute exact maximum
    return beta_ML


def _beta_DP(mk, K, K1, N):  # gives ML beta from Dirichlet process (faster to compute)
    # Case where not enough coincidences were observed to find maximum likely beta
    if _dlogrho(10**0, mk, K, N) > 0:
        print("Warning, no ML parameter can be found!")
        beta_ML = 10.  # Arbitrary choice of beta, in this case the data does not allow sensible estimation anyway
    else:
        # first guess computed via likelihood of Dirichlet process simply because it's more robust. This is a different first guess as in the thesis.
        DP_est = _alpha_ML(mk, K1, N) / K
    return DP_est


def _beta_MAP(mk, K, K1, N):  # same as above but computes MAP parameter, i.e. the maximum of the posterior w.r.t. NSB prior
    if _dlogrho(10**1, mk, K, N) > 0:
        print("Warning, no ML parameter can be found!")
        beta_MAP = 1.  # Arbitrary choice of beta, in this case the data does not allow sensible estimation anyway
    else:
        # first guess computed via posterior of Dirichlet process
        DP_est = _alpha_ML(mk, K1, N) / K
        beta_MAP = newton(lambda beta: float(_dlogrho_xi(beta, mk, K, N)), DP_est,
                          lambda beta: float(_d2logrho_xi(beta, mk, K, N)), tol=5e-08, maxiter=500)
    return beta_MAP


def _H1_w(w, mk, K, K1, N):
    sbeta = w / (1 - w)
    beta = sbeta * sbeta
    return _H1(beta, mk, K, K1, N)


def _rho_xi_w(w, mk, K, N):
    sbeta = w / (1 - w)
    beta = sbeta * sbeta
    return _rho(beta, mk, K, N) * _dxi(beta, K) * 2 * sbeta / (1 - w) / (1 - w)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
-------------Likelihood functions for Dirichlet process priors------------------
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# same as above but for a Dirichlet process


def _logvarrhoi_DP(n, mk):
    return mp.log(mp.rf(1., n - 1.)) * mk[n]


def _logprod_DP(a, K1):
    return (K1 - 1.) * mp.log(a)


def _logvarrho_DP(a, rnsum, K1, N):
    return rnsum + _logprod_DP(a, K1) - mp.log(mp.rf(a + 1., N - 1.))


def _dh(a):
    return mp.psi(1, a + 1)


def _logvarrho_h_DP(a, mk, K1, N):
    return _logvarrho_DP(a, mk, K1, N) + mp.log(mp.psi(1, a + 1))


def _alpha_ML(mk, K1, N):  # computes ML alpha
    mk = removekey(mk, 0)
    rnsum = np.array([_logvarrhoi_DP(n, mk) for n in mk]).sum()
    estlist = [N * (K1 - 1.) / 6. / (N - K1), N * (K1 - 1.) / 5.5 / (N - K1), N * (K1 - 1.) / 5 / (N - K1), N * (K1 - 1.) / 4.5 / (N - K1), N * (K1 - 1.) / 4. /
               (N - K1), N * (K1 - 1.) / 3.5 / (N - K1), N * (K1 - 1.) / 3. / (N - K1), N * (K1 - 1.) / 2.5 / (N - K1), N * (K1 - 1.) / 2 / (N - K1)]  # list of first guesses
    varrholist = np.zeros(len(estlist))
    for i,a in enumerate(estlist):
        varrholist[i] = _logvarrho_DP(a, rnsum, K1, N)
        # varrholist[_logvarrho_DP(a, rnsum, K1, N)] = a
    # choose the best first guess
    a_est = estlist[np.where(varrholist == np.amax(varrholist))[0][0]]
    # find ML alpha
    res = minimize(
        lambda a: -_logvarrho_DP(a[0], rnsum, K1, N), a_est, method='Nelder-Mead')
    return res.x[0]


def _alpha_MAP(mk, K1, N):  # computes MAP alpha w.r.t. flat prior on entropy
    mk = removekey(mk, 0)
    estlist = [N * (K1 - 1.) / 6. / (N - K1), N * (K1 - 1.) / 5.5 / (N - K1), N * (K1 - 1.) / 5 / (N - K1), N * (K1 - 1.) / 4.5 / (N - K1), N * (K1 - 1.) /
               4. / (N - K1), N * (K1 - 1.) / 3.5 / (N - K1), N * (K1 - 1.) / 3. / (N - K1), N * (K1 - 1.) / 2.5 / (N - K1), N * (K1 - 1.) / 2 / (N - K1)]
    varrholist = {}
    for a in estlist:
        varrholist[_logvarrho_h_DP(a, mk, K1, N)] = a
    a_est = varrholist[np.amax(varrholist.keys())]
    res = minimize(
        lambda a: -_logvarrho_h_DP(a[0], mk, K1, N), a_est, method='Nelder-Mead')
    return res.x[0]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
----------Posterior moments of entropy and Shannon information content-----------
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def _H1(beta, mk, K, K1, N):  # Computes expression from eq. (78)
    norm = N + beta * K
    H1 = mp.psi(0, norm + 1)
    for n in mk:
        H1 += -mk[n] * (n + beta) * mp.psi(0, n + beta + 1) / norm
    return H1


def _I(nj, nk, psi0norm, psi1norm):
    return (mp.psi(0, nj + 1) - psi0norm) * (mp.psi(0, nk + 1) - psi0norm) - psi1norm


def _J(nk, psi0norm, psi1norm):
    return (mp.psi(0, nk + 2) - psi0norm) * (mp.psi(0, nk + 2) - psi0norm) - psi1norm + mp.psi(1, nk + 2)


def _H2summand(index, k, mk, beta, psi0norm, psi1norm):
    nk = k + beta
    diag = (nk + 1) * nk * _J(nk, psi0norm, psi1norm)
    nondiag = nk**2 * _I(nk, nk, psi0norm, psi1norm)
    sum = mk[k] * diag + mk[k] * (mk[k] - 1) * nondiag
    for i in np.arange(index, len(mk.keys())):
        j = mk.keys()[i]
        nj = j + beta
        nondiag = nj * nk * _I(nj, nk, psi0norm, psi1norm)
        sum += 2 * mk[k] * mk[j] * nondiag
    return sum


def _H2(beta, mk, K, K1, N):  # Computes 2nd moment of entropy
    norm = N + beta * K
    psi0norm = mp.psi(0, norm + 2)
    psi1norm = mp.psi(1, norm + 2)
    H2 = 0
    index = 0
    for k in mk:
        index += 1
        H2 += _H2summand(index, k, mk, beta, psi0norm, psi1norm)
    H2 = H2 / (norm + 1) / norm
    return H2


def _h1(beta, K, N, n):  # mean of Shannon information content
    return mp.psi(0, N + beta * K) - mp.psi(0, n + beta)


def _h2(beta, K, N, n):  # 2nd moment of Shannon information content
    return (mp.psi(0, N + beta * K) - mp.psi(0, n + beta))**2 - mp.psi(1, N + beta * K) + mp.psi(1, n + beta)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
---------- Calculation of finite K entropy estimates-----------------------------
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def h_Diri(mk, K, N):  # Computes estimates for Shannon information contents. Returns a dictionary that contains the estimates, ordered by the corresponding number of counts.
    mp.pretty = True
    K1 = K - mk[0]  # number of coincidences
    beta_ML = _beta_ML(mk, K, K1, N)
    # beta_MAP=_beta_MAP(mk,K,K1,N)
    # std=np.sqrt(-_d2logrho_xi(beta_MAP,mk,K,N)**(-1)) #std of Gaussian approximation at MAP parameter
    # intbounds=[np.amax([10**(-50),beta_MAP-8*std]),beta_MAP+8*std] #Set integration bounds to pm 8std around MAP beta
    # rhonorm=mp.quadgl(lambda beta: _rho_xi(beta,mk,K,N), intbounds) #Compute normalization constant
    estimates = {}
    for n in mk:
        h_mean_ML = _h1(beta_ML, K, N, n)
        h_std_ML = mp.sqrt(_h2(beta_ML, K, N, n) - h_mean_ML**2)
        #h_mean_NSB=mp.quadgl(lambda beta: _h1(beta,K,N,n)*_rho_xi(beta,mk,K,N), intbounds)/rhonorm
        #h_std_NSB=mp.sqrt(mp.quadgl(lambda beta: _h2(beta,K,N,n)*_rho_xi(beta,mk,K,N), intbounds)/rhonorm-h_mean_NSB**2)
        # ,h_mean_NSB,h_std_NSB   #returns the ML estimate plus std for fixed ML beta and the NSB estimate	plus std over all pdfs and concentration parameters for all n that occured in the sample
        estimates[n] = h_mean_ML, h_std_ML
    return estimates


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
----------NSB and ML-beta entropy estimates-----------------------------
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# Computes estimates for the Shannon entropy. Returns ML mean and NSB mean plus std as well as the ML concentration parameter.

# def H_Diri(mk, K, N):
#     mp.pretty = True
#     K1 = K - mk[0]
#     beta_MAP = _beta_MAP(mk, K, K1, N)
#     if beta_MAP = 1.:
#
#         def f(w): return _Si(w, nxkx, N, K)
#         def g(w): return _measure(w, nxkx, N, K)
#         return quadgl(f, [0, 1]) / quadgl(g, [0, 1])
#         # std of Gaussian approximation at MAP parameter
#     std = np.sqrt(-_d2logrho_xi(beta_MAP, mk, K, N)**(-1))
#     # Set integration bounds to pm 8std around MAP beta
#     intbounds = [np.amax([10**(-50), beta_MAP - 8 * std]), beta_MAP + 8 * std]
#     rhonorm = mp.quadgl(lambda beta: _rho_xi(beta, mk, K, N),
#                         intbounds)  # Compute normalization constant
#     # H_ML=_H1(beta_ML,mk,K,K1,N) #Compute H with ML prior
#     H_nsb = mp.quadgl(lambda beta: _H1(beta, mk, K, K1, N) * _rho_xi(beta,
#                                                                      mk, K, N), intbounds) / rhonorm  # Computes NSB estimator
#     # stdH_nsb=mp.sqrt(mp.quadgl(lambda beta: _H2(beta,mk,K,K1,N)*_rho_xi(beta,mk,K,N), intbounds)/rhonorm-H_nsb*H_nsb) #Computes std of NSB estimator
#     return H_nsb


def H_Diri(mk, K, N):
    mp.pretty = True
    K1 = K - mk[0]
    beta_MAP = _beta_MAP(mk, K, K1, N)
    if beta_MAP == 1.:
        def f(w): return _rho_xi_w(w, mk, K, N)
        def g(w): return _H1_w(w, mk, K, K1, N) * _rho_xi_w(w, mk, K, N)
        H_nsb = mp.quadgl(g, [0, 1]) / mp.quadgl(f, [0, 1])
    else:
        # std of Gaussian approximation at MAP parameter
        std = np.sqrt(-_d2logrho_xi(beta_MAP, mk, K, N)**(-1))
        # Set integration bounds to pm 8std around MAP beta
        intbounds = [np.amax([10**(-50), beta_MAP - 8 * std]),
                     beta_MAP + 8 * std]

        def f(beta): return _rho_xi(beta, mk, K, N)
        def g(beta): return _H1(beta, mk, K, K1, N) * _rho_xi(beta, mk, K, N)
        H_nsb = mp.quadgl(g, intbounds) / mp.quadgl(f, intbounds)
        # Computes NSB estimator
    # stdH_nsb=mp.sqrt(mp.quadgl(lambda beta: _H2(beta,mk,K,K1,N)*_rho_xi(beta,mk,K,N), intbounds)/rhonorm-H_nsb*H_nsb) #Computes std of NSB estimator
    return H_nsb

# Computes estimates for the Shannon entropy. Returns ML mean and NSB mean plus std as well as the ML concentration parameter.


def H_NSB(mk, K, N):
    mp.pretty = True
    K1 = K - mk[0]
    beta_ML = _beta_ML(mk, K, K1, N)
    beta_MAP = _beta_MAP(mk, K, K1, N)
    # std of Gaussian approximation at MAP parameter
    std = np.sqrt(-_d2logrho_xi(beta_MAP, mk, K, N)**(-1))
    # Set integration bounds to pm 8std around MAP beta
    intbounds = [np.amax([10**(-50), beta_MAP - 8 * std]), beta_MAP + 8 * std]
    rhonorm = mp.quadgl(lambda beta: _rho_xi(beta, mk, K, N),
                        intbounds)  # Compute normalization constant
    H_ML = _H1(beta_ML, mk, K, K1, N)  # Compute H with ML prior
    H_nsb = mp.quadgl(lambda beta: _H1(beta, mk, K, K1, N) *
                      _rho_xi(beta, mk, K, N), intbounds) / rhonorm  # Computes NSB estimator
    stdH_nsb = mp.sqrt(mp.quadgl(lambda beta: _H2(beta, mk, K, K1, N) * _rho_xi(beta,
                                                                                mk, K, N), intbounds) / rhonorm - H_nsb * H_nsb)  # Computes std of NSB estimator
    return H_ML, H_nsb, stdH_nsb, beta_ML


# Computes estimates for the Shannon entropy. Returns ML mean and NSB mean plus std as well as the ML concentration parameter.
def H_beta_ML(mk, K, N):
    mp.pretty = True
    K1 = K - mk[0]
    beta_ML = _beta_ML(mk, K, K1, N)
    H_ML = _H1(beta_ML, mk, K, K1, N)  # Compute H with ML prior
    return H_ML


# Computes estimates for the Shannon entropy. Returns ML mean and NSB mean plus std as well as the ML concentration parameter.
def H_beta_ML_approx(mk, K, N):
    mp.pretty = True
    K1 = K - mk[0]
    beta_ML = _beta_DP(mk, K, K1, N)
    H_ML = _H1(beta_ML, mk, K, K1, N)  # Compute H with ML prior
    return H_ML


##### Minimal example ########
# bin_nbumber=10
#
# past = np.loadtxt("past.dat")
# N_past= len(past)
# joint = np.loadtxt("joint.dat")
# N_joint =len(joint)

# cts_past = make_cts(past)
# mk_past = make_mk(cts_past, 2**bin_number)
# H_past = H_Diri(mk_past, 2**bin_number,)
# H_past_plugin = H_plugin(cts_past)

# cts_joint = make_cts(joint)
# mk_joint = make_mk(cts_joint, 2**16)
# H_joint = H_Diri(mk_joint, 2**16, len(joint))
# H_joint_plugin = H_plugin(cts_joint)
#
#
#H_cond = H_joint -H_past
