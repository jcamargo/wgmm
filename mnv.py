# -*- coding: utf-8 -*-
"""
Functions for marginalizing and conditioning in a gaussian distribution.

A multivariate gaussian or MVN can be represented with mu and cov matrix
and there is support in numpy and scipy for it.

Marginalizing means ignoring some of the variables of the MVN.
Conditioning means finding another MVN for the remaining variables once
some of them were fixed. These functionalities are on the wishlist, but
currently not implemented by scipy.

The first one is trivial, while the second uses the Schur complement.

Created on Wed Nov 17 14:35:54 2021

@author: CAMARGOJ
"""
import numpy as np

def _mask_cast(mask_keep, sz):
    if len(mask_keep) < sz:
        # Assume inds
        inds = mask_keep
        mask_keep = np.zeros(sz)
        mask_keep[inds] = True

    return np.array(mask_keep).astype(bool)


# %% marginalizing gaussian
def marginal(mean, cov, mask_keep):
    """ Given mean and a covariance matrix representing the joint multivariate
    gaussian for all variables, finds the mean and cov for the variables
    we want to keep"""
    mask_keep = _mask_cast(mask_keep, len(mean))
    return mean[mask_keep], cov[:, mask_keep][mask_keep]


# %% conditioning gaussian
def conditional(mean, cov, mask_keep, vals_cond):
    """ Finds mean and cov for the variables we want to keep conditioned to
    the other variables assuming provided values.
    """
    mask_keep = _mask_cast(mask_keep, len(mean))

    # block matrix notation
    mu1 = mean[~mask_keep]
    mu2 = mean[mask_keep]

    msg = 'Conditional must receive values of all the previous variables'
    assert(len(vals_cond)==len(mu1)), msg

    sigma11 = cov[:, ~mask_keep][~mask_keep]
    sigma12 = cov[:, mask_keep][~mask_keep]
    sigma21 = cov[:, ~mask_keep][mask_keep]
    sigma22 = cov[:, mask_keep][mask_keep]

    # no optimizations for the moment, use inv, check that code is correct
    # Schur complement
    res_a = vals_cond - mu1
    sigma11_inv = np.linalg.inv(sigma11)
    mu_cond = mu2 + sigma21 @ sigma11_inv @ res_a

    sigma_cond = sigma22 - sigma21 @ sigma11_inv @ sigma12
    return mu_cond, sigma_cond


def random_mvn(nd=2, mu_sc=10.0, T_sc=10.0):
    """Generates a random mvn for testing"""
    mu = mu_sc*(np.random.uniform(size=nd) - 0.5)
    T = T_sc*(np.random.uniform(size=(nd, nd)) - 0.5)
    cov = T.T @ T
    return (mu, cov)


def random_mvn_mix(K, nd=2, mu_sc=10.0, T_sc=10.0):
    mu_cov_lst = []
    for k in range(K):
        mu_cov = random_mvn(nd, mu_sc, T_sc)
        mu_cov_lst.append(mu_cov)

    # sample w:
    w = np.random.uniform(size=K)
    w = w / w.sum()
    return (w, mu_cov_lst)


# %%
def ts_plot(mean, cov, vals=None, mask=None, sc=1.0):
    """ Plots the multivariate gaussian as a plot on time plus confidence
    region.

    It is possible to provide known values, for instance from the
    first time-steps. This will be used for conditioning the mvn.

    If mask not provived we will assume the provided values are values for the
    first dimensions
    """
    import matplotlib.pyplot as plt
    mean = mean.copy()
    cov_diag = np.diag(cov).copy()

    if vals is not None:
        if mask is None:
            mask = np.zeros_like(mean).astype(bool)
            mask[:len(vals)] = True

        # conditioning
        mu_cond, cov_cond = conditional(mean, cov, ~mask, vals)

        # For the provided values there is no uncertainty anymore
        mean[mask] = vals
        mean[~mask] = mu_cond
        # For the ts plotting we only need the diag, it could be optimized
        cov_diag[mask] = 0.0
        # print('prev:', cov_diag[~mask])
        cov_diag[~mask] = np.diag(cov_cond)
        # print('post:', cov_diag[~mask])

    # for gaussian univariate
    ci = 1.96*np.sqrt(np.abs(sc**2 * cov_diag))
    mean = sc*mean

    # Now plot with conf bands
    plt.plot(mean)
    plt.fill_between(mean.index, (mean-ci), (mean+ci), color='b', alpha=.1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    N = 100000
    M = N//10
    nd = 10

    mu, cov = random_mvn(nd)
    X = np.random.multivariate_normal(mu, cov, N).T
    eps = 3e1

    # %% Gaussian mixture
    K = 2

    # %% Test conditioning with All except k
    for k in range(nd):
        Xk = X[k][0]
        mask_keep = np.ones(nd).astype(bool)
        mask_keep[k] = False
        mu_k, cov_k = conditional(mu, cov, mask_keep, [Xk])

        # Validate with rejection sampling
        mask = np.abs(X[k] - Xk) < 1e-1
        X_sel = X[mask_keep][:, mask]

        cov_emp_k = np.cov(X_sel)
        mu_emp_k = np.mean(X_sel, axis=1)

        res_mu = np.abs(mu_k - mu_emp_k)
        res_cov = np.abs(cov_k - cov_emp_k)
        print("Max res, cov", np.max(res_cov))

        assert np.all(np.isclose(mu_k, mu_emp_k, atol=eps))
        assert np.all(np.isclose(cov_k, cov_emp_k, atol=eps))
