# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 19:41:00 2022

@author: CAMARGOJ
"""
import numpy as np
import mvn


def random_mvn_mix(K, nd=2, mu_sc=10.0, T_sc=10.0):
    mu_cov_lst = []
    for k in range(K):
        mu_cov = mvn.gen_random(nd, mu_sc, T_sc)
        mu_cov_lst.append(mu_cov)

    # sample w:
    w = np.random.uniform(size=K)
    w = w / w.sum()
    return w, mu_cov_lst


def sample(w, mu_cov_lst, N):
    """We don't shuffle the sampled points after sampling"""
    K = len(w)
    # sample components according to weights
    wk = np.random.choice(np.arange(K), size=N, p=w)
    sel, Nk = np.unique(wk, return_counts=True)
    res = [mvn.sample(*mu_cov, Nk) for Nk, mu_cov in zip(Nk, mu_cov_lst)]
    return np.concatenate(res, axis=1), (sel, Nk)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    K = 5
    print(random_mvn_mix(K))
    w, mu_cov_lst = random_mvn_mix(K)

    # %%
    N = 1000000
    X, ck = sample(w, mu_cov_lst, N)
    inds = np.cumsum(np.insert(ck[1], 0, 0.0))
    b = inds[:-1]
    e = inds[1:]

    plt.clf()
    for k in range(K):
        plt.scatter(*X[:, b[k]:e[k]], s=3, alpha=0.025)

    ax = plt.gca()
    ax.set_axis_off()
