# -*- coding: utf-8 -*-
"""
EM algorithm is working, great!

Created on Sat Apr 23 19:41:00 2022

@author: CAMARGOJ
"""
import numpy as np
import scipy.stats as st

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
    # %%
    import matplotlib.pyplot as plt
    K = 3
    nd = 3
    print(random_mvn_mix(K))
    w, mu_cov_lst = random_mvn_mix(K, nd=nd)

    # %%
    N = 10000
    X, ck = sample(w, mu_cov_lst, N)
    inds = np.cumsum(np.insert(ck[1], 0, 0.0))
    b = inds[:-1]
    e = inds[1:]
    tags = np.zeros(N)
    for k in range(K):
        tags[b[k]:e[k]] = k

    # %%
    cols = 'rgbcmy'
    plt.clf()
    for k in range(K):
        plt.scatter(*X[:2, b[k]:e[k]], s=3, alpha=0.025,
                    color=cols[k % len(cols)])

    ax = plt.gca()
    ax.set_axis_off()

    # %%
    import plotly.express as px
    from plotly.offline import plot
    fig = px.scatter_3d(x=X[0, :], y=X[1, :], z=X[2, :], color=tags)
    fig.update_traces(marker=dict(size=2), marker_coloraxis=None)
    plot(fig, filename='gmm_cloud.html')

    # %% EM iterations E-step
    # https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#E_step
    w_ref, mu_cov_ref = w, mu_cov_lst

    # starting guess
    w_est, mu_cov_lst_est = random_mvn_mix(K, nd=nd)
    w_score = np.zeros((K, N))

    # %%
    # Calculate membership probabilities w_score
    for k, mu_cov in enumerate(mu_cov_lst_est):
        w_score[k, :] = w_est[k]*st.multivariate_normal(*mu_cov).pdf(X.T)
    # normalizing
    w_score /= np.sum(w_score, axis=0)
    # result of the soft assignment using starting guess
    z_sel = np.argmax(w_score, axis=0)

    # % Now recalculate parameters
    w_est = np.sum(w_score, axis=1)/N
    for k in range(K):
        mu = (w_score[k]*X).sum(axis=1) / (w_est[k]*N)
        # not sure if cov formula is ok
        cov = (w_score[k]*(X - np.vstack(mu))) @ (X - np.vstack(mu)).T
        cov /= (w_est[k]*N)
        mu_cov_lst_est[k] = (mu, cov)

    print("w=", w_est)
    print("mu_cov_lst=", mu_cov_lst_est)

    # %%
    mu_arr = np.vstack([mu_cov[0] for mu_cov in mu_cov_lst_est]).T
    fig = px.scatter_3d(x=X[0, :], y=X[1, :], z=X[2, :], color=z_sel)
    fig.update_traces(marker=dict(size=1), marker_coloraxis=None)
    fig.add_scatter3d(mode="markers", x=mu_arr[0], y=mu_arr[1], z=mu_arr[2])
    plot(fig, filename='gmm_cloud.html')

    # %%
    #
    # TODO Plot log likelihood on the sample data? That will increase with the
    # steps
    #
    # TODO