import scipy.special
import numpy as np
import itertools


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 10000  # approximation of inf with some large weight
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


def get_weights(X_data, s, x, sigma=0.4):
    sample_cov_s = np.linalg.pinv(np.cov(X_data[:, s], rowvar=False))
    D_s = np.einsum(
        "ji,ji->j", np.dot(x[s] - X_data[:, s], sample_cov_s), x[s] - X_data[:, s]
    )
    D_s = np.sqrt(D_s / len(s))
    w_s = np.exp(-np.square(D_s) / (2 * (sigma ** 2)))
    return w_s


def get_weighted_mean(w_s, s, f, x, reference):
    w_s_sort_idx = np.argsort(w_s)[::-1]
    wp_sum, w_sum = 0.0, 0.0
    for idx in w_s_sort_idx[:10]:
        x_eval = reference[idx].copy()
        x_eval[s] = x[s]
        wp_sum += w_s[idx] * f(x_eval.reshape(1, -1))
        w_sum += w_s[idx]
    return wp_sum / w_sum


def kernel_shapr_opt(f, x, reference, M, sigma, masks):
    X = np.zeros((len(masks), M + 1))
    X[:, -1] = 1
    weights = np.zeros(len(masks))
    y = np.zeros(len(masks))

    for i, s in enumerate(masks):
        s = list(np.nonzero(s))
        w_s = np.ones(len(reference))
        if len(s) > 1:
            w_s = get_weights(reference, s, x, sigma)
        X[i, s] = 1
        weights[i] = shapley_kernel(M, len(s))
        y[i] = get_weighted_mean(w_s, s, f, x, reference)

    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))


def kernel_shapr(f, x, reference, M, sigma):
    X = np.zeros((2 ** M, M + 1))
    X[:, -1] = 1
    weights = np.zeros(2 ** M)
    y = np.zeros(2 ** M)

    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        w_s = np.ones(len(reference))
        if len(s) > 1:
            w_s = get_weights(reference, s, x, sigma)
        X[i, s] = 1
        weights[i] = shapley_kernel(M, len(s))
        y[i] = get_weighted_mean(w_s, s, f, x, reference)

    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))