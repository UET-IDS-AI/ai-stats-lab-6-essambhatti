import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    # Validation
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be empty")
    if not (0 < theta < 1):
        raise ValueError("Theta must be in (0,1)")

    data = np.array(data)

    if not np.all(np.isin(data, [0, 1])):
        raise ValueError("Data must contain only 0 and 1")

    # Log-likelihood
    return np.sum(data * np.log(theta) + (1 - data) * np.log(1 - theta))


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    # Validation
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be empty")

    data = np.array(data)

    if not np.all(np.isin(data, [0, 1])):
        raise ValueError("Data must contain only 0 and 1")

    # Counts
    num_successes = int(np.sum(data))
    n = len(data)
    num_failures = n - num_successes

    # MLE (mean)
    mle = num_successes / n

    # Candidates
    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    log_likelihoods = {}
    best_candidate = None
    best_ll = -np.inf

    for theta in candidate_thetas:
        ll = bernoulli_log_likelihood(data, theta)
        log_likelihoods[theta] = ll

        if ll > best_ll:
            best_ll = ll
            best_candidate = theta

    return {
        'mle': mle,
        'num_successes': num_successes,
        'num_failures': num_failures,
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }


def poisson_log_likelihood(data, lam):
    # Validation
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be empty")
    if lam <= 0:
        raise ValueError("Lambda must be > 0")

    data = np.array(data)

    if not np.all((data >= 0) & (data == data.astype(int))):
        raise ValueError("Data must be nonnegative integers")

    # Log-likelihood
    ll = 0.0
    for x in data:
        ll += x * math.log(lam) - lam - math.lgamma(x + 1)

    return ll


def poisson_mle_analysis(data, candidate_lambdas=None):
    # Validation
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be empty")

    data = np.array(data)

    if not np.all((data >= 0) & (data == data.astype(int))):
        raise ValueError("Data must be nonnegative integers")

    # Stats
    n = len(data)
    total_count = int(np.sum(data))
    sample_mean = total_count / n

    # MLE
    mle = sample_mean

    # Candidates
    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    log_likelihoods = {}
    best_candidate = None
    best_ll = -np.inf

    for lam in candidate_lambdas:
        ll = poisson_log_likelihood(data, lam)
        log_likelihoods[lam] = ll

        if ll > best_ll:
            best_ll = ll
            best_candidate = lam

    return {
        'mle': mle,
        'sample_mean': sample_mean,
        'total_count': total_count,
        'n': n,
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }