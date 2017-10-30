from __future__ import division
from warnings import warn

import numpy as np
import scipy.stats as stats

def bootstrap_sample(est, *x, **kwds):
    """Given an estimator, transform x to the bootstrap sample

    Arguments:
      - `est`: Estimator function
      - `x`: Arguments to estimator function, in batches
      - `nsample=10000`: size of bootstrap sample to return
      - `preaggregate=True`: if True, estimator is a derived function that
                             expects means as input, if false, whole sample.
      - `rng=RandomState(0)`: random number generator
    """
    if not x:
        raise ValueError("must give at least one argument array")
    x = map(np.asarray, x)

    # Parameter checks
    norig = x[0].shape[0]
    if any(xi.shape[0] != norig for xi in x[1:]):
        raise ValueError("Number of samples disagree along axis")
    nsample = kwds.get('nsample', 10000)
    preaggregate = kwds.get('preaggregate', True)

    # Generate the sample indices for later bootstrap
    try:
        rng = kwds['rng']
    except KeyError:
        rng = np.random.RandomState(0)
    indices = rng.randint(0, norig, (nsample, norig))

    # Compute sample
    if preaggregate:
        sample = [est(*(xi[curr_indices].mean(0) for xi in x))
                  for curr_indices in indices]
    else:
        sample = [est(*(xi[curr_indices] for xi in x))
                  for curr_indices in indices]

    return np.asarray(sample)

def pseudovalues(est, *x, **kwds):
    """Given an estimator function theta, transform x to their pseudovalues"""
    if not x:
        raise ValueError("must give at least one argument array")
    x = map(np.asarray, x)

    axis = kwds.get('axis', 0)
    if axis < 0:
        axis += x[0].ndim
    selaxis = (slice(None),) * axis

    n = x[0].shape[axis]
    if any(item.shape[axis] != n for item in x):
        raise ValueError("argument need consistent shape along given axis")

    J = np.arange(n)
    result = np.repeat((n * est(*x))[np.newaxis], n, 0)
    for i in range(n):
        xremi = (xitem[selaxis + (J != i,)] for xitem in x)
        result[i] -= (n - 1) * est(*xremi)

    result = np.rollaxis(result, 0, axis + 1)
    return result

def get_covariance(batches, exact, axis=-1):
    if axis != -1:
        batches = np.rollaxis(batches, axis, batches.ndim)
    if batches.shape[:-1] != exact.shape:
        raise ValueError("Each batch must match exact result in shape")

    exact = exact.ravel()
    batches = batches.reshape(exact.size, -1)

    diff = batches.mean(1) - exact
    cov = np.cov(batches, ddof=1)
    return diff, cov

def get_pooled_covariance(batches_x, batches_y, axis=-1):
    if axis != -1:
        batches_x = np.rollaxis(batches_x, axis, batches_x.ndim)
        batches_y = np.rollaxis(batches_y, axis, batches_y.ndim)
    if batches_x.shape[:-1] != batches_y.shape[:-1]:
        raise ValueError("Shape mismatch")

    n_x = batches_x.shape[-1]
    batches_x = batches_x.reshape(-1, n_x)
    cov_x = np.cov(batches_x, ddof=1)

    n_y = batches_y.shape[-1]
    batches_y = batches_y.reshape(-1, n_y)
    cov_y = np.cov(batches_y, ddof=1)

    diff = batches_x.mean(1) - batches_y.mean(1)
    cov = (n_x * cov_x + n_y * cov_y)/(n_x + n_y - 2)
    return diff, cov

def diag_covariance(diff, cov, evtol=1e-14):
    ndata, = diff.shape
    if cov.shape != (ndata, ndata):
        raise ValueError("cov is not a covariance matrix w.r.t. to diff")
    if not np.allclose(cov.conj().T, cov):
        raise ValueError("cov is not Hermitean")

    # diagonalize covariance matrix and discard small eigenvalues, removing
    #   - data points that are just copies of other (perfect correlation)
    #   - data points that are forced to a value by symmetry (zero variance)
    ev, eb = np.linalg.eigh(cov)
    assert np.allclose(ev.imag, 0)
    ev = ev.real
    if (ev < -evtol).any():
        warn("Covariance matrix is not positive semidefinite", UserWarning, 2)
    ndata_dep = ev.searchsorted(evtol)

    # Project difference vector and covariance matrix to independent subspace
    proj = eb[:, ndata_dep:]
    var_proj = ev[ndata_dep:]
    cov_test = np.einsum('ij,j,kj->ik', proj, var_proj, proj.conj())
    if not np.allclose(cov_test, cov):
        warn("Covariance matrix not reproduced", UserWarning, 2)

    # Return the projected quantities
    diff_proj = (proj.conj().T).dot(diff)
    return diff_proj, var_proj

def cov_normalise(cov):
    nfact = np.sqrt(cov.diagonal()[:,None] * cov.diagonal()[None,:])
    corr = cov / nfact
    corr[nfact == 0] = 1.0
    assert (np.abs(corr) <= 1).all()
    return corr

def tsquared_score(diff, var, nbatches):
    ndata, = diff.shape
    if nbatches < 1:
        raise ValueError("must have at least one batch")
    if var.shape != (ndata,):
        raise ValueError("var is not variances w.r.t. to diff")
    if (var <= 0).any() or not np.allclose(var.imag, 0):
        raise ValueError("variances must be positive and real")

    tsquared = nbatches * (diff.conj() * diff / var.real).sum()
    assert np.allclose(tsquared.imag, 0)
    tsquared = tsquared.real

    # Check that the degrees of freedom are enough
    dof = nbatches - ndata
    if dof < 3:
        raise RuntimeError("Need three degrees of freedom for Hotelling test")
    if dof < ndata:
        warn("Low number of degrees of freedom", UserWarning, 2)

    # Compute p-values by two one-tailed tests with the F-distribution
    tsquared_n = dof/(ndata*(nbatches - 1.)) * tsquared
    fdist = stats.f(ndata, dof)
    pval_upper = fdist.sf(tsquared_n)
    pval_lower = 1. - pval_upper
    return {'t2': tsquared,
            't2_normed': tsquared_n,
            'pupper': pval_upper,
            'plower': pval_lower,
            'ndata': ndata,
            'dof': dof,
            'fdist': fdist}

def tsquared_symm_score(diff, pooled_var, nbatches_x, nbatches_y):
    ndata, = diff.shape
    if nbatches_x < 1 or nbatches_y < 1:
        raise ValueError("must have at least one batch")
    if pooled_var.shape != (ndata,):
        raise ValueError("var is not variances w.r.t. to diff")
    if (pooled_var <= 0).any() or not np.allclose(pooled_var.imag, 0):
        raise ValueError("variances must be positive and real")

    nbatches = nbatches_x + nbatches_y
    nbatches_weighed = nbatches_x * nbatches_y/(nbatches_x + nbatches_y)
    tsquared = nbatches_weighed * (diff.conj() * diff / pooled_var.real).sum()
    assert np.allclose(tsquared.imag, 0)
    tsquared = tsquared.real

    # Check that the degrees of freedom are enough
    dof = nbatches - ndata - 1
    if dof < 3:
        raise RuntimeError("Need three degrees of freedom for Hotelling test")
    if dof < ndata:
        warn("Low number of degrees of freedom", UserWarning, 2)

    # Compute p-values by two one-tailed tests with the F-distribution
    tsquared_n = dof/(ndata*(nbatches - 2.)) * tsquared
    fdist = stats.f(ndata, dof)
    pval_upper = fdist.sf(tsquared_n)
    pval_lower = 1. - pval_upper
    return {'t2': tsquared,
            't2_normed': tsquared_n,
            'pupper': pval_upper,
            'plower': pval_lower,
            'ndata': ndata,
            'dof': dof,
            'fdist': fdist}
