from __future__ import division

import numpy as np

from _npcompat import linalg_inv

def fermi(x, beta):
    """Fermi function"""
    arg = np.asarray(float(beta) * x)
    result = np.empty_like(arg)

    # Create a piece-wise function in the argument: in the asymptotic region,
    # where |Re(beta*x)| > 50, the Fermi function is approximated by a step.
    # This is to quiet the warning for invalid divisions. np.where does not
    # do the trick because it does not short-circuit. In principle, one could
    # rewrite the function in terms of the tanh, however one runs into numpy
    # bug #5518 for complex numbers.
    inner_region = np.asarray(np.abs(arg.real) < 50)
    result[...] = arg < 0
    result[inner_region] = 1/(1. + np.exp(arg[inner_region]))
    return result

def fft_from_iw(aiw, fft_len=None, axis=-1):
    """Convert Masubara to FFT convention over the doubled interval.

    >>> fft_from_iw(np.array([1, 2, 3, 4]), 12)
    array([0, 3, 0, 4, 0, 0, 0, 0, 0, 1, 0, 2])
    """
    if axis != -1:
        aiw = np.rollaxis(aiw, axis, aiw.ndim)

    niw = aiw.shape[-1]
    if niw % 2:
        raise ValueError("Number of frequencies must be even")
    iw_zero = niw // 2

    if fft_len is None:
        fft_len = 2 * niw
    elif fft_len % 4:
        raise ValueError("Illegal FFT length")

    # Only use part of the axis
    niw_fft = niw
    if niw_fft > fft_len // 2:
        niw_fft = fft_len // 2

    ahat = np.zeros(aiw.shape[:-1] + (fft_len,), aiw.dtype)
    ahat[..., 1:niw_fft:2] = aiw[..., iw_zero:iw_zero+niw_fft//2]
    ahat[..., -niw_fft+1::2] = aiw[..., iw_zero-niw_fft//2:iw_zero]
    return ahat

def iw_from_fft(ahat, niw=None, fft_len=None, axis=-1):
    """Extract the Matsubara axis from an FFT over the doubled interval.

    >>> iw_from_fft(np.array([0, 3, 0, 4, 0, 0, 0, 0, 0, 1, 0, 2]), 4)
    array([1, 2, 3, 4])
    """
    if fft_len is None:
        fft_len = ahat.shape[-1]
    elif fft_len > ahat.shape[-1]:
        raise ValueError("Forced FFT length exceeds provided length")
    if fft_len % 4:
        raise ValueError("Illegal FFT length")

    niw_fft = fft_len // 2
    if niw is None:
        niw = niw_fft
    if niw_fft > niw:
        niw_fft = niw    # Ensure that we don't copy over

    if niw % 2:
        raise ValueError("Number of frequencies must be even")
    iw_zero = niw // 2

    aiw = np.zeros(ahat.shape[:-1] + (niw,), ahat.dtype)
    aiw[..., iw_zero:iw_zero+niw_fft//2] = ahat[..., 1:niw_fft:2]
    aiw[..., iw_zero-niw_fft//2:iw_zero] = ahat[..., -niw_fft+1::2]

    if axis != -1:
        aiw = np.rollaxis(aiw, -1, axis)
    return aiw

def tau_from_fft(a, beta, axis=-1):
    """Extract the positive tau points from the FFT over the doubled array."""
    if a.shape[-1] % 2:
        raise ValueError("Illegal FFT axis")

    # This is to give the complete tau axis, i.e., from 0 ... beta, both
    # inclusive. While for the FT, this is redundant, as there is strict
    # anti-symmetry between the two points, it is useful for the later addition
    # of the tau-model.
    ntau = a.shape[-1]//2 + 1
    atau = a[..., :ntau]/beta

    if axis != -1:
        atau = np.rollaxis(atau, -1, axis)
    return atau

def fft_from_tau(atau, beta, axis=-1):
    """Convert tau values to the FFT convention"""
    if axis != -1:
        atau = np.rollaxis(atau, axis, atau.ndim)

    # The axis runs from 0, beta, both inclusive!
    ntau_cut = atau.shape[-1] - 1
    fft_len = 2 * ntau_cut

    # Here, don't fill the endpoint, otherwise the normalisation gets confused
    # (alternatively, we could anti-symmetrise over the two points).
    a = np.zeros(atau.shape[:-1] + (fft_len,), atau.dtype)
    a[..., :ntau_cut] = atau[..., :-1]

    # The FT to Matsubara is given by the inverse FFT in the numpy convention,
    # which carries a normalisation factor we have to cancel here.
    a *= 2 * beta
    return a

def fft_from_taubins(atau, beta, axis=-1):
    """Convert tau bin centres to FFT convention over the quadrupled array"""
    if axis != -1:
        atau = np.rollaxis(atau, axis, atau.ndim)

    n = atau.shape[-1]
    fft_len = 4 * n
    result = np.zeros(atau.shape[:-1] + (fft_len,), atau.dtype)
    result[..., 1:2*n:2] = atau

    # The FT to Matsubara is given by the inverse FFT in the numpy convention,
    # which carries a normalisation factor we have to cancel here.
    result *= 4 * beta
    return result

def iw_from_tau(atau, beta, niw=None, axis=-1):
    """Compute quantity on the Matsubara axis from tau values"""
    return iw_from_fft(
                np.fft.ifft(fft_from_tau(atau, beta, axis)),
                niw, None, axis)

def iw_from_taubins(atau, beta, niw=None, axis=-1):
    """Compute quantity on the Matsubara axis from tau-binned values"""
    return iw_from_fft(
                np.fft.ifft(fft_from_taubins(atau, beta, axis)),
                niw, 2 * atau.shape[axis], axis)

def tau_from_iw(aiw, beta, ntau=None, axis=-1):
    """Compute quantity on an equispaced tau grid from Matsubara axis"""
    if ntau % 2 != 1:
        raise ValueError("number of tau points must be odd (including beta)")

    return tau_from_fft(
                np.fft.fft(fft_from_iw(aiw, 2*(ntau-1), axis)),
                beta, axis)

def euler(n, x):
    """Returns the value at `x` of the Euler polynomial of order `n`"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x - 0.5
    elif n == 2:
        return (x - 1) * x
    else:
        raise NotImplementedError("Well ...")

def pmodel_tau(moments, tau, beta, axis=-1):
    """Generate model function for moments, transformed to the tau axis"""
    moments = np.asarray(moments)
    tau = np.asarray(tau)
    if tau.ndim != 1:
        raise ValueError("Expect 1D array of tau points")

    # We make use of the formula:
    #
    #     1/\beta \sum_\nu exp(-i\nu\tau) / (i\nu)^(n+1)
    #                          = (-1)^(n+1)/2 \beta^n/(n!) E_n(\tau/beta),
    #
    # where E_n(x) is the n-th Euler polynomial (this can easily be proven
    # using the identities DLMF 24.8.4 and 24.8.5.
    prefactor = -0.5
    model = np.zeros(moments.shape[1:] + tau.shape, moments.dtype)
    for n, moment in enumerate(moments):
        if n != 0:
            prefactor *= -beta/n
        model += prefactor * euler(n, tau/beta) * moment[..., None]

    if axis != -1:
        model = np.rollaxis(model, -1, axis)
    return model

def pmodel_iw(moments, iw, axis=-1):
    """Generate model function for moments on the Matsubara axis"""
    moments = np.asarray(moments)
    iw = np.asarray(iw)
    if iw.ndim != 1:
        raise ValueError("Expect 1D array of iw points")

    # Horner scheme, yo!
    model = np.zeros(moments.shape[1:] + iw.shape, complex)
    for moment in reversed(moments):
        model += moment[..., None]
        model /= 1j * iw

    if axis != -1:
        model = np.rollaxis(model, -1, axis)
    return model

def moments_from_density(step, dens, beta):
    """Get moment for a function modelling a certain density"""
    return [step, (step - 2*dens) * (2./beta)]

def moments_from_borders(zero_val, beta_val, beta):
    """Get moment for a function modelled after border values"""
    return [-zero_val - beta_val, (beta_val - zero_val) * (2./beta)]

def fmodel_tau(epsilon, tau, beta, axis=-1):
    """Generate model of type `1/(iw - epilson)` on the imaginary time axis"""
    epsilon = np.asarray(epsilon)
    tau = np.asarray(tau)
    if epsilon.shape != (epsilon.shape[0],) * 2:
        raise ValueError("Expecting square array for epsilon")
    if tau.ndim != 1:
        raise ValueError("Expect 1D array of iw points")

    tau_neg = tau < 0
    tau = np.where(tau_neg, tau + beta, tau)
    ev, eb = np.linalg.eigh(epsilon)
    occ = fermi(ev, beta)

    forward = ev > 0
    backward = ~forward
    model = np.einsum('ij,tj,jk->ikt',
                eb[:, forward],
                np.exp(-tau[...,None] * ev[forward]) * (-1 + occ[forward]),
                eb[:, forward].conj().T)
    model += np.einsum('ij,tj,jk->ikt',
                eb[:, backward],
                np.exp((beta - tau[...,None]) * ev[backward]) * -occ[backward],
                eb[:, backward].conj().T)

    model[..., tau_neg] = -model[..., tau_neg]
    if axis != -1:
        model = np.rollaxis(model, -1, axis)
    return model

def fmodel_iw(epsilon, iw, axis=-1):
    """Generate model of type `1/(iw - epilson)` on the Matsubara axis"""
    epsilon = np.asarray(epsilon)
    iw = np.asarray(iw)
    if epsilon.shape != (epsilon.shape[0],) * 2:
        raise ValueError("Expecting square array for epsilon")
    if iw.ndim != 1:
        raise ValueError("Expect 1D array of iw points")

    model = linalg_inv(iw[:,None,None] - epsilon).transpose(1,2,0)
    if axis != -1:
        model = np.rollaxis(model, -1, axis)
    return model
