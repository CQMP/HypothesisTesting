"""Defines methods for generating and manipulating impurity problems"""

import numpy as np
import hashlib

from ctseg_py.proc import promote_diag

def hybr_is_diagonal(vki):
    return ((vki != 0).sum(1) <= 1).all()

def hybr_from_sites(epsk, vki, ffreq, tau, beta=None):
    r"""Constructs the Hybridisation function from bath sites

    Given a set of bath sites with energy levels `epsk` and Hybridisation
    strength to the impurity `vki`, computes the Hybridisation function
    on the Matsubara axis and on the imaginary time axis
    $$
        \Delta_{ij}(i\omega) =
              \sum_k \frac {V^*_{ki} V_{kj}} {-i\omega - \epsilon_k} \\
        \Delta_{ij}(\tau) = \sum_k V^*_{ki} V_{kj}
              \frac {\exp \tau\epsilon_k} {1 + \exp \beta\epsilon_k}
    $$
    Note that the Hybridsation function is returned in the "bath picture", i.e.,
    $\Delta(\tau)>0$ and $\Im\Delta(i\omega > 0) > 0$.

    Parameters:
      - epsk     set of bath energies
      - vki      two-by-two Hybridisation matrix $V_{k,i}$, where the rows
                 correspond to bath levels and the columns to impurity sites
      - ffreq    fermionic Matsubara frequencies
      - tau      imagninary time grid
      - beta     inverse temperature
    """
    epsk = np.atleast_1d(epsk)
    vki = np.atleast_2d(vki)
    iwn = 1j*np.asarray(ffreq)
    tau = np.asarray(tau)
    if beta is None: beta = tau[-1]

    # hybriw is diagonal iff any bath site is connected to at most one impurity
    # site, i.e., iff each row of vik has at most one nonzero element
    vijk = vki.T.conj()[:,None,:] * vki.T[None,:,:]
    diag = hybr_is_diagonal(vki)
    hybriv = (vijk[:,:,None,:] / (-iwn[:,None] - epsk)).sum(-1)
    hybrtau = ((vijk/(1 + np.exp(beta * epsk)))[:,:,None,:] *
               np.exp(tau[:,None] * epsk)).sum(-1)

    return diag, hybriv, hybrtau

def udensity_values(nbands=1, u=0., v=0., j=0.):
    """Creates the U matrix for the density-density interaction"""
    udens = np.zeros((nbands, 2, nbands, 2))
    if nbands > 1:
        if v: udens[...] = v
        if j: udens -= np.reshape((j, 0, 0, j), (1, 2, 1, 2))
    # the nbands > 1 statements also set diagonal elements,  so we
    # unconditionally need to set them here
    band = np.arange(nbands)
    udens[band, :, band, :] = np.array(((0, u), (u, 0)))
    return udens.reshape(2*nbands, 2*nbands)

def ufull_kanamori(nbands=1, u=0., v=0., j=0.):
    """Creates the U matrix for the Slater-Kanamori interaction"""
    ufull = np.zeros((nbands, nbands, nbands, nbands))
    band = np.arange(nbands)
    if nbands > 1:
        iband = band[:, None]
        jband = band[None, :]
        if v:  # set inter-orbital Hubbard U terms
            ufull[iband, jband, iband, jband] = v
        if j:  # set pair-hopping and spin-flip terms
            ufull[iband, jband, jband, iband] = j
            ufull[iband, iband, jband, jband] = j
    # the nbands > 1 statements also set iband == jband elements, so we
    # unconditionally need to set them here
    ufull[band, band, band, band] = u
    return ufull

def udensity_from_ufull(ufull):
    r"""Extracts  the density-density part from a full U-matrix

    .. math::   U_{i\sigma,j\tau} = U_{ijij} - U_{ijji} \delta_{\sigma\tau}

    Example:
    --------
    >>> ufull = ufull_kanamori(2, 10.0, 1.0, 0.1)
    >>> udens = udensity_values(2, 10.0, 1.0, 0.1)
    >>> np.allclose(udens, udensity_from_ufull(ufull))
    True
    """
    nbands = ufull.shape[0]
    spdiag = 0, 1
    udens = np.zeros((nbands, 2, nbands, 2), ufull.dtype)
    udens[...] = ufull.diagonal(0, 0, 2).diagonal(0, 0, 1)[:,None,:,None]
    udens[:,spdiag,:,spdiag] -= ufull.diagonal(0, 1, 2).diagonal(0, 0, 1)
    return udens.reshape(2*nbands, 2*nbands)

def ufull_from_udensity(udens):
    """Constructs the full U-matrix for the density-density part"""
    raise NotImplementedError()

def uflavour_from_ufull(ufull):
    """Constructs the full U-matrix for the density-density part"""
    nbands = ufull.shape[0]
    uflv = np.zeros((nbands, 2) * 4, ufull.dtype)
    uflv[:, 0, :, 0, :, 0, :, 0] = ufull
    uflv[:, 0, :, 1, :, 0, :, 1] = ufull
    uflv[:, 1, :, 0, :, 1, :, 0] = ufull
    uflv[:, 1, :, 1, :, 1, :, 1] = ufull
    return uflv.reshape((2 * nbands,) * 4)

def get_ffreq(beta, n):
    return (2*np.pi/beta) * (np.arange(-n, n) + 0.5)

def get_tau(beta, n):
    return np.linspace(0, beta, n+1, endpoint=True)

class ImpurityProblem:
    @classmethod
    def compute_hash(cls, data):
        my_hash = hashlib.sha1()
        for key, arr in sorted(data.items()):
            arr = np.array(arr)
            my_hash.update("A %s %s %s\n" % (key, arr.shape, arr.dtype))
            my_hash.update(arr.tobytes())
        return my_hash.hexdigest()

    def upfold(self, qtty, axis=0):
        if axis != 0:
            qtty = np.rollaxis(qtty, axis, 0)

        full_shape = (self.nflavours,) * 2 + qtty.shape[1:]
        if self.hybr_is_diagonal:
            i = np.diag_indices(self.nflavours)
            full_qtty = np.zeros(full_shape, qtty.dtype)
            full_qtty[i, i] = qtty
        else:
            full_qtty = qtty.reshape(full_shape)

        return full_qtty

    def __init__(self, beta, u_dens, site_ham, tau_hybr, hybrtau, ffreq,
                 hybriw):
        TOL = 1e-8

        # Extracting problem from parameters
        self.beta = beta
        self.u_dens = u_dens
        self.nflavours = u_dens.shape[0]

        # Get the local levels
        self.site_ham = site_ham
        self.site_ham_is_real = (np.abs(self.site_ham.imag) < TOL).all()
        if self.site_ham_is_real:
            self.site_ham = self.site_ham.real    # suppresses warning
        else:
            print "Note: site levels have non-zero IMAGINARY part."

        site_ham_components = np.abs(self.site_ham) >= TOL
        self.site_ham_is_diagonal = (site_ham_components <= np.eye(self.nflavours)).all()
        if not self.site_ham_is_diagonal:
            print "WARNING: site levels have OFF-DIAGONAL elements. Discarded!"
        self.site_ham = promote_diag(self.site_ham.diagonal())

        # Get the hybridisation function
        self.tau_hybr = tau_hybr
        self.hybrtau = hybrtau
        self.ffreq = ffreq
        self.hybriw = hybriw

        self.hybr_is_real = (np.abs(self.hybrtau.imag) < TOL).all()
        if self.hybr_is_real:
            self.hybrtau = self.hybrtau.real    # suppresses warning
        else:
            print "Note: hybridisation function has non-zero IMAGINARY part."

        hybr_components = (np.abs(self.hybrtau) >= TOL).any(axis=-1)
        self.hybr_is_diagonal = (hybr_components <= np.eye(self.nflavours)).all()
        if self.hybr_is_diagonal:
            self.hybrtau = promote_diag(self.hybrtau.diagonal().T)
            self.hybriw = promote_diag(self.hybriw.diagonal().T)
        else:
            print "Note: hybridisation function has OFF-DIAGONAL elements."

        # Generate dict with all the physical inputs
        self.hash = self.compute_hash({
            "beta": self.beta,
            "site_ham": self.site_ham,
            "u_dens": self.u_dens,
            "hybrtau": self.hybrtau
            })
