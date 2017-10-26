/*
 * Perform delayed NDFT without external library requirements
 *
 * Author: Markus Wallerberger
 */
#ifndef _DNDFT_HH
#define _DNDFT_HH

#include <cassert>
#include <cmath>

/**
 * Deferred Non-equidistant fast discrete Fourier transforms.
 *
 * This class implements the adjoint discrete Fourier transform in the NFFT
 * convention [1]:
 *
 *         f_hat[k] = sum(j, 0, M-1) exp(2*pi*i * k * x[j]) f(x[j])
 *
 * where `N` is the number of frequencies, `x[j]` is in the semi-open interval
 * [-0.5, 0.5), `f(x)` is the function in normal space and `f_hat[k]` is the
 * `k`-th Fourier coefficient.  It does so along the lines of NFFT:
 *
 *    phi_hat[k] f_hat[k] ~= sum(l,j) exp(2*pi*i*k*l/n) phi(l/n - x[j]) f(x[j])
 *
 * with some window function `phi` chosen to minimize the transformation error.
 * However, following Staar et al. [2], it only performs the convolution of the
 * window functions immediately and defers the expensive Fourier transform and
 * division of the transformed phi function until later, thereby achieving
 * linear scaling.
 *
 * [1]: Keiner, J. et al, ACM Trans. Math. Softw. 36, 19 (2009)
 * [2]: Staar et al, J. Phys.: Conf. Ser. 402, 012015 (2012)
 */
template <typename Window>
class DeferredAdjointNDFT
{
public:
    /** Default constructor with zero frequencies */
    DeferredAdjointNDFT();

    /** Create new deferred NDFT for given window function */
    DeferredAdjointNDFT(const Window &window);

    /** Resets the convoluted array - f_conv must hold ngrid() elements */
    void reset(std::complex<double> *f_conv);

    /** Adds a point to the element - f_conv must hold ngrid() elements */
    void add(std::complex<double> *f_conv, std::complex<double> f, double x);

    /** Computes the adjoint NDFT - f_hat must hold ntotfreq() */
    void compute(std::complex<double> *f_hat, std::complex<double> *f_conv);

    /** Return number of frequencies */
    unsigned nfreq() const { return window_.N(); }

    /** Return size of the oversampled grid f_conv */
    unsigned ngrid() const { return window_.n() + 2 * window_.m(); }

    /** Return frequency dimension of the transform (1) */
    unsigned freq_dim() const { return 1; }

    /** Return window function */
    const Window &window() const { return window_; }

protected:
    /** Step 1 of computation: enforce periodicity */
    void fold_back(std::complex<double> *f_conv) const;

    /** Step 2 of computation: Fourier transform */
    void dft_naive(std::complex<double> *f_hat,
                   const std::complex<double> *f_conv) const;

    Window window_;
};

/** Convenience function augmenting and repeating the window */
template <typename Window>
double phi_full(const Window &window, double t);

/** Convenience function augmenting the transformed window */
template <typename Window>
std::complex<double> phi_hat_full(const Window &window, int k);

/**
 * Diluted Gaussian window function aka. Gaussian heat kernel.
 *
 * Given a shape parameter `b = 2*sigma/(2*sigma - 1) * m/pi`, given by:
 *
 *     phi(t) = 1./sqrt(pi * b) * exp(-b * t**2)
 *     phi_hat(k) = 1./n * exp(-1/b * (pi*k/n)**2)
 *
 * (for 1e-13 precision, you should choose at least sigma=4, m=13.)  We use
 * fast Gaussian gridding [1] to speed up the computation of the window
 * function:
 *
 *     a*exp[c(l-delta)**2)] = a*exp[c*l**2] exp[c*delta**2] exp[-2*c*delta]**l
 *
 * where the first factor can be precomputed once, the second and the base of
 * the third precomputed for every complete window insertion, and the powers
 * by (stable) recursion, thus reducing the number of (expensive) calls to
 * the exponential function from 2*m to 2.
 *
 * [1]: Greengard and Lee, SIAM Rev. 46, 443 (2004)
 */
class GaussianWindow
{
public:
    const static bool DEBUG_GRIDDING = false;

public:
    /** Constructs for delta peak */
    GaussianWindow();

    /** Window of size m on a sigma-fold oversampled grid of length N */
    GaussianWindow(unsigned m, unsigned N, unsigned sigma);

    /** Window function defined on [-m, m] */
    double phi(double t) const;

    /** Fourier transform of window function defined on [-N/2, N/2) */
    std::complex<double> phi_hat(int k) const;

    /** Do `buffer[l] += phi(l - delta) * fx` for `-m < l <= m` */
    void add_phi(std::complex<double> *buffer, double delta,
                 std::complex<double> fx) const;

    /** Window size */
    unsigned m() const { return m_; }

    /** Number of frequencies in the result */
    unsigned N() const { return N_; }

    /** Total (oversampled) grid size of  number of DFT frequencies */
    unsigned n() const { return n_; }

protected:
    unsigned m_, N_, n_;
    double tnorm_, texp_, knorm_, kexp_;
    std::vector<double> lfact_;
};

#ifndef SWIG
#   include "dndft.tcc"
#endif

#endif /* _DNDFT_HH */
