/**
 * Implementation of delayed NDFT.
 *
 * Author: Markus Wallerberger
 */
#ifndef _DNDFT_HH
#  error "Do not include this file directly, include det.hh instead"
#endif

template <typename Window>
DeferredAdjointNDFT<Window>::DeferredAdjointNDFT()
    : window_()
{ }

template <typename Window>
DeferredAdjointNDFT<Window>::DeferredAdjointNDFT(const Window &window)
    : window_(window)
{ }

template <typename Window>
void DeferredAdjointNDFT<Window>::reset(std::complex<double> *f_conv)
{
    std::fill(f_conv, f_conv + ngrid(), 0);
}

template <typename Window>
void DeferredAdjointNDFT<Window>::add(std::complex<double> *f_conv,
                                      std::complex<double> f, double x)
{
    assert(x >= -0.5 && x < 0.5);

    // Map the interval -0.5 .. 0.5 to 0 .. 1 of the grid.  Keiner et al.'s
    // convention is to wrap the negative part over to the positive one
    if (x < 0)
        x += 1;

    // Sort of implements Staar, Alg.3, formula 1.  We do not do the
    // truncation of coefficients like they do, since I think it is wrong
    //
    // For pos being an exact integer we would actually have 2n+1 points, but
    // to avoid an if we conceptually replace t -> t - epsilon, thereby
    // eliminating the additional point (it lies epsilon outside [-m,m]).
    const double pos = window_.n() * x + window_.m();
    const unsigned base = pos;
    window_.add_phi(f_conv + base, pos - base, f);
}

template <typename Window>
void DeferredAdjointNDFT<Window>::fold_back(std::complex<double> *f_conv) const
{
    for (unsigned i = 0; i != window_.m(); ++i)
        f_conv[i + window_.n()] += f_conv[i];
    for (unsigned i = window_.m(); i != 2 * window_.m(); ++i)
        f_conv[i] += f_conv[i + window_.n()];
}

template <typename Window>
void DeferredAdjointNDFT<Window>::dft_naive(std::complex<double> *f_hat,
                                    const std::complex<double> *f_conv) const
{
    // naive implementation of the formula
    //         f_hat[k] = sum_j exp(2pi i/N n j) f_conv[j]
    const std::complex<double> twopii_over_n(0, 2*M_PI/window_.n());
    const int niw = nfreq()/2;
    for (int k = -niw; k != niw; ++k) {
        std::complex<double> f_hatk = 0;
        for (int j = 0; j != int(window_.n()); ++j) {
            f_hatk += std::exp(twopii_over_n * std::complex<double>(k * j))
                      * f_conv[j];
        }
        // perform inverse convolution with phi_hat
        f_hatk /= std::complex<double>(window_.n()) * window_.phi_hat(k);
        *(f_hat++) = f_hatk;
    }
}

template <typename Window>
void DeferredAdjointNDFT<Window>::compute(std::complex<double> *f_hat,
                                          std::complex<double> *f_conv)
{
    // Catch the trivial case
    if(window_.n() == 0)
        return;

    fold_back(f_conv);
    dft_naive(f_hat, f_conv + window_.m());
}

template <typename Window>
double phi_full(const Window &window, double t)
{
    t = std::fmod(std::abs(t) + window.n()/2., window.n()) - window.n()/2.;
    return t < -int(window.m()) || t > window.m() ? 0 : window.phi(t);
}

template <typename Window>
std::complex<double> phi_hat_full(const Window &window, int k)
{
    return (k < -int(window.N()/2) || k >= int((window.N() + 1)/2))
            ? 0 : window.phi_hat(k);
}


GaussianWindow::GaussianWindow()
    : m_(0),
      N_(0),
      n_(0),
      tnorm_(INFINITY),
      texp_(0),
      knorm_(INFINITY),
      kexp_(0)
{ }

GaussianWindow::GaussianWindow(unsigned m, unsigned N, unsigned sigma)
    : m_(m),
      N_(N),
      n_(N * sigma),
      lfact_(m + 1)
{
    // Runtime checks
    if (sigma == 0)
        throw std::invalid_argument("sigma must be positive");
    if (m_ > n_)
        throw std::invalid_argument("window cannot exceed DFT length");

    // Follows Staar et al (2012), eq. (C.1)
    double b = 2. * sigma/(2 * sigma - 1) * m_/M_PI;
    tnorm_ = 1./std::sqrt(M_PI * b);
    texp_ = -1./b;
    knorm_ = 1./n_;
    kexp_ = -b * M_PI * M_PI/(1. * n_ * n_);

    // for -m < l <= m, precompute first term of:
    // a*exp[c(l-delta)**2)] = a*exp[c*l**2] exp[c*delta**2] exp[-2*c*delta]**l
    for (unsigned l = 0; l != m + 1; ++l) {
        lfact_[l] = tnorm_ * std::exp(texp_ * l * l);
    }
}

double GaussianWindow::phi(double t) const
{
    assert(t >= -int(m_) && t <= m_);
    return tnorm_ * std::exp(texp_ * t * t);
}

std::complex<double> GaussianWindow::phi_hat(int k) const
{
    assert(k >= -int(N_/2) && k < int((N_ + 1)/2));
    return knorm_ * std::exp(kexp_ * k * k);
}

void GaussianWindow::add_phi(std::complex<double> *buffer, double delta,
                             std::complex<double> fx) const
{
    assert(delta >= 0 && delta < 1);

    // That is what we are trying to do more efficiently
    if (DEBUG_GRIDDING) {
        for (int l = -int(m_) + 1; l != int(m_) + 1; ++l)
            buffer[l] += fx * phi(l - delta);
        return;
    }

    // precompute second and base of third term of:
    // a*exp[c*(l-delta)**2)] = a*exp[c*l**2] exp[c*delta**2] exp[-2*c*delta]**l
    const std::complex<double> zeroval = fx * std::exp(texp_ * delta * delta);
    const double step = std::exp(-2. * texp_ * delta);

    // now do: buffer[l] += phi(l-delta) * fx for -m < l <= m
    std::complex<double> val = zeroval;
    for (unsigned l = 0; l != m_ + 1; ++l) {
        buffer[l] += val * lfact_[l];
        val *= step;
    }
    val = zeroval;
    for (unsigned l = 1; l != m_; ++l) {
        val /= step;
        buffer[-int(l)] += val * lfact_[l];
    }
}
