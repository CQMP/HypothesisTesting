/*
 * Transparently wrap NFFT library, provide dummy fall-back functions.
 *
 * Author: Markus Wallerberger
 */
#ifndef NFFT_H_
#   error "Do not include this file directly, include 'nfft.hh' instead"
#endif /* NFFT_H_ */

NDFTPlan::NDFTPlan(unsigned freq_dim, unsigned nfreq, bool use_nfft)
    : use_nfft_(use_nfft),
      freq_dim_(freq_dim),
      nfreq_(nfreq),
      ntotfreq_(get_ntotfreq(freq_dim, nfreq)),
      nsample_max_(4),
      plan_(use_nfft ? create_plan(freq_dim, nfreq, nsample_max_) : nfft_plan())
{ }

NDFTPlan::NDFTPlan(const NDFTPlan &other)
    : use_nfft_(other.use_nfft_),
      freq_dim_(other.freq_dim_),
      nfreq_(other.nfreq_),
      ntotfreq_(other.ntotfreq_),
      nsample_max_(other.nsample_max_),
      plan_(use_nfft_ ? create_plan(freq_dim_, nfreq_, nsample_max_) : nfft_plan())
{ }

NDFTPlan &NDFTPlan::operator=(NDFTPlan other)
{
    // Use the copy-and-swap idiom: pass-by-value + swap function
    swap(*this, other);
    return *this;
}

NDFTPlan::~NDFTPlan()
{
    if (use_nfft_)
        nfft_finalize(&plan_);
}

void swap(NDFTPlan &left, NDFTPlan &right)
{
    using std::swap;        // using ADL

    swap(left.use_nfft_, right.use_nfft_);
    swap(left.freq_dim_, right.freq_dim_);
    swap(left.nfreq_, right.nfreq_);
    swap(left.ntotfreq_, right.ntotfreq_);
    swap(left.nsample_max_, right.nsample_max_);
    swap(left.plan_, right.plan_);  // This works because of RAII
}

void NDFTPlan::reserve(unsigned nsample_new)
{
    if (use_nfft_ && nsample_new > nsample_max_) {
        nfft_finalize(&plan_);
        nsample_max_ = nfft_next_power_of_2(nsample_new);
        plan_ = create_plan(freq_dim_, nfreq_, nsample_max_);
    }
    plan_.M_total = nsample_new;
}

void NDFTPlan::adjoint(unsigned nsample, ValueT *fhat, const ValueT *f,
                             const double *x)
{
    reserve(nsample);
    plan_.f_hat = reinterpret_cast<fftw_complex*>(fhat);
    plan_.f = reinterpret_cast<fftw_complex*>(const_cast<ValueT *>(f));
    plan_.x = const_cast<double*>(x);

    if (use_nfft_) {
        adjoint_nfft();
    } else {
        adjoint_naive();
    }
}

void NDFTPlan::adjoint_nfft()
{
    // Precompute and perform the Fourier transform using NFFT
    if (plan_.flags & PRE_ONE_PSI)
        nfft_precompute_one_psi(&plan_);
    nfft_adjoint(&plan_);

    if (MODE_DEBUG) {
        NDFTPlan naive(freq_dim_, nfreq_, false);
        naive.plan_ = plan_;
        naive.adjoint_naive();
        verify(*this, naive);
    }
}

void NDFTPlan::adjoint_naive()
{
    ValueT *fhat = reinterpret_cast<ValueT*>(plan_.f_hat);
    const ValueT *f = reinterpret_cast<ValueT*>(plan_.f);
    const double *x = plan_.x;
    const unsigned nsample = nsample_last();

    if (freq_dim_ == 1) {
        // naive implementation of the formula
        //         f_hat(iw_n) = sum_j exp(2pi i n x_j) n'(t_j)
        const ValueT twopii(0, 2*M_PI);
        const int niw = nfreq_/2;
        for (int n = -niw; n != niw; ++n) {
            ValueT f_hatn = 0, n_cplx = n;
            for (unsigned j = 0; j != nsample; ++j) {
                f_hatn += std::exp(twopii * n_cplx * x[j]) * f[j];
            }
            *(fhat++) = f_hatn;
        }
    } else if (freq_dim_ == 2) {
        // naive implementation of the formula
        //  f_hat(iw_m, iw_n) = sum_(i,j) exp(2pi i (m x_i * n x_j) f(t_i, t_j)
        const ValueT twopii(0, 2*M_PI);
        const int niw = nfreq_/2;
        for (int m = -niw; m != niw; ++m) {
            for (int n = -niw; n != niw; ++n) {
                ValueT f_hatn = 0, n_cplx = n, m_cplx = m;
                for (unsigned j = 0; j != nsample; ++j) {
                    f_hatn += std::exp(twopii * (m_cplx * x[2*j] +
                                                 n_cplx * x[2*j + 1])) * f[j];
                }
                *(fhat++) = f_hatn;
            }
        }
    } else {
        throw std::runtime_error("Naive not implemented for d > 2");
    }
}

void NDFTPlan::verify(const NDFTPlan &nfft, const NDFTPlan &naive)
{
    try {
        for (unsigned i = 0; i != nfft.ntotfreq_; ++i) {
            if (std::abs(nfft.plan_.f_hat[i] - naive.plan_.f_hat[i]) > 1e-6)
                throw VerificationError("NFFT and FFTW inconsistent");
        }
    } catch(VerificationError &) {
        for (unsigned i = 0; i != nfft.ntotfreq_; ++i) {
            fprintf(stderr, "%5d   %10.5f %10.5f   %10.5f %10.5f\n", i,
                    nfft.plan_.f_hat[i][0], nfft.plan_.f_hat[i][1],
                    naive.plan_.f_hat[i][0], naive.plan_.f_hat[i][1]);
        }
        throw;
    }
}

unsigned long NDFTPlan::get_ntotfreq(unsigned freq_dim, unsigned nfreq)
{
    unsigned tot = 1;
    while(freq_dim--)
        tot *= nfreq;
    return tot;
}

nfft_plan NDFTPlan::create_plan(unsigned freq_dim, unsigned nfreq,
                                      unsigned nsample_max)
{
    std::vector<int> nfreq_nfft(freq_dim, nfreq);
    std::vector<int> nfreq_fftw(freq_dim, nfft_next_power_of_2(nfreq));

    // This was hard-copied from WINDOW_HELP_ESTIMATEm for a Gaussian-type
    // window function and double precision
    const int window_cutoff = 13;

    // do not allocate f_hat, as we need it for the computation of the
    // NFFT.  This is because usually we need the complete result at the
    // same time for later assembly (otherwise one may set nvec to one).
    const unsigned nfft_flags = PRE_PHI_HUT | PRE_PSI | FFTW_INIT |
                                FFT_OUT_OF_PLACE;
    const unsigned fftw_flags = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

    nfft_plan plan;
    nfft_init_guru(&plan, freq_dim, &nfreq_nfft[0], nsample_max,
                   &nfreq_fftw[0], window_cutoff, nfft_flags, fftw_flags);

    // Now we set the arrays to compute the stuff
    plan.f_hat = NULL;
    return plan;
}


AdjointNDFT::AdjointNDFT(unsigned nvec, unsigned freq_dim, unsigned nfreq, bool use_nfft)
    : f_hat_(nvec * NDFTPlan::get_ntotfreq(freq_dim, nfreq), ValueT(0)),
      f_(nvec),
      x_(nvec),
      plan_(freq_dim, nfreq, use_nfft)
{ }

void AdjointNDFT::reset()
{
    for (unsigned ivec = 0; ivec != nvec(); ++ivec) {
        f_[ivec].clear();
        x_[ivec].clear();
    }
}

void AdjointNDFT::add(unsigned ivec, std::complex<double> f, double x)
{
    assert(plan_.freq_dim() == 1);
    assert(x >= -0.5 && x < 0.5);

    f_[ivec].push_back(f);
    x_[ivec].push_back(x);
}

void AdjointNDFT::add(unsigned ivec, std::complex<double> f, double x1, double x2)
{
    assert(plan_.freq_dim() == 2);
    assert(x1 >= -0.5 && x1 < 0.5);
    assert(x2 >= -0.5 && x2 < 0.5);

    f_[ivec].push_back(f);
    x_[ivec].push_back(x1);
    x_[ivec].push_back(x2);
}

void AdjointNDFT::compute()
{
    for (unsigned ivec = 0; ivec != nvec(); ++ivec) {
        // Advance to the next flavour
        std::vector<ValueT> &fcurr = f_[ivec];
        std::vector<double> &xcurr = x_[ivec];
        ValueT *fhat_curr = &f_hat_[ivec * plan_.ntotfreq()];

        // Make sure that the f and x arrays match in size
        const unsigned nsample = fcurr.size();
        assert(xcurr.size() == nsample * plan_.freq_dim());

        // Perform the adjoint
        if (nsample != 0)
            plan_.adjoint(nsample, fhat_curr, &fcurr[0], &xcurr[0]);
        else
            plan_.adjoint(nsample, fhat_curr, NULL, NULL);
    }
}


