/*
 * Transparently wrap NFFT library, provide dummy fall-back functions.
 *
 * This header transparently wraps the NFFT 3.x interface, providing dummy
 * functions if NFFT is not available.  This allows to replace #ifdef's with
 * regular ifs.  There is no runtime cost, since every single compiler is able
 * to fold the constant and eliminate the dead path, but is still able to
 * perform static analysis on the code.
 *
 * Author: Markus Wallerberger
 */
#ifndef NFFT_H_
#define NFFT_H_

#if HAVE_NFFT

#include <nfft3.h>

#else

#include <cstdlib>
#include <complex>

class NfftNotAvailable : public std::exception
{ };

typedef double (fftw_complex)[2];

// Dummy library for NFFT3
struct nfft_plan {
    int d;
    int *N;
    int M_total;
    unsigned flags;

    double *x;
    fftw_complex *f;
    fftw_complex *f_hat;
};

// Dummy flags for NFFT3
#define PRE_ONE_PSI        1
#define PRE_PHI_HUT        1
#define PRE_PSI            1
#define MALLOC_X           1
#define MALLOC_F_HAT       1
#define MALLOC_F           1
#define FFT_OUT_OF_PLACE   1
#define FFTW_INIT          1
#define FFTW_ESTIMATE      1
#define FFTW_DESTROY_INPUT 1

#define DUMMY { throw NfftNotAvailable(); }

static inline void nfft_init(nfft_plan *p, int d, int *N, int M_total) DUMMY

static inline void nfft_init_1d(nfft_plan *p, int N, int M) DUMMY

static inline void nfft_init_2d(nfft_plan *p, int N1, int N2, int M) DUMMY

static inline void nfft_init_guru(nfft_plan *p, int d, int *N, int M, int *n,
                      int m, unsigned nfft_flags, unsigned fftw_flags) DUMMY

static inline void nfft_precompute_one_psi(nfft_plan *p) DUMMY

static inline void nfft_adjoint(nfft_plan *p) DUMMY

static inline void nfft_finalize(nfft_plan *p) DUMMY

static inline int nfft_next_power_of_2(int N) DUMMY

#undef DUMMY

#endif

#include <vector>
#include <complex>


/**
 * Generator of non-equidistant discrete Fourier transforms.
 *
 * This class provides a wrapper for the NFFT library for a set of adjoint
 * NFFTs for different sizes, as usually required from QMC:
 *
 *       fhat(n) = sum_j exp(i 2 pi n * x_j) f(x_j)     -0.5 <= x_j < 0.5
 *
 *       f(x_j) = 1/N sum_n exp(-i 2 pi n * x_j) fhat(n)
 *
 * Rationale: The problem with calling the NFFT routines directly is that
 * setting up plans, even when allocating x, f, f_hat yourself, requires a set
 * of malloc() calls that become expensive.  This class caches the NFFT plan
 * and adjusts M_total according to the current input size, re-creating the
 * plan as needed.  This provides an about 3x speedup.
 */
class NDFTPlan
{
public:
    typedef std::complex<double> ValueT;

    /** Set to true whenever the NFFT library support is compiled in */
    static const bool NFFT_AVAILABLE = HAVE_NFFT;

    /** Return number of total frequencies: nfreq ** freq_dim */
    static unsigned long get_ntotfreq(unsigned freq_dim, unsigned nfreq);

    /** Verifies the result of two NFFT plans against each other */
    static void verify(const NDFTPlan &nfft, const NDFTPlan &naive);

public:
    /**
     * Create new extensible plan for a NDFT.
     *
     * Parameters:
     *   - freq_dim: Dimension of frequency/time
     *   - nfreq:    Number of frequencies in each dimension
     *   - use_nfft: If true, use NFFT to speed up the transform
     */
    NDFTPlan(unsigned freq_dim, unsigned nfreq, bool use_nfft);

    NDFTPlan(const NDFTPlan &other);

#if __cplusplus >= 201103L
    NDFTPlan(NDFTPlan &&other) : NDFTPlan() { swap(*this, other); }
#endif

    NDFTPlan &operator=(NDFTPlan other);

    ~NDFTPlan();

    /** Prepare for N samples */
    void reserve(unsigned nsample_max);

    /** Compute adjoint NDFT and fill fhat */
    void adjoint(unsigned nsample, ValueT *fhat, const ValueT *f, const double *x);

    /** Return whether to use NFFT */
    bool use_nfft() const { return use_nfft_; }

    /** Return frequency dimension */
    unsigned freq_dim() const { return freq_dim_; }

    /** Return number of frequencies */
    unsigned nfreq() const { return nfreq_; }

    /** Return total number of frequencies */
    unsigned ntotfreq() const { return ntotfreq_; }

    /** Number of samples for the latest NFFT */
    unsigned nsample_last() const { return plan_.M_total; }

    /** Returns the underlying NFFT plan instance */
    const nfft_plan &plan() const { return plan_; }

    /** Swap data with another instance (used for copy-and-swap) */
    friend void swap(NDFTPlan &left, NDFTPlan &right);

protected:
    NDFTPlan() : use_nfft_(false) { }

    static nfft_plan create_plan(unsigned freq_dim, unsigned nfreq,
                                 unsigned nsample_max);

    void adjoint_nfft();

    void adjoint_naive();

    bool use_nfft_;
    unsigned freq_dim_, nfreq_, ntotfreq_, nsample_max_;
    nfft_plan plan_;
};


/**
 * Generator of a set of adjoint Fourier transforms
 */
class AdjointNDFT
{
public:
    typedef std::complex<double> ValueT;

    static const bool NFFT_AVAILABLE = HAVE_NFFT;

public:
    /**
     * Create new plan for adjoint NDFT.
     *
     * Parameters:
     *   - nvec:     Number of components of the vector field f(x)
     *   - freq_dim: Dimension of frequency/time
     *   - nfreq:    Number of frequencies in each dimension
     *   - use_nfft: If true, use NFFT to speed up the transform
     */
    AdjointNDFT(unsigned nvec, unsigned freq_dim, unsigned nfreq, bool use_nfft);

    /** Discard points added by add() */
    void reset();

    /** Add an additional sampling point for 1D transform */
    void add(unsigned ivec, std::complex<double> f, double x);

    /** Add an additional sampling point for 2D transform */
    void add(unsigned ivec, std::complex<double> f, double x1, double x2);

    /** Compute adjoint NDFT and fill f_hat() */
    void compute();

    /** Return fhat as `nvec * nfreq**freq_dim` array */
    const std::vector<ValueT> &f_hat() const { return f_hat_; }

    /** Return fhat as `nvec * nfreq**freq_dim` array */
    std::vector<ValueT> &f_hat() { return f_hat_; }

    /** Return stored points in f for the `ivec`-th component */
    const std::vector<ValueT> &f(unsigned ivec) const { return f_[ivec]; }

    /** Return stored points in f for the `ivec`-th component */
    std::vector<ValueT> &f(unsigned ivec) { return f_[ivec]; }

    /** Return sampling points x for the `ivec`-th component */
    const std::vector<double> &x(unsigned ivec) const { return x_[ivec]; }

    /** Return sampling points x for the `ivec`-th component */
    std::vector<double> &x(unsigned ivec) { return x_[ivec]; }

    /** Return whether to use NFFT */
    const NDFTPlan &plan() const { return plan_; }

    /** Return number of vector components */
    unsigned nvec() const { return f_.size(); }

protected:
    std::vector<ValueT> f_hat_;
    std::vector< std::vector<ValueT> > f_;
    std::vector< std::vector<double> > x_;
    NDFTPlan plan_;
};

#ifndef SWIG
#   include "nfft.tcc"
#endif

#endif  /* NFFT_H_ */
