/* Utility functions for QMC
 *
 * Author: Markus Wallerberger
 */
#ifndef _UTIL_HH
#define _UTIL_HH

#include <limits>
#include <vector>

#include "base.hh"

// Forward declarations

template <typename ValueT> struct InplaceLUTraits;
template <typename ValueT> class InplaceLU;
template <typename T> struct make_complex;

// Actual declarations


/** "Complexifies" the type in the spirit of `std::make_signed` etc */
template <typename T>
struct make_complex
{
    typedef std::complex<T> type;
};

template <typename T>
struct make_complex< std::complex<T> >
{
    typedef T type;
};

/** Returns the underlying data pointer of vector with fallback for C++03 */
template <typename T, typename Alloc>
inline const T *data(const std::vector<T, Alloc> &vec)
{
    #if __cplusplus >= 201103L
        return vec.data();
    #else
        return MODE_DEBUG && vec.size() == 0 ? NULL : &vec[0];
    #endif
}

/** Returns the underlying data pointer of vector with fallback for C++03 */
template <typename T, typename Alloc>
inline T *data(std::vector<T, Alloc> &vec)
{
    #if __cplusplus >= 201103L
        return vec.data();
    #else
        return MODE_DEBUG && vec.size() == 0 ? NULL : &vec[0];
    #endif
}

/**
 * Sorts a range in unstable fashion and returns the number of required swaps.
 *
 * Arguments:
 *  - begin: Beginning of range to be sorted
 *  - end:   End of range to be sorted (exclusive)
 *  - comp:  Strict ordering, defaults to: std::less
 *  - swap:  Swap two elements, defaults to: std::swap
 */
template <typename IterT, typename Comp, typename Swap>
unsigned tsort(IterT begin, IterT end, Comp comp, Swap swap);

template <typename IterT, typename Comp>
unsigned tsort(IterT begin, IterT end, Comp comp);

template <typename IterT>
unsigned tsort(IterT begin, IterT end);

/**
 * Integer power with range checks
 */
template <typename T>
T ipow(T base, T exp)
{
    double val = std::pow<double>(base, exp);
    if (val < std::numeric_limits<T>::min())
        throw std::range_error("result will be too small for int type");
    if (val > std::numeric_limits<T>::max())
        throw std::range_error("result will be too small for int type");
    if (exp < 0)
        throw std::range_error("negative exponents not supported by ipow");

    int result = 1;
    while (exp) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}

/**
 * In-place LU decompositions with partial pivoting for variable problem sizes.
 *
 * This class essentially provides a stripped-down the interface of
 * Eigen::PartialPivLU< > to work around the fact that while it allows
 * pre-allocating its buffers via the constructor, it will not allow to then
 * only parts of the buffer (it will do a reallocation instead).
 */
template <typename ValueT>
class InplaceLU
{
public:
    static const bool DEBUG_DECOMPOSE = false;

public:
    /** Instantiate new inplace decomposer */
    InplaceLU();

    /** Decomposes matrix in-place into P.L.U */
    void decompose(ValueT *matrix, unsigned size);

    /** Computes the inverse from the decomposed matrix out-of-place */
    void inverse(ValueT *out, unsigned out_stride) const;

    /** Return the determinant from the (decomposed) matrix */
    ValueT determinant() const { return determinant_; }

    /** Returns the decomposed matrix */
    ValueT *matrix() { return matrix_; }

    /** Returns the decomposed matrix */
    const ValueT *matrix() const { return matrix_; }

    /** Returns the transposition indicies */
    const long *transp() const { return &transp_[0]; }

    /** Returns the number of rows/columns of the matrix */
    const unsigned size() const { return size_; }

protected:
    typedef InplaceLUTraits<ValueT> Traits;

    ValueT *matrix_;
    std::vector<long> transp_, perm_;
    unsigned size_;
    ValueT determinant_;
};


template <typename ValueT>
class PiecewisePolynomial
{
public:
    PiecewisePolynomial()
        : order_(0),
          coeffs_()
    { }

    PiecewisePolynomial(ValueT *coeffs, unsigned n, unsigned order)
        : order_(order),
          coeffs_(coeffs, coeffs + n * order)
    {
        assert(n != 0 && order != 0);
    }

    ValueT value(double x) const
    {
        assert(x >= 0 && x < n());
        const unsigned xbase = x;
        const double xincr = x - xbase;
        const ValueT *curr = &coeffs_[xbase * order()];

        ValueT result = 0;
        for (unsigned k = order() - 1; k != 0; --k) {
            result += curr[k];
            result *= xincr;
        }
        result += curr[0];
        return result;
    }

    unsigned order() const { return order_; }

    unsigned n() const { return coeffs_.size() / order(); }

protected:
    unsigned order_;
    std::vector<ValueT> coeffs_;
};



#ifndef SWIG
#  include "util.tcc"
#endif

#endif /* _UTIL_HH */
