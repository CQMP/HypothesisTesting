/**
 * Module for rank-k updates of determinants and inverse matrices.
 *
 * Author: Markus Wallerberger
 */
#ifndef _DET_HH
#define _DET_HH

#include <cmath>

#include "base.hh"
#include "util.hh"

// Forward declarations

template <typename ValueT> class Determinant;
template <typename ValueT> struct DeterminantTraits;
template <typename ValueT> class DetMove;
template <typename ValueT> class DetAppendMove;
template <typename ValueT> class DetRemoveMove;
template <typename ValueT> class DetSetMove;

// We must declare templated swap functions before, otherwise C++ gets confused

template <typename ValueT>
void swap(Determinant<ValueT> &left, Determinant<ValueT> &right);

template <typename ValueT>
void swap(DetMove<ValueT> &left, DetMove<ValueT> &right);

/**
 * Fast updates of the determinant and the inverse of a matrix.
 *
 * Given a `n` times `n` square matrix `A`, this class stores its determinant
 * in `weight` and its inverse in `invmat`.  It will update these quantities
 * for rank-1 operations on `A` (like inserting a row and a column).  These
 * updates can be performed more efficiently (in `n**2` time) than naive
 * updates of `A` followed by the calculation of the inverse and the
 * determinant (both `n**3` time). Note that `A` itself is not stored, but is
 * usually not needed.
 */
template <typename ValueT = double>
class Determinant
{
public:
    static const unsigned DEFAULT_CAPACITY = 8, CAPACITY_RESERVE = 1;

public:
    /** Allocate a new determinant instance with given capacity */
    Determinant(unsigned capacity = DEFAULT_CAPACITY);

    /** Copy a determinant from a different instance */
    Determinant(const Determinant &other);

#if __cplusplus >= 201103L
    /** Move a determinant from an rvalue instance */
    Determinant(Determinant &&other) : Determinant(0) { swap(*this, other); }
#endif

    /** Assign a determinant from another instance */
    Determinant &operator=(Determinant other);

    /** Frees associated buffers of determinant instance */
    ~Determinant();

    /** Verify that the stored determinant matches the stored inverse */
    void verify(double tol = 1e-6) const;

    /** Return the number of rows (or columns) of the stored matrix */
    unsigned order() const { return order_; }

    /** Return the current capacity of the matrix buffer */
    unsigned capacity() const { return capacity_; }

    /** Reserve buffer space for a `capacity` times `capacity` matrix */
    void reserve(unsigned capacity);

    /** Return the determinant of the matrix (not of the stored inverse) */
    ValueT weight() const { return weight_; }

    /** Return the absolute value of the determinant for use in QMC */
    double abs_weight() const { return std::abs(weight()); }

    /** Return a pointer to the inverse matrix (the matrix stored) */
    ValueT *invmat() { return invmat_buffer_; }

    /** Return a pointer to the inverse matrix (the matrix stored) */
    const ValueT *invmat() const { return invmat_buffer_; }

    /** Fast swap of elements from two determinant instances */
    friend void swap<>(Determinant &left, Determinant &right);

protected:
    typedef DeterminantTraits<ValueT> Traits;

    ValueT weight_;
    unsigned order_, capacity_;
    ValueT *invmat_buffer_;

    friend class DetAppendMove<ValueT>;
    friend class DetRemoveMove<ValueT>;
    friend class DetSetMove<ValueT>;
};

/**
 * Base class for determinant move generator.
 *
 * Allocates the result buffers for rank-`k` update's intermediate results
 * (since they must be aligned and we do not care about retaining the results
 * when resizing the buffer, we use flat arrays in place of vectors).
 */
template <typename ValueT = double>
class DetMove
{
public:
    DetMove(Determinant<ValueT> &target, unsigned max_rank);

    DetMove(const DetMove &other);

#if __cplusplus >= 201103L
    DetMove(DetMove &&other) : DetMove() { swap(*this, other); }
#endif

    DetMove &operator=(DetMove other);

    ~DetMove();

    unsigned rank() const { return rank_; }

    unsigned max_rank() const { return max_rank_; }

    unsigned capacity() const { return capacity_; }

    const ValueT *rbar() const { return rbar_buffer_; }

    const ValueT *cbar() const { return cbar_buffer_; }

    const ValueT *sbar() const { return sbar_buffer_; }

    friend void swap<>(DetMove &left, DetMove &right);

    const Determinant<ValueT> &target() const { return *target_; }

    ValueT ratio() const { return ratio_; }

    bool hard_reject() const { return false; }

protected:
    DetMove();

    void reserve();

    Determinant<ValueT> *target_;
    unsigned max_rank_, capacity_, rank_;
    ValueT *rbar_buffer_, *cbar_buffer_, *sbar_buffer_;
    ValueT ratio_;
};

/**
 * Generate moves that append rows and columns to the inverse.
 *
 * Given a n-by-n matrix `A` (stored as the inverse `A**1` in `invmat`), the
 * k-th order move is equivalent to enlarging the matrix by a n-by-k row-like
 * vector `row`, a k-by-n column-like vector `col` and k-by-k square matrix
 * `star`, like so:
 *
 *                                ( A      col  )
 *                        A   ->  (             )
 *                                ( row    star )
 *
 * and updating the inverse and the determinant accordingly.  Instead of the
 * naive version, which scales as `O(n**3)`, the Woodbury matrix identity is
 * used, which reduces the scaling to `O(n**2)`.
 *
 * Note that insertion of rows and columns at arbitrary positions is not
 * supported for efficiency, however can easily be realised by maintaining a
 * permutation array.
 */
template <typename ValueT = double>
class DetAppendMove
        : public DetMove<ValueT>
{
public:
    typedef ValueT Value;

public:
    DetAppendMove(Determinant<ValueT> &target, unsigned max_rank)
            : DetMove<ValueT>(target, max_rank)
    { }

    void propose(unsigned rank, ValueT *row, ValueT *col, ValueT *star);

    void accept();

    int perm_sign() const { return 1; }

protected:
    typedef DeterminantTraits<ValueT> Traits;

    ValueT *row_;
};

/**
 * Generate moves that remove rows and columns from the inverse.
 *
 * Given a n-by-n inverse matrix `M`, the k-th order move is equivalent to
 * first exchanging the `rowno[i]`'th row with the last but (k-i)'th row (we
 * denote this by the row permutation matrix `R`), and the `colno[j]`'th
 * column with the last but (k-i)'th column (column permutation matrix `C`),
 * followed by the removal of these rows and columns from the inverse, like so:
 *
 *                          ( A      col  )
 *                        R (             ) C  ->  A
 *                          ( row    star )
 *
 * and updating the inverse and the determinant accordingly.  Instead of the
 * naive version, which scales as `O(n**3)`, the Woodbury matrix identity is
 * used, which reduces the scaling to `O(n**2)`.
 *
 * Note that this is *not* the same as simply removing the rows/columns, i.e.,
 * the rows following the removed rows do not shift upwards.  Rather, the
 * removed row/column is replaced by the last one, since this can be done more
 * efficiently.  A true removal can be realised by maintaining a permutation
 * array.
 */
template <typename ValueT = double>
class DetRemoveMove
        : public DetMove<ValueT>
{
public:
    typedef ValueT Value;

    static const bool DEBUG_REPL = false;

    static unsigned replacements(unsigned *repl, const unsigned *target,
                                 unsigned order, unsigned rank);

public:
    DetRemoveMove(Determinant<ValueT> &target, unsigned max_rank)
            : DetMove<ValueT>(target, max_rank),
              rowrepl_(max_rank, -1),
              colrepl_(max_rank, -1)
    { }

    void verify_args(unsigned rank, unsigned *rowno, unsigned *colno) const;

    void propose(unsigned rank, unsigned *rowno, unsigned *colno);

    void accept();

    int perm_sign() const { assert(perm_sign_); return perm_sign_; }

    // Note that these switch row <-> col to convert internal representation

    const unsigned *rowno() const { return colno_; }

    const unsigned *rowrepl() const { return &colrepl_[0]; }

    const unsigned *colno() const { return rowno_; }

    const unsigned *colrepl() const { return &rowrepl_[0]; }

protected:
    typedef DeterminantTraits<ValueT> Traits;

    int perm_sign_;
    unsigned *rowno_, *colno_;
    std::vector<unsigned> rowrepl_, colrepl_;
};

/**
 * Generate moves that overrides the determinant with another matrix.
 */
template <typename ValueT = double>
class DetSetMove
{
public:
    typedef ValueT Value;

public:
    DetSetMove(Determinant<ValueT> &target, unsigned max_rank)
            : target_(&target)
    { }

    void verify_args(unsigned new_order, ValueT *new_mat) const { }

    void propose(unsigned new_order, ValueT *new_mat);

    void accept();

    const Determinant<ValueT> &target() const { return *target_; }

    bool hard_reject() const { return false; }

    ValueT ratio() const { return new_mat_.determinant()/target().weight(); }

    int perm_sign() const { return 1; }

    unsigned order() const { return new_mat_.size(); }

    unsigned rank() const { return 0; }

    const ValueT *mat() const { return new_mat_.matrix(); }

protected:
    typedef DeterminantTraits<ValueT> Traits;

    Determinant<ValueT> *target_;
    InplaceLU<ValueT> new_mat_;
};


// Auxiliary functions

template <typename ValueT>
static ValueT *aligned_new(unsigned size);

template <typename ValueT>
static void aligned_free(ValueT *what);


#ifndef SWIG
    #include "det.tcc"
#endif

#endif /* _DET_HH */
