/**
 * Implementation of rank-k updates of determinants and inverse matrices.
 *
 * Author: Markus Wallerberger
 */
#ifndef _DET_HH
#  error "Do not include this file directly, include det.hh instead"
#endif

#include <cassert>
#include <cstring>
#include <iostream>

/* --------------------------  EIGEN WRAPPING  --------------------------- */

// The Eigen dependencies are moved to the implementation file.  This is more
// of a cosmetic change than a semantic one, since the det.hh file always needs
// to pull the implementation.

#include <complex>
#include <Eigen/Dense>

template <typename ValueT>
struct DeterminantTraits
{
    typedef typename Eigen::Matrix<ValueT, Eigen::Dynamic, Eigen::Dynamic> MatrixT;
    typedef typename Eigen::Matrix<ValueT, Eigen::Dynamic, 1> ColumnT;
    typedef typename Eigen::Matrix<ValueT, 1, Eigen::Dynamic> RowT;
    typedef Eigen::OuterStride<Eigen::Dynamic> Strided;

    typedef Eigen::Map<MatrixT, Eigen::Aligned, Strided> BufferMap;
    typedef Eigen::Map<MatrixT, Eigen::Aligned> MatrixMap;
    typedef Eigen::Map<ColumnT, Eigen::Aligned> ColumnMap;
    typedef Eigen::Map<RowT, Eigen::Aligned> RowMap;

    typedef Eigen::Map<const MatrixT, Eigen::Aligned, Strided> BufferConstMap;
    typedef Eigen::Map<const MatrixT, Eigen::Aligned> MatrixConstMap;
    typedef Eigen::Map<const ColumnT, Eigen::Aligned> ColumnConstMap;
    typedef Eigen::Map<const RowT, Eigen::Aligned, Strided> RowConstMap;
};

template <typename ValueT>
static ValueT *aligned_new(unsigned size)
{
    return static_cast<ValueT *>(
            Eigen::internal::aligned_malloc(size * sizeof(ValueT)));
}

template <typename ValueT>
static void aligned_free(ValueT *what)
{
    Eigen::internal::aligned_free(what);
}


/* ---------------------------  DETERMINANT  --------------------------- */

template <typename ValueT>
Determinant<ValueT>::Determinant(unsigned capacity)
    : weight_(1),
      order_(0),
      capacity_(capacity),
      invmat_buffer_(capacity_ ? aligned_new<ValueT>(capacity_*capacity_) : NULL)
{ }

template <typename ValueT>
Determinant<ValueT>::Determinant(const Determinant &other)
    : weight_(other.weight_),
      order_(other.order_),
      capacity_(other.capacity_),
      invmat_buffer_(capacity_ ? aligned_new<ValueT>(capacity_*capacity_) : NULL)
{
    if (MODE_DEBUG)
        fprintf(stderr, "NOTE: copying of Determinant instance\n");

    std::copy(other.invmat_buffer_, other.invmat_buffer_ + capacity_*capacity_,
              invmat_buffer_);

    if (MODE_DEBUG)
        verify();
}

template <typename ValueT>
Determinant<ValueT>::~Determinant()
{
    // Works with null pointer
    aligned_free(invmat_buffer_);
}

template <typename ValueT>
void swap(Determinant<ValueT> &left, Determinant<ValueT> &right)
{
    using std::swap;        // using ADL

    swap(left.weight_, right.weight_);
    swap(left.order_, right.order_);
    swap(left.capacity_, right.capacity_);
    swap(left.invmat_buffer_, right.invmat_buffer_);  // here is the magic
}

template <typename ValueT>
Determinant<ValueT> &Determinant<ValueT>::operator=(Determinant other)
{
    // implement copy-and-swap idiom: pass-by-value + swap function
    swap(*this, other);
    return *this;
}

template <typename ValueT>
void Determinant<ValueT>::verify(double tol) const
{
    if (weight_ == ValueT(0)) {
        throw VerificationError("weight must be nonzero");
    }

    typename Traits::BufferMap invmat(invmat_buffer_, order_, order_, capacity_);

    // check weight
    ValueT real_weight = ValueT(1)/invmat.determinant();
    if (std::abs(real_weight - weight_) > tol * std::abs(real_weight)) {
        std::cerr << "det: " << real_weight << " != " << weight_ << std::endl;
        throw VerificationError("weight is %.10g, but determinant gives %.10g",
                                real_weight, weight_);
    }
}

template <typename ValueT>
void Determinant<ValueT>::reserve(unsigned new_cap)
{
    if (new_cap > capacity_) {
        new_cap += CAPACITY_RESERVE;

        // expand the matrix, but let the results "carry" over
        ValueT *new_buffer = aligned_new<ValueT>(new_cap * new_cap);
        for (unsigned j = 0; j < order(); ++j) {
            for (unsigned i = 0; i < order(); ++i)
                new_buffer[j*new_cap + i] = invmat_buffer_[j*capacity_ + i];
        }
        aligned_free(invmat_buffer_);
        invmat_buffer_ = new_buffer;
        capacity_ = new_cap;

        if (MODE_DEBUG)
            verify();
    }
}


template <typename ValueT>
DetMove<ValueT>::DetMove(Determinant<ValueT> &target, unsigned max_rank)
    : target_(&target),
      max_rank_(max_rank),
      capacity_(target.capacity()),
      rank_(0),
      rbar_buffer_(aligned_new<ValueT>(capacity_ * max_rank_)),
      cbar_buffer_(aligned_new<ValueT>(capacity_ * max_rank_)),
      sbar_buffer_(aligned_new<ValueT>(max_rank_ * max_rank_)),
      ratio_(0)
{
    assert(max_rank != 0);
}

template <typename ValueT>
DetMove<ValueT>::DetMove(const DetMove &other)
    : target_(other.target_),
      max_rank_(other.max_rank_),
      capacity_(other.capacity_),
      rank_(other.rank_),
      rbar_buffer_(aligned_new<ValueT>(capacity_ * max_rank_)),
      cbar_buffer_(aligned_new<ValueT>(capacity_ * max_rank_)),
      sbar_buffer_(aligned_new<ValueT>(max_rank_ * max_rank_)),
      ratio_(0)
{
    std::copy(other.rbar_buffer_, other.rbar_buffer_ + capacity_*max_rank_,
              rbar_buffer_);
    std::copy(other.cbar_buffer_, other.cbar_buffer_ + capacity_*max_rank_,
              cbar_buffer_);
    std::copy(other.sbar_buffer_, other.sbar_buffer_ + max_rank_*max_rank_,
              sbar_buffer_);
}

template <typename ValueT>
DetMove<ValueT>::DetMove()
    : target_(NULL),
      rbar_buffer_(NULL),
      cbar_buffer_(NULL),
      sbar_buffer_(NULL)
{ }

template <typename ValueT>
DetMove<ValueT>::~DetMove()
{
    aligned_free(rbar_buffer_);
    aligned_free(cbar_buffer_);
    aligned_free(sbar_buffer_);
}

template <typename ValueT>
void swap(DetMove<ValueT> &left, DetMove<ValueT> &right)
{
    using std::swap;        // using ADL

    swap(left.target_, right.target_);
    swap(left.max_rank_, right.max_rank_);
    swap(left.capacity_, right.capacity_);
    swap(left.rank_, right.rank_);
    swap(left.rbar_buffer_, right.rbar_buffer_);
    swap(left.cbar_buffer_, right.cbar_buffer_);
    swap(left.sbar_buffer_, right.sbar_buffer_);
    swap(left.ratio_, right.ratio_);
}

template <typename ValueT>
DetMove<ValueT> &DetMove<ValueT>::operator=(DetMove other)
{
    // copy-and-swap idiom: pass-by-value (reuse copy c'tor) + swap function
    swap(*this, other);
    return *this;
}

template <typename ValueT>
void DetMove<ValueT>::reserve()
{
    const unsigned new_cap = this->target_->capacity();
    if (new_cap > capacity_) {
        aligned_free(rbar_buffer_);
        aligned_free(cbar_buffer_);

        rbar_buffer_ = aligned_new<ValueT>(max_rank_ * new_cap);
        cbar_buffer_ = aligned_new<ValueT>(max_rank_ * new_cap);
        capacity_ = new_cap;
    }
}


template <typename ValueT>
void DetAppendMove<ValueT>::propose(unsigned rank, ValueT *row, ValueT *col,
                                    ValueT *star)
{
    this->rank_ = rank;

    Determinant<ValueT> &target = *this->target_;
    const unsigned old_size = target.order_;

    // reserve some more space
    target.reserve(old_size + rank);
    this->reserve();

    // row does not need to be transformed for the ratio, so we are keeping
    // the original.
    row_ = row;

    // set up maps of pointers to Eigen stuff
    typename Traits::BufferMap M(target.invmat_buffer_, old_size, old_size,
                                 target.capacity_);

    if (rank == 1) {
        typename Traits::RowMap r(row_, old_size);
        typename Traits::ColumnMap c(col, old_size);
        typename Traits::ColumnMap cbar_over_sbar(this->cbar_buffer_, old_size);

        cbar_over_sbar.noalias() = -M * c;

        #ifdef __clang__
            // Workaround because clang somehow does not understand the
            // conversion of a scalar product for std::complex<double>
            this->ratio_ = *star + (r * cbar_over_sbar)(0, 0);
        #else
            this->ratio_ = *star + static_cast<ValueT>(r * cbar_over_sbar);
        #endif

    } else {
        typename Traits::MatrixMap r(row_, rank, old_size);
        typename Traits::MatrixMap c(col, old_size, rank);
        typename Traits::MatrixMap s(star, rank, rank);

        typename Traits::MatrixMap sbar_inv(this->sbar_buffer_, rank, rank);
        typename Traits::MatrixMap cbar_over_sbar(this->cbar_buffer_, old_size, rank);

        cbar_over_sbar.noalias() = -M * c;

        sbar_inv.noalias() = s + r * cbar_over_sbar;

        // FIXME: slow
        this->ratio_ = sbar_inv.determinant();
    }
}

template <typename ValueT>
void DetAppendMove<ValueT>::accept()
{
    Determinant<ValueT> &target = *this->target_;
    const unsigned old_size = target.order();
    const unsigned new_size = old_size + this->rank_;

    // adjust the weight and the order
    target.weight_ *= this->ratio_;

    if (this->rank_ == 1) {
        // set up maps of pointers to Eigen stuff
        typename Traits::BufferMap M(target.invmat_buffer_, old_size,
                                     old_size, target.capacity_);
        typename Traits::RowMap r(row_, old_size);
        typename Traits::RowMap rbar(this->rbar_buffer_, old_size);
        typename Traits::ColumnMap cbar(this->cbar_buffer_, old_size);

        const ValueT sbar = ValueT(1)/this->ratio_;

        rbar.noalias() = -sbar * r * M;

        M.noalias() += cbar * rbar;

        cbar *= sbar;

        target.order_ += 1;

        typename Traits::BufferMap M_ext(target.invmat_buffer_, new_size,
                                         new_size, target.capacity_);

        M_ext.bottomLeftCorner(1, old_size) = rbar;
        M_ext.topRightCorner(old_size, 1) = cbar;
        M_ext(old_size, old_size) = sbar;

    } else {
        typename Traits::BufferMap M(target.invmat_buffer_, old_size,
                                     old_size, target.capacity_);
        typename Traits::MatrixMap r(row_, this->rank_, old_size);
        typename Traits::MatrixMap rbar(this->rbar_buffer_, this->rank_, old_size);
        typename Traits::MatrixMap cbar(this->cbar_buffer_, old_size, this->rank_);
        typename Traits::MatrixMap sbar(this->sbar_buffer_, this->rank_, this->rank_);

        // FIXME: slow
        sbar = sbar.inverse();

        rbar.noalias() = -sbar * r * M;

        M.noalias() += cbar * rbar;

        cbar = cbar * sbar;

        target.order_ += this->rank_;

        typename Traits::BufferMap M_ext(target.invmat_buffer_, new_size,
                                         new_size, target.capacity_);

        M_ext.bottomLeftCorner(this->rank_, old_size) = rbar;
        M_ext.topRightCorner(old_size, this->rank_) = cbar;
        M_ext.bottomRightCorner(this->rank_, this->rank_) = sbar;
    }

    if (MODE_DEBUG)
        this->target_->verify();
}


template <typename ValueT>
void DetRemoveMove<ValueT>::verify_args(unsigned rank, unsigned *rowno,
                                        unsigned *colno) const
{
    Determinant<ValueT> &target = *this->target_;

    if (rank > this->max_rank_) {
        throw VerificationError("Rank %d exceeds max rank of move", rank);
    }
    if (rank > target.order()) {
        throw VerificationError("Not enough rows %d to remove %d from",
                                target.order(), rank);
    }

    for (unsigned i = 0; i != rank; ++i) {
        if (rowno[i] >= target.order())
            throw VerificationError("Row number at %d exceeds matrix order", i);
        if (colno[i] >= target.order())
            throw VerificationError("Col number at %d exceeds matrix order", i);
    }
}

template <typename ValueT>
unsigned DetRemoveMove<ValueT>::replacements(unsigned *repl,
                         const unsigned *target, unsigned order, unsigned rank)
{
    const unsigned split = order - rank;

    // So, what we want to do is to replace the rows/cols to be removed
    // (target) with the last rows (split, split+1, ..., order-1).  So this
    // will be our initial guess.
    //
    // We note that this would produce a sign whenever this indeed exchanges
    // two elements, i.e., whenever target[i] != split + i.
    for (unsigned i = 0; i != rank; ++i)
        repl[i] = split + i;

    // Now for rank > 1 this is not the whole story, since elements may just be
    // transposed, i.e., target[i] == split + j for some j != i.  (This is a
    // case we want to handle explicitly, otherwise we need to allocate some
    // temporary space for the swap.)
    unsigned nswap = 0;
    for (unsigned i = 0; i != rank; ++i) {
        if (target[i] < split) {
            // An element that cannot possibly map to itself -> permutation
            // Note that as the expansion order grows and rank is typically
            // very small, that will actually be the vast majority of cases.
            ++nswap;
        } else {
            // An target element that has a counter-part in repl, look for it
            unsigned r;
            for (r = 0; target[i] != repl[r]; ++r)
                assert(r < rank);

            // Now we have found a match.  If the position is already right,
            // we do not need to do anything, otherwise we swap the 'right'
            // replacement with the 'wrong' one.  We need to track these as
            // well, as the "S" matrix has then elements transposed w.r.t. to
            // the last-but-nth, ... elements.
            if (r != i) {
                std::swap(repl[i], repl[r]);
                ++nswap;
            }
        }
    }

    if (DEBUG_REPL) {
        fprintf(stderr, "REPL (order=%d, nswap=%d): ",
                order, nswap);
        for (unsigned it = 0; it != rank; ++it)
            fprintf(stderr, "%d ", target[it]);
        fprintf(stderr, "<-");
        for (unsigned it = 0; it != rank; ++it)
            fprintf(stderr, "%d ", repl[it]);
        fprintf(stderr, "\n");
    }

    return nswap;
}

template <typename ValueT>
void DetRemoveMove<ValueT>::propose(unsigned rank, unsigned *rowno,
                                    unsigned *colno)
{
    if (MODE_DEBUG)
        verify_args(rank, rowno, colno);

    this->rank_ = rank;

    const Determinant<ValueT> &target = *this->target_;

    // note we switched around row and column to index the inverse
    this->rowno_ = colno;
    this->colno_ = rowno;

    // set up maps of pointers to Eigen stuff
    typename Traits::BufferMap M(target.invmat_buffer_, target.order_,
                                 target.order_, target.capacity_);

    if (rank == 1) {
        // the ratio is just given as element of the inverse
        //const unsigned row_stride = this->target_->capacity_;
        this->ratio_ = M(*rowno_, *colno_);

    } else {
        typename Traits::MatrixMap sbar(this->sbar_buffer_, rank, rank);

        // Extract sbar
        for (unsigned j = 0; j != rank; ++j) {
            for (unsigned i = 0; i != rank; ++i) {
                sbar(i, j) = M(rowno_[i], colno_[j]);
            }
        }

        // FIXME slow
        this->ratio_ = sbar.determinant();
    }

    // set up a vectors of replacements for the rows/columns we need to remove
    // that are not in the area of to be-removed-rows anyway
    unsigned nperm = 0;
    nperm += replacements(&this->rowrepl_[0], rowno_, target.order_, rank);
    nperm += replacements(&this->colrepl_[0], colno_, target.order_, rank);

    // we need to add a sign whenever we do a replacement
    this->perm_sign_ = nperm % 2 ? -1 : 1;
}

template <typename ValueT>
void DetRemoveMove<ValueT>::accept()
{
    Determinant<ValueT> &target = *this->target_;

    const unsigned old_order = target.order_;
    const unsigned new_order = target.order_ - this->rank_;

    // reserve space for the buffers
    this->reserve();

    // adjust the weight and the order
    target.weight_ *= this->ratio_;
    target.weight_ *= this->perm_sign_;

    // set up maps of pointers to Eigen matrix
    typename Traits::BufferMap M(target.invmat_buffer_, old_order,
                                 old_order, target.capacity_);

    if (this->rank_ == 1) {
        // calculate scale (-1/sbar).
        const ValueT scale = ValueT(-1)/this->ratio_;

        // set up maps of pointers to Eigen stuff
        typename Traits::RowMap rbar(this->rbar_buffer_, old_order);
        typename Traits::ColumnMap cbar(this->cbar_buffer_, old_order);

        // first cache C and R/sbar to avoid a temporary in the assignment
        cbar = M.col(*colno_) * scale;
        rbar = M.row(*rowno_);

        // Now, replace the c'th column with the last column
        if (*colno_ != new_order) {
            rbar(*colno_) = rbar(new_order);
            M.col(*colno_) = M.col(new_order);
        }

        // Now, replace the r'th row with the last row
        if (*rowno_ != new_order) {
            cbar(*rowno_) = cbar(new_order);
            M.row(*rowno_) = M.row(new_order);
        }

        // invmat -= C (x) R / sbar
        M.noalias() += cbar * rbar;

    } else {
        typename Traits::MatrixMap rbar(this->rbar_buffer_, this->rank_, old_order);
        typename Traits::MatrixMap cbar(this->cbar_buffer_, old_order, this->rank_);
        typename Traits::MatrixMap sbar(this->sbar_buffer_, this->rank_, this->rank_);

        // FIXME slow
        sbar = sbar.inverse();

        // first cache C and R/sbar to avoid a temporary in the assignment
        for(unsigned i = 0; i != this->rank_; ++i) {
            cbar.col(i) = M.col(colno_[i]);
            rbar.row(i) = M.row(rowno_[i]);
        }

        // Now perform the proposed row (column) replacements both in the
        // matrix and the stored columns (rows).
        for (unsigned i = 0; i != this->rank_; ++i) {
            const unsigned ctarget = colno_[i], csource = colrepl_[i];

            if (ctarget == csource)
                continue;
            rbar.col(ctarget) = rbar.col(csource);
            M.col(ctarget) = M.col(csource);
        }
        for (unsigned i = 0; i != this->rank_; ++i) {
            const unsigned rtarget = rowno_[i], rsource = rowrepl_[i];

            if (rtarget == rsource)
                continue;
            cbar.row(rtarget) = cbar.row(rsource);
            M.row(rtarget) = M.row(rsource);
        }

        // invmat -= C (x) R / sbar
        M.noalias() -= cbar * sbar * rbar;
    }

    // Remove r-th row and c-th column (now the last ones)
    target.order_ = new_order;

    if (MODE_DEBUG)
        this->target_->verify();
}


template <typename ValueT>
void DetSetMove<ValueT>::propose(unsigned new_order, ValueT *new_mat)
{
    new_mat_.decompose(new_mat, new_order);
}

template <typename ValueT>
void DetSetMove<ValueT>::accept()
{
    Determinant<ValueT> &target = *this->target_;

    new_mat_.inverse(target.invmat_buffer_, target.capacity_);
    target.order_ = new_mat_.size();
    target.weight_ = new_mat_.determinant();

    if (MODE_DEBUG)
        target.verify();
}

