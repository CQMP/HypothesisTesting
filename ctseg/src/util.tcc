/* Module implementing utility functions for QMC
 *
 * Author: Markus Wallerberger
 */
#ifndef _UTIL_HH
#  error "Do not include this file directly, include util.hh instead"
#endif

#include <Eigen/Dense>

template <typename IterT, typename Comp, typename Swap>
unsigned tsort(IterT begin, IterT end, Comp comp, Swap swap)
{
    assert(begin <= end);
    typedef typename std::iterator_traits<IterT>::value_type ValueT;

    // There is usually no excuse for re-implementing a sorting algorithm in
    // C++, but unfortunately we need two things that are very cumbersome to
    // achieve with the STL: (1) a custom swap function and, ideally, (2) easy
    // access to the number of swaps.
    unsigned swaps = 0;
    IterT left = begin;
    IterT right = end - 1;

    // We are sorting zero or one element, so we can return immediately
    if (left >= right)
        return swaps;

    // Partitioning phase: choose the center *value* as pivot and move all
    // elements that are less to the left and all elements that are greater
    // to the right.
    ValueT pivot = *(begin + (end - begin)/2);
    for (;;) {
        while (comp(*left, pivot))
            ++left;
        while (comp(pivot, *right))
            --right;
        if (left >= right)
            break;

        swap(*left, *right);
        ++swaps;
        ++left;
        --right;
    }
    if (left == right) {
        ++left;
        --right;
    }

    // Recursion phase: with the new split point, sort the smaller lists.
    assert(right < left);
    swaps += tsort(begin, right + 1, comp, swap);
    swaps += tsort(left, end, comp, swap);

    // Check that the list is indeed sorted
    if (MODE_DEBUG) {
        for (IterT curr = begin; curr != end - 1; ++curr)
            if (comp(*(curr + 1), *curr))
                throw VerificationError("Did not produce sorted list");
    }
    return swaps;
}

template <typename IterT, typename Comp>
unsigned tsort(IterT begin, IterT end, Comp comp)
{
    using std::swap;
    typedef typename std::iterator_traits<IterT>::value_type ValueT;

    return tsort(begin, end, comp,
                 static_cast<void (*)(ValueT &, ValueT &)>(swap));
}

template <typename IterT>
unsigned tsort(IterT begin, IterT end)
{
    typedef typename std::iterator_traits<IterT>::value_type ValueT;
    return tsort(begin, end, std::less<ValueT>());
}


template <typename ValueT>
struct InplaceLUTraits
{
    typedef Eigen::Matrix<ValueT, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
    typedef Eigen::Map<MatrixType> MatrixMap;
    typedef Eigen::Map<const MatrixType> MatrixCMap;
    typedef Eigen::Map<MatrixType, Eigen::Unaligned,
                       Eigen::OuterStride<Eigen::Dynamic> > StridedMap;

    typedef Eigen::Matrix<long, Eigen::Dynamic, 1> IndicesType;
    typedef Eigen::Map<IndicesType, Eigen::Unaligned> IndicesMap;

    typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, long> PermType;
    typedef Eigen::Map<PermType, Eigen::Unaligned> PermMap;
};


template <typename ValueT>
InplaceLU<ValueT>::InplaceLU()
    : matrix_(NULL),
      transp_(1),
      perm_(1),
      size_(0),
      determinant_(0)
{ }

#include <iostream>

template <typename ValueT>
void InplaceLU<ValueT>::decompose(ValueT *matrix, unsigned size)
{
    matrix_ = matrix;
    size_ = size;

    // We need to handle zero sizes explicitly here, because Eigen's LU
    // decomposition crashes in this case (booo!)
    if (size_ == 0) {
        determinant_ = 1.;
        return;
    }

    // Ensure that the row buffer can handle that sort of thing.  We also need
    // to make sure that transp is the trivial permutation (partial_lu_inplace
    // does not initialize it).
    if (size_ > transp_.size()) {
        transp_.resize(size_);
        perm_.resize(size_);
    }

    typename Traits::MatrixMap mat(matrix_, size_, size_);
    typename Traits::IndicesMap row_trans(&transp_[0], size_);
    long ntransp;

    if (DEBUG_DECOMPOSE) {
        Eigen::PartialPivLU<typename Traits::MatrixType> lu(mat);
        std::cerr << "DET " << lu.determinant() << std::endl;
        std::cerr << "PERM " << lu.permutationP().indices() << std::endl;
        std::cerr << "MAT\n" << lu.matrixLU() << std::endl;
    }

    // Perform LU decomposition and compute determinant from it
    Eigen::internal::partial_lu_inplace(mat, row_trans, ntransp);
    determinant_ = ValueT(ntransp % 2 ? -1 : 1) * mat.diagonal().prod();

    // HACK: In Eigen, transpositions and permutations indices have different
    // meanings: trans[i] == j means swap element i with j, then do i<-i+1,
    // while permutation indices correspond to the usual notation. Since Eigen
    // `Map`s over PermutationMatrix and Transpositions are immutable, one
    // would have to use the heap-allocating version for the index conversion.
    // To avoid this, we basically copy the code of PermutationMatrix::
    // operator=(Transpositions&) here.
    for (unsigned i = 0; i != size_; ++i)
        perm_[i] = i;
    for (long k = size_ - 1; k >= 0; --k)
        std::swap(perm_[k], perm_[transp_[k]]);

    if (DEBUG_DECOMPOSE) {
        typename Traits::IndicesMap row_perm(&perm_[0], size_);
        std::cerr << "DET " << determinant_ << std::endl;
        std::cerr << "TRANS " << row_trans.transpose() << std::endl;
        std::cerr << "PERM " << row_perm.transpose() << std::endl;
        std::cerr << "MAT\n" << mat << std::endl;
    }
}

#include <iostream>

template <typename ValueT>
void InplaceLU<ValueT>::inverse(ValueT *out, unsigned out_stride) const
{
    assert(matrix_ != NULL);

    typename Traits::PermMap perm_mat(&perm_[0], size_);
    typename Traits::MatrixMap lu_mat(matrix_, size_, size_);
    typename Traits::StridedMap out_mat(out, size_, size_, out_stride);

    out_mat = perm_mat;
    lu_mat.template triangularView<Eigen::UnitLower>().solveInPlace(out_mat);
    lu_mat.template triangularView<Eigen::Upper>().solveInPlace(out_mat);
}
