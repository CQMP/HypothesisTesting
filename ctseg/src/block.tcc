/**
 * Implementation of bath blocks/non-interacting diagrams (determinants).
 *
 * Author: Markus Wallerberger
 */
#ifndef _BLOCK_HH
#  error "Do not include this file directly, include block.hh instead"
#endif

#include <algorithm>
#include <iostream>
#include <cmath>

BlockOperator::BlockOperator()
    : tau(-1),
      block_flavour(-1),
      slot(-1),
      block_no(-1),
      move_pos(-1)
{ }

template<typename ValueT>
BlockOperator::BlockOperator(const Block<ValueT> &block, double tau,
                             unsigned block_flavour, bool effect,
                             unsigned block_no, unsigned move_pos)
    : tau(tau),
      block_flavour(block_flavour),
      slot(block.slot(tau, block_flavour, effect)),  // verification here
      block_no(block_no),
      move_pos(move_pos)
{ }

template <typename ValueT>
std::vector<ValueT> spline_linear(const ValueT *values, unsigned npairs,
                                  unsigned npoints)
{
    const unsigned nbins = npoints - 1;      // endpoint removed
    const unsigned order = 2;
    std::vector<ValueT> spline(npairs * nbins * 2);

    for (unsigned ipair = 0; ipair != npairs; ++ipair) {
        const ValueT *source = &values[ipair * npoints];
        ValueT *sink = &spline[ipair * nbins * order];

        // Set up f(x) = f[n] + (x - x[n]) * f'[n]
        for (unsigned i = 0; i != nbins; ++i) {
            sink[i * order] = source[i];
            sink[i * order + 1] = source[i+1] - source[i];
        }
    }
    return spline;
}

template <typename ValueT>
PolyHybFunc<ValueT> PolyHybFunc<ValueT>::linear(unsigned nflavours,
                unsigned nvalues, const ValueT *values, double beta, int sign)
{
    std::vector<ValueT> spline = spline_linear(values, nflavours * nflavours,
                                               nvalues);
    return PolyHybFunc(data(spline), nflavours, nvalues - 1, 2, beta, sign);
}

template <typename ValueT>
PolyHybFunc<ValueT>::PolyHybFunc(const ValueT *values, unsigned nflavours,
                         unsigned nbins, unsigned order, double beta, int sign)
    : sign_(sign),
      beta_(beta),
      nflavours_(nflavours),
      nbins_(nbins),
      values_()
{
    assert(nflavours > 0);
    assert(nbins_ >= 1);
    assert(sign == 1 || sign == -1);

    const unsigned npairs = nflavours * nflavours;
    const unsigned nderivs = nbins * order;

    // double the interval for each pair
    std::vector<ValueT> doubled(npairs * 2 * nderivs);
    for (unsigned ipair = 0; ipair != npairs; ++ipair) {
        const ValueT *source = &values[ipair * nderivs];
        ValueT *sink = &doubled[ipair * 2 * nderivs];

        for (unsigned i = 0; i != nderivs; ++i) {
            sink[i] = sign * source[i];
            sink[i + nderivs] = source[i];
        }
    }

    values_ = PiecewisePolynomial<ValueT>(data(doubled), npairs * 2 * nbins, order);
}

template <typename ValueT>
ValueT PolyHybFunc<ValueT>::value(BlockOperator coper, BlockOperator aoper) const
{
    return values_.value((coper.slot - aoper.slot) * nbins_);
}

template <typename ValueT>
void PolyHybFunc<ValueT>::values(ValueT *buffer,
                            unsigned ncoper, const BlockOperator *coper,
                            unsigned naoper, const BlockOperator *aoper) const
{
    // Eigen and LAPACK store column-major, so we do too
    for (unsigned l = 0; l != naoper; ++l)
        for (unsigned i = 0; i != ncoper; ++i)
            buffer[l * ncoper + i] = value(coper[i], aoper[l]);
}

/* ------------------------------ BATH BLOCK ------------------------------- */

template <typename ValueT>
Block<ValueT>::Block()
    : nflavours_(0),
      beta_(-1),
      hybrfunc_(NULL)
{ }

template <typename ValueT>
Block<ValueT>::Block(const IHybFunc<ValueT> &hybrfunc)
    : nflavours_(hybrfunc.nflavours()),
      beta_(hybrfunc.beta()),
      hybrfunc_(hybrfunc.clone()),
      det_(),
      copers_(),
      aopers_()
{
    if (MODE_DEBUG)
        verify();
}

template <typename ValueT>
Block<ValueT>::Block(const Block &other)
    : nflavours_(other.nflavours_),
      beta_(other.beta_),
      hybrfunc_(other.hybrfunc_->clone()),
      det_(other.det_),
      copers_(other.copers_),
      aopers_(other.aopers_)
{ }

template <typename ValueT>
void swap(Block<ValueT> &left, Block<ValueT> &right)
{
    using std::swap;  // ADL

    swap(left.nflavours_, right.nflavours_);
    swap(left.beta_, right.beta_);
    swap(left.hybrfunc_, right.hybrfunc_);  // Magic here
    swap(left.det_, right.det_);
    swap(left.copers_, right.copers_);
    swap(left.aopers_, right.aopers_);
}

template <typename ValueT>
std::vector<ValueT> Block<ValueT>::calc_hyb_matrix() const
{
    const unsigned order = det_.order();

    std::vector<ValueT> result(order * order);
    if (order != 0)
        hybrfunc().values(&result[0], order, &copers_[0], order, &aopers_[0]);

    return result;
}

template <typename ValueT>
void Block<ValueT>::verify(bool recursive, double tol) const
{
    if (recursive)
        det_.verify(tol);

    // verify contents of creation operators
    for (std::vector<BlockOperator>::const_iterator it = copers_.begin();
            it != copers_.end(); ++it) {
        if (it->block_flavour >= nflavours())
            throw VerificationError("illegal coper block flv %d", it->block_flavour);
        if (it->tau < 0 || it->tau >= beta())
            throw VerificationError("illegal coper tau %f", it->tau);
        if (std::abs(it->slot - slot(it->tau, it->block_flavour, true)) > tol)
            throw VerificationError("inconsistent coper slot %g", it->slot);
    }

    // verify contents of annihilation operators
    for (std::vector<BlockOperator>::const_iterator it = aopers_.begin();
            it != aopers_.end(); ++it) {
        if (it->block_flavour >= nflavours())
            throw VerificationError("illegal aoper block flv %d", it->block_flavour);
        if (it->tau < 0 || it->tau >= beta())
            throw VerificationError("illegal aoper tau %f", it->tau);
        if (std::abs(it->slot - slot(it->tau, it->block_flavour, false)) > tol)
            throw VerificationError("inconsistent aoper slot %g", it->slot);
    }

    // the case of an empty matrix must be handled specially since eigen cannot
    // deal with it (booo!)
    if (order()) {
        // verify matrix equality
        std::vector<ValueT> hyb_from_func_values = calc_hyb_matrix();

        typename DeterminantTraits<ValueT>::BufferConstMap invmat(
                det_.invmat(), det_.order(), det_.order(), det_.capacity());
        typename DeterminantTraits<ValueT>::BufferConstMap hyb_from_func(
                &hyb_from_func_values[0], det_.order(), det_.order(), det_.order());

        typename DeterminantTraits<ValueT>::MatrixT hyb_from_det = invmat.inverse();
        if (!hyb_from_func.isApprox(hyb_from_det, tol)) {
            std::cerr << "\n\nHYB_FROM_FUNC =\n" << hyb_from_func
                      << "\n\nHYB_FROM_DET =\n" << hyb_from_det << std::endl;

            typename DeterminantTraits<ValueT>::MatrixT diff = hyb_from_det - hyb_from_func;
            throw VerificationError(
                    "stored and computed inverse off by avg. %.3g, max. %.3g",
                    diff.norm(), diff.template lpNorm<Eigen::Infinity>()
                    );
        }
    }
}

template <typename ValueT>
double Block<ValueT>::slot(double tau, unsigned block_flavour, bool effect) const
{
    assert(tau >= 0 && tau < beta());
    assert(block_flavour < nflavours());

    // The main idea of the slot function is to pre-compute the location
    // of a creator/annihilator index pair in the bath picture:
    //
    //   c_aflv(atau) c^+_cflv(ctau)
    //         --> 2 * (cflv * nflavours + aflv) + (ctau - atau)/beta + 1
    //         --> cslot - aslot
    //
    // This is useful for speeding up the hybridisation lookup and G(tau)
    if (effect)
        return 2 * block_flavour * nflavours() + tau/beta() + 1;
    else
        return -2 * static_cast<int>(block_flavour) + tau/beta();
}


template <typename DetMoveT>
BlockMove<DetMoveT>::BlockMove(Block<Value> &target, unsigned max_rank)
    : target_(&target),
      det_move_(target.det(), max_rank)
{ }


template <typename ValueT>
BlockAppendMove<ValueT>::BlockAppendMove(Block<ValueT> &target, unsigned max_rank)
    : BlockMove< DetAppendMove<ValueT> >(target, max_rank),
      max_order_(max_rank),
      capacity_(target.det_.capacity()),
      coper_(NULL),
      aoper_(NULL),
      newrow_buffer_(aligned_new<ValueT>(capacity_ * max_order_)),
      newcol_buffer_(aligned_new<ValueT>(capacity_ * max_order_)),
      newstar_buffer_(aligned_new<ValueT>(max_order_ * max_order_))
{
    assert(max_rank);
}

template <typename ValueT>
BlockAppendMove<ValueT>::BlockAppendMove(const BlockAppendMove &other)
    : BlockMove< DetAppendMove<ValueT> >(other),
      max_order_(other.max_order_),
      capacity_(other.capacity_),
      coper_(other.coper_),
      aoper_(other.aoper_),
      newrow_buffer_(aligned_new<ValueT>(capacity_ * max_order_)),
      newcol_buffer_(aligned_new<ValueT>(capacity_ * max_order_)),
      newstar_buffer_(aligned_new<ValueT>(max_order_ * max_order_))
{
    std::copy(other.newrow_buffer_, other.newrow_buffer_ + capacity_*max_order_,
              newrow_buffer_);
    std::copy(other.newcol_buffer_, other.newcol_buffer_ + capacity_*max_order_,
              newcol_buffer_);
    std::copy(other.newstar_buffer_, other.newstar_buffer_ + max_order_*max_order_,
              newstar_buffer_);
}

template <typename ValueT>
BlockAppendMove<ValueT>::~BlockAppendMove()
{
    aligned_free(newrow_buffer_);
    aligned_free(newcol_buffer_);
    aligned_free(newstar_buffer_);
}

template <typename ValueT>
void BlockAppendMove<ValueT>::reserve()
{
    unsigned new_cap = this->target_->det_.capacity();
    if (new_cap > capacity_) {
        aligned_free(newrow_buffer_);
        aligned_free(newcol_buffer_);

        newrow_buffer_ = aligned_new<ValueT>(max_order_ * new_cap);
        newcol_buffer_ = aligned_new<ValueT>(max_order_ * new_cap);
        capacity_ = new_cap;
    }
}

template <typename ValueT>
void BlockAppendMove<ValueT>::propose(unsigned rank, BlockOperator *coper,
                                      BlockOperator *aoper)
{
    const Block<ValueT> &target = *this->target_;
    const unsigned old_order = target.order();
    const IHybFunc<ValueT> &hybr = target.hybrfunc();

    coper_ = coper;
    aoper_ = aoper;

    reserve();

    hybr.values(newstar_buffer_, rank, coper_, rank, aoper_);
    if (old_order) {
        hybr.values(newcol_buffer_, old_order, &target.copers_[0], rank, aoper_);
        hybr.values(newrow_buffer_, rank, coper_, old_order, &target.aopers_[0]);
    }

    this->det_move_.propose(rank, newrow_buffer_, newcol_buffer_, newstar_buffer_);
}

template <typename ValueT>
void BlockAppendMove<ValueT>::accept()
{
    const unsigned rank = this->rank();

    this->det_move_.accept();

    Block<ValueT> &target = *this->target_;
    target.copers_.insert(target.copers_.end(), coper_, coper_ + rank);
    target.aopers_.insert(target.aopers_.end(), aoper_, aoper_ + rank);

    if (MODE_DEBUG)
        target.verify();
}

template <typename ValueT>
void BlockRemoveMove<ValueT>::propose(unsigned rank,
                                      unsigned *cblockpos, unsigned *ablockpos)
{
    this->det_move_.propose(rank, cblockpos, ablockpos);
}

template <typename ValueT>
void BlockRemoveMove<ValueT>::accept()
{
    const unsigned rank = this->rank();

    this->det_move_.accept();

    Block<ValueT> &target = *this->target_;
    for (unsigned i = 0; i != rank; ++i) {
        target.copers_[this->det_move_.rowno()[i]] =
                target.copers_[this->det_move_.rowrepl()[i]];
        target.aopers_[this->det_move_.colno()[i]] =
                target.aopers_[this->det_move_.colrepl()[i]];
    }

    const unsigned new_order = target.order();
    target.copers_.resize(new_order);
    target.aopers_.resize(new_order);

    if (MODE_DEBUG)
        this->target_->verify();
}

template <typename ValueT>
void BlockSetMove<ValueT>::propose(unsigned order, const BlockOperator *coper,
                                   const BlockOperator *aoper)
{
    const Block<ValueT> &target = *this->target_;

    this->new_hybr_.resize(order * order + 1);
    target.hybrfunc().values(&this->new_hybr_[0], order, coper, order, aoper);

    this->coper_ = coper;
    this->aoper_ = aoper;
    this->det_move_.propose(order, &this->new_hybr_[0]);
}

template <typename ValueT>
void BlockSetMove<ValueT>::accept()
{
    Block<ValueT> &target = *this->target_;
    const unsigned new_order = this->det_move_.order();

    this->det_move_.accept();

    target.copers_.resize(new_order);
    target.aopers_.resize(new_order);

    std::copy(this->coper_, this->coper_ + new_order, target.copers_.begin());
    std::copy(this->aoper_, this->aoper_ + new_order, target.aopers_.begin());

    if (MODE_DEBUG)
        target.verify();
}

template <typename ValueT>
void BlockRecompute<ValueT>::propose()
{
    const Block<ValueT> &target = move_.target();

    if (target.order() != 0)
        move_.propose(target.order(), &target.copers()[0], &target.aopers()[0]);
    else
        move_.propose(0, NULL, NULL);

    if (move_.hard_reject())
        throw LossOfPrecision();
}

// ------------------------------- ESTIMATORS ---------------------------------

template <typename ValueT>
GtauBlockEstimator<ValueT>::GtauBlockEstimator(const Block<ValueT> &source,
                                               unsigned ntau_bins)
    : ntau_(ntau_bins),
      source_(&source)
{
    assert(ntau_bins > 0);
}

template <typename ValueT>
void GtauBlockEstimator<ValueT>::estimate(ValueT *accum, ValueT weight,
                                          const ValueT *hartree)
{
    const Block<ValueT> &source = *this->source_;
    const unsigned order = source.det_.order();

    // convert to Eigen matrix (probably unnecessary)
    typename DeterminantTraits<ValueT>::BufferConstMap M(
                source.det_.invmat(), order, order, source.det_.capacity());

    for (unsigned i = 0; i != order; ++i) {
        const ValueT cslot = source.copers_[i].slot;
        for (unsigned j = 0; j != order; ++j) {
            // 1. Note the transposition: M_ji, but tau'_i - tau_j.  This is
            //    because the removal of a line introduces a transpsotion in
            //    the inverse matrix: det(M^-1_rem(ij))/det(M^-1) = M_ji.
            //
            // 2. The time and orbital arguments are reversed wrt. to the
            //    Green's function, which corresponds to the switch from the
            //    bath to the impurity picture. We do this switch in post-
            //    processing:  g_ij(-t) -> -g_ji(t)
            //
            // 3. Finally, we do not fold negative times back to positive
            //    ones; to avoid CPU pipe flushes, we measure the function on
            //    the interval [-beta, beta) and only transform the result.
            //
            const ValueT aslot = source.aopers_[j].slot;
            const ValueT curr_val = M(j, i) * (hartree == NULL ? 1 : hartree[i]);

            accum[unsigned(ntau_ * (cslot - aslot))] += curr_val * weight;
        }
    }
}

template <typename ValueT>
void GtauBlockEstimator<ValueT>::postprocess(ValueT *result, const ValueT *accum,
                                             ValueT sum_weights)
{
    const Block<ValueT> &source = *this->source_;
    const double scale = ntau_/source.beta() * -1./source.beta() * 1./sum_weights;

    for (unsigned i = 0; i != source.nflavours(); ++i) {
        for (unsigned j = 0; j != source.nflavours(); ++j) {
            // Note the transposition, which is undoing the transposition in
            // the measured quantities.
            ValueT *pair_result = &result[(i * source.nflavours() + j) * ntau_];
            const ValueT *pair_accum = &accum[(j * source.nflavours() + i) * 2*ntau_];
            for (unsigned t = 0; t != ntau_; ++t) {
                pair_result[t] = scale * (-pair_accum[ntau_ - t - 1]
                                          + pair_accum[2*ntau_ - t - 1]);
            }
        }
    }
}

template <typename ValueT>
GiwBlockEstimator<ValueT>::GiwBlockEstimator(const Block<ValueT> &source,
                                             unsigned niwf, bool use_nfft)
    : niwf_(niwf),
      source_(&source),
      plan_(GaussianWindow(14, 4 * niwf, 4)),
      f_hat_(plan_.nfreq())
{
    assert(niwf > 0);
}

template <typename ValueT>
void GiwBlockEstimator<ValueT>::estimate(AccumulatorT accum, ValueT weight,
                                         const ValueT *hartree)
{
    const Block<ValueT> &source = *this->source_;
    const unsigned order = source.det_.order();

    // convert to Eigen matrix (probably unnecessary)
    typename DeterminantTraits<ValueT>::BufferConstMap M(
                    source.det_.invmat(), order, order, source.det_.capacity());

    for (unsigned i = 0; i != order; ++i) {
        const BlockOperator &coper = source.copers_[i];
        for (unsigned j = 0; j != order; ++j) {
            // 1. Note the transposition: M_ji, but tau'_i - tau_j.  This is
            //    because the removal of a line introduces a transpsotion in
            //    the inverse matrix: det(M^-1_rem(ij))/det(M^-1) = M_ji.
            //
            // 2. The time and orbital arguments are reversed wrt. to the
            //    Green's function, which corresponds to the switch from the
            //    bath to the impurity picture: g_ij(-t) -> -g_ji(t)
            //
            // 3. Finally, we do not fold negative times back to positive
            //    ones; to avoid CPU pipe flushes, we measure the function on
            //    the interval [-beta, beta) and only transform the result.
            //
            const BlockOperator &aoper = source.aopers_[j];
            std::complex<double> *accum_part = accum + plan_.ngrid() *
                    (aoper.block_flavour * source.nflavours() + coper.block_flavour);

            plan_.add(accum_part,
                      weight * M(j, i) * (hartree == NULL ? 1 : hartree[i]),
                      (aoper.tau - coper.tau)/(2 * source.beta())
                      );
        }
    }
}

template <typename ValueT>
void GiwBlockEstimator<ValueT>::postprocess(ResultT result,
                                        AccumulatorT accum, ValueT sum_weights)
{
    // TODO: figure out why we don't need a minus sign here
    const Block<ValueT> &source = *this->source_;
    const double scale = 1./source.beta() * 1./sum_weights;

    // Perform deferred NDFT for each pair and add to results.
    for (unsigned i = 0; i != source.npairs(); ++i) {
        plan_.compute(&f_hat_[0], &accum[i * plan_.ngrid()]);
        for (unsigned n = 0; n != 2*niwf_; ++n)
            result[2*niwf_*i + n] = scale * f_hat_[2*n + 1];
    }
}
