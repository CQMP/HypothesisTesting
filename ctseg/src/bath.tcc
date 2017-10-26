/**
 * Module for storing bath/non-interacting diagrams (determinants).
 *
 * Author: Markus Wallerberger
 */
#ifndef _BATH_HH
#  error "Do not include this file directly, include hybr.hh instead"
#endif

#include <algorithm>
#include <iostream>
#include <cmath>

#include "util.hh"

template <typename ValueT>
Bath<ValueT>::Bath()
    : blocks_(),
      nflavours_(0),
      weight_(1)
{ }

template <typename ValueT>
void Bath<ValueT>::analyse_blocks()
{
    const unsigned nblocks = blocks_.size();
    assert(nblocks > 0);

    unsigned curr_flavour_offset = 0, curr_pair_offset = 0;
    for (unsigned blockno = 0; blockno != nblocks; ++blockno) {
        const Block<ValueT> &block = blocks_[blockno];
        for (unsigned i = 0; i != block.nflavours(); ++i)
            block_for_flavour_.push_back(blockno);

        flavour_offset_.push_back(curr_flavour_offset);
        pair_offset_.push_back(curr_pair_offset);

        curr_flavour_offset += block.nflavours();
        curr_pair_offset += block.npairs();
    }
    nflavours_ = curr_flavour_offset;
    npairs_ = curr_pair_offset;

    if (MODE_DEBUG)
        verify();
}

template <typename ValueT>
Bath<ValueT>::Bath(const std::vector<Block<ValueT> > &blocks)
    : blocks_(blocks),
      clookup_(),
      alookup_(),
      weight_(1)
{
    analyse_blocks();
}

template <typename ValueT>
Bath<ValueT>::Bath(const Block<ValueT> *blocks, unsigned nblocks)
    : blocks_(blocks, blocks + nblocks),
      clookup_(),
      alookup_(),
      weight_(1)
{
    analyse_blocks();
}

template <typename ValueT>
unsigned Bath<ValueT>::find(double tau, bool effect) const
{
    unsigned lower = 0, upper = order();

    // run-of-the-mill binary search (from the left - insertion points)
    if (!upper)
        return 0;
    for (;;) {
        // determine centre and double lookup to get sorted array
        const unsigned centre = lower + (upper - lower)/2;
        double tau_centre;
        if (effect) {
            const BlockPosition pos = this->clookup_[centre];
            tau_centre = this->blocks_[pos.block_no].copers()[pos.block_pos].tau;
        } else {
            const BlockPosition pos = this->alookup_[centre];
            tau_centre = this->blocks_[pos.block_no].aopers()[pos.block_pos].tau;
        }

        // if only zero or one element in the bracket -> one of the endpoints
        if (lower == centre)
            return tau < tau_centre ? lower : upper;
        // else move the bracket
        if(tau < tau_centre)
            upper = centre;
        else
            lower = centre;
    }
}

template <typename ValueT>
BathOperator Bath<ValueT>::coper(unsigned index) const
{
    assert(index < order());
    const BlockPosition pos = clookup_[index];
    const BlockOperator &oper = blocks_[pos.block_no].copers()[pos.block_pos];

    BathOperator ret = {oper.tau, oper.block_flavour + flavour_offset_[pos.block_no]};
    return ret;
}

template <typename ValueT>
BathOperator Bath<ValueT>::aoper(unsigned index) const
{
    assert(index < order());
    const BlockPosition pos = alookup_[index];
    const BlockOperator &oper = blocks_[pos.block_no].aopers()[pos.block_pos];

    BathOperator ret = {oper.tau, oper.block_flavour + flavour_offset_[pos.block_no]};
    return ret;
}

template <typename ValueT>
unsigned Bath<ValueT>::crlookup(BlockPosition pos) const
{
    const BlockOperator oper = blocks_[pos.block_no].copers()[pos.block_pos];
    unsigned index = find(oper.tau, true) - 1;
    // An error here may signify a broken sorting or a float comparison problem
    assert(clookup_[index] == pos);
    return index;
}

template <typename ValueT>
unsigned Bath<ValueT>::arlookup(BlockPosition pos) const
{
    const BlockOperator oper = blocks_[pos.block_no].aopers()[pos.block_pos];
    unsigned index = find(oper.tau, false) - 1;
    // An error here may signify a broken sorting or a float comparison problem
    assert(alookup_[index] == pos);
    return index;
}

template <typename ValueT>
int Bath<ValueT>::calc_perm_sign() const
{
    std::vector<double> ctaus, ataus;
    ctaus.reserve(order());
    ataus.reserve(order());

    // fill array with taus, in the natural order of the blocks.
    for (typename std::vector< Block<ValueT> >::const_iterator
            bl = blocks_.begin(); bl != blocks_.end(); ++bl) {

        for (std::vector<BlockOperator>::const_iterator
                cop = bl->copers().begin(); cop != bl->copers().end(); ++cop) {
            ctaus.push_back(cop->tau);
        }
        for (std::vector<BlockOperator>::const_iterator
                aop = bl->aopers().begin(); aop != bl->aopers().end(); ++aop) {
            ataus.push_back(aop->tau);
        }
    }

    // stupid O(n^2) algorithm that just "counts" the necessary swaps to get
    // the thing sorted
    unsigned swaps = 0;
    for (unsigned i = 1; i < order(); ++i) {
        for (unsigned j = 0; j != i; ++j) {
            if (ctaus[j] > ctaus[i])
                ++swaps;
            if (ataus[j] > ataus[i])
                ++swaps;
        }
    }

    return swaps % 2 ? -1 : 1;
}

template <typename ValueT>
ValueT Bath<ValueT>::calc_weight() const
{
    ValueT weight = calc_perm_sign();
    for (Bath<ValueT>::bciter it = blocks_.begin(); it != blocks_.end(); ++it) {
        weight *= it->weight();
    }
    return weight;
}

template <typename ValueT>
void Bath<ValueT>::verify(bool recursive, double tol) const
{
    // verify children first
    if (recursive) {
        for (typename std::vector< Block<ValueT> >::const_iterator
                it = blocks_.begin(); it != blocks_.end(); ++it)
            it->verify();
    }

    // Set up visitor array to ensure that each lookup occurs exactly once.
    std::vector< std::vector<bool> > cvisited, avisited;
    for (unsigned bl = 0; bl != nblocks(); ++bl) {
        cvisited.push_back(std::vector<bool>(blocks_[bl].order(), false));
        avisited.push_back(std::vector<bool>(blocks_[bl].order(), false));
    }

    // verify that all entries are unique and the sorting of the lookup by tau
    double ctau = -1., atau = -1.;
    for (unsigned i = 0; i < order(); ++i) {
        const BlockPosition cpos = clookup_[i], apos = alookup_[i];

        if (cpos.block_no >= cvisited.size())
            throw VerificationError("clookup[%d] points to invalid block", i);
        if (cpos.block_pos >= cvisited[cpos.block_no].size())
            throw VerificationError("clookup[%d] points to invalid pos", i);

        if (apos.block_no >= avisited.size())
            throw VerificationError("alookup[%d] points to invalid block", i);
        if (apos.block_pos >= avisited[apos.block_no].size())
            throw VerificationError("alookup[%d] points to invalid pos", i);

        if (cvisited[cpos.block_no][cpos.block_pos])
            throw VerificationError("clookup[%d] is a duplicate", i);
        cvisited[cpos.block_no][cpos.block_pos] = true;

        if (avisited[apos.block_no][apos.block_pos])
            throw VerificationError("alookup[%d] is a duplicate", i);
        avisited[apos.block_no][apos.block_pos] = true;

        const BlockOperator &coper = blocks_[cpos.block_no].copers()[cpos.block_pos];
        if (coper.tau <= ctau)
            throw VerificationError("clookup[%d] breaks strict tau order", i);
        ctau = coper.tau;

        const BlockOperator &aoper = blocks_[apos.block_no].aopers()[apos.block_pos];
        if (aoper.tau <= atau)
            throw VerificationError("alookup[%d] breaks strict tau order", i);
        atau = aoper.tau;
    }

    // verify weight
    ValueT real_weight = calc_weight();
    if (std::abs(weight_ - real_weight) > tol * std::abs(real_weight)) {
        throw VerificationError("wrong bath weight: %g != %g",
                                weight_, real_weight);
    }
}

template <typename BlockMoveT>
BathMove<BlockMoveT>::BathMove()
    : target_(NULL)
{ }

template <typename BlockMoveT>
BathMove<BlockMoveT>::BathMove(Bath<Value> &target, unsigned max_rank)
    : target_(&target),
      block_moves_(),
      rank_(-1),
      max_rank_(max_rank),
      hard_reject_(false),
      ratio_(0.),
      sign_change_(0)
{
    for (unsigned i = 0; i != target.nblocks(); ++i) {
        block_moves_.push_back(BlockMoveT(target.block(i), max_rank));
    }
}


template <typename ValueT>
BathInsertMove<ValueT>::BathInsertMove()
    : BathMove< BlockAppendMove<ValueT> >()
{ }

template <typename ValueT>
BathInsertMove<ValueT>::BathInsertMove(Bath<ValueT> &target, unsigned max_rank)
    : BathMove< BlockAppendMove<ValueT> >(target, max_rank),
      coper_(max_rank),
      aoper_(max_rank),
      cblock_rank_(target.nblocks(), 0),
      ablock_rank_(target.nblocks(), 0),
      cindex_(max_rank),
      aindex_(max_rank),
      clookup_(max_rank),
      alookup_(max_rank)
{ }

template <typename ValueT>
void BathInsertMove<ValueT>::verify_args(
                unsigned rank, const double *ctau, const unsigned *cflavour,
                const double *atau, const unsigned *aflavour) const
{
    const Bath<ValueT> &target = *this->target_;

    double min_tau = 0;
    for (unsigned i = 0; i != rank; ++i) {
        if (cflavour[i] >= target.nflavours())
            throw VerificationError("coper %d illegal flavour", i);
        if (ctau[i] < 0 || ctau[i] >= target.beta())
            throw VerificationError("coper %d illegal tau", i);
        if (ctau[i] < min_tau)
            throw VerificationError("coper %d tau must be sorted", i);
        min_tau = ctau[i];
    }

    min_tau = 0;
    for (unsigned i = 0; i != rank; ++i) {
        if (aflavour[i] >= target.nflavours())
            throw VerificationError("aoper %d illegal flavour", i);
        if (atau[i] < 0 || atau[i] >= target.beta())
            throw VerificationError("aoper %d illegal tau", i);
        if (atau[i] < min_tau)
            throw VerificationError("aoper %d tau must be sorted", i);
        min_tau = atau[i];
    }

}

bool comp_block(const BlockOperator &left, const BlockOperator &right)
{
    return left.block_no < right.block_no;
}

struct SwapTwo
{
    SwapTwo(unsigned max_rank) : max_rank(max_rank) { }

    void operator() (unsigned &left, unsigned &right)
    {
        std::swap(left, right);
        std::swap((&left)[max_rank], (&right)[max_rank]);
    }

    unsigned max_rank;
};


template <typename ValueT>
void BathInsertMove<ValueT>::propose(unsigned rank, double *ctau,
                         unsigned *cflavour, double *atau, unsigned *aflavour)
{
    if (MODE_DEBUG) {
        verify_args(rank, ctau, cflavour, atau, aflavour);
        this->sign_change_ = 0;
    }

    const Bath<ValueT> &target = *this->target_;

    // Prepare the sorting
    std::fill(cblock_rank_.begin(), cblock_rank_.end(), 0);
    std::fill(ablock_rank_.begin(), ablock_rank_.end(), 0);

    // Sort the insertion into the corresponding blocks and construct block
    // moves.  Quantum number violations are rare (we mostly insert segments),
    // therefore we do not check them immediately.
    for (unsigned i = 0; i != rank; ++i) {
        const unsigned block_no = target.block_for_flavour_[cflavour[i]];
        coper_[i] = BlockOperator(target.blocks_[block_no], ctau[i],
                                  cflavour[i] - target.flavour_offset_[block_no],
                                  true, block_no, i);
        ++cblock_rank_[block_no];
    }
    for (unsigned i = 0; i != rank; ++i) {
        const unsigned block_no = target.block_for_flavour_[aflavour[i]];
        aoper_[i] = BlockOperator(target.blocks_[block_no], atau[i],
                                  aflavour[i] - target.flavour_offset_[block_no],
                                  false, block_no, i);
        ++ablock_rank_[block_no];
    }

    // Next, we need to sort to make sure we have the block moves "grouped".
    //
    // Now, here it is important to keep track of the number of permutations.
    // One reason is that for insertion, "cindex" and "aindex" are not unique
    // since we can insert, e.g., two operators before index 0.  So we can
    // think of the insertion as (1) appending the operators to their blocks,
    // (2) move them to the very end (2 * X mod 2 == 0), (3) reorder them
    // according to their tau value (-block_perm_ mod 2), and (4) moving them
    // to their position (2 * X - cindex + aindex mod 2).
    this->block_perm_ = 0;
    this->block_perm_ += tsort(coper_.begin(), coper_.begin() + rank, comp_block);
    this->block_perm_ += tsort(aoper_.begin(), aoper_.begin() + rank, comp_block);

    // Now check quantum numbers, which is equivalent to the inverse matrices
    // for each block staying square and propose the move (if not a no-op
    // rank-0 update).
    this->ratio_ = 1.;
    unsigned start = 0;
    for (unsigned bl = 0; bl != target.nblocks(); ++bl) {
        const unsigned block_rank = cblock_rank_[bl];
        if (block_rank != ablock_rank_[bl]) {
            this->hard_reject_ = true;
            return;
        }
        // Skip blocks which are trivial
        if (block_rank == 0)
            continue;

        BlockAppendMove<ValueT> &block_move = this->block_moves_[bl];
        block_move.propose(block_rank, &coper_[start], &aoper_[start]);
        this->ratio_ *= block_move.ratio();

        start += block_rank;
    }
    assert(start == rank);

    this->hard_reject_ = false;
    this->rank_ = rank;
}

template <typename ValueT>
void BathInsertMove<ValueT>::accept()
{
    assert(!this->hard_reject_);

    Bath<ValueT> &target = *this->target_;

    unsigned num_perm = this->block_perm_;

    unsigned start = 0;
    for (unsigned bl = 0; bl != target.nblocks(); ++bl) {
        // Skip blocks where no move (i.e., rank-0) move was proposed
        const unsigned block_rank = cblock_rank_[bl];
        if (block_rank == 0)
            continue;

        // Cache a couple of things
        const unsigned old_bl_order = target.blocks_[bl].order();

        // Accept the corresponding block move
        this->block_moves_[bl].accept();

        // Prepare the insertion points
        for (unsigned i = 0; i != block_rank; ++i) {
            const BlockPosition blockpos(bl, old_bl_order + i);
            const BlockOperator &coper_curr = coper_[start + i];
            const BlockOperator &aoper_curr = aoper_[start + i];

            const unsigned cindex_curr = target.find(coper_curr.tau, true);
            const unsigned aindex_curr = target.find(aoper_curr.tau, false);

            cindex_[coper_curr.move_pos] = cindex_curr;
            aindex_[aoper_curr.move_pos] = aindex_curr;

            clookup_[coper_curr.move_pos] = blockpos;
            alookup_[aoper_curr.move_pos] = blockpos;

            num_perm += cindex_curr + aindex_curr;
        }
        start += block_rank;
    }
    assert(start == this->rank_);

    this->sign_change_ = num_perm % 2 ? -1 : 1;
    target.weight_ *= this->ratio_ * this->sign_change_;

    // TODO: slow for rank > 1
    for (unsigned i = this->rank_; i != 0; --i) {
        target.clookup_.insert(target.clookup_.begin() + cindex_[i - 1],
                               clookup_[i - 1]);
        target.alookup_.insert(target.alookup_.begin() + aindex_[i - 1],
                               alookup_[i - 1]);
    }

    if (MODE_DEBUG)
        target.verify();
}

template <typename ValueT>
BathRemoveMove<ValueT>::BathRemoveMove()
    : BathMove< BlockRemoveMove<ValueT> >()
{ }

template <typename ValueT>
BathRemoveMove<ValueT>::BathRemoveMove(Bath<ValueT> &target, unsigned max_rank)
    : BathMove< BlockRemoveMove<ValueT> >(target, max_rank),
      cblock_(2 * max_rank, -1),
      ablock_(2 * max_rank, -1),
      blocks_(max_rank),
      cindex_(NULL),
      aindex_(NULL),
      crepl_index_(max_rank, -1),
      crepl_repl_(max_rank, -1),
      arepl_index_(max_rank, -1),
      arepl_repl_(max_rank, -1)
{ }

template <typename ValueT>
void BathRemoveMove<ValueT>::verify_args(unsigned rank, const unsigned *cindex,
                                         const unsigned *aindex) const
{
    const Bath<ValueT> &target = *this->target_;

    if (rank > this->max_rank_)
        throw VerificationError("Rank exceeds maximum acceptable rank");

    unsigned min_index = 0;
    for (const unsigned *curr = cindex; curr != cindex + rank; ++curr) {
        if (*curr >= target.order())
            throw VerificationError("illegal cindex");
        if (*curr < min_index)
            throw VerificationError("cindex breaks ordering");
        min_index = *curr + 1;
    }

    min_index = 0;
    for (const unsigned *curr = aindex; curr != aindex + rank; ++curr) {
        if (*curr >= target.order())
            throw VerificationError("illegal aindex");
        if (*curr < min_index)
            throw VerificationError("aindex breaks ordering");
        min_index = *curr + 1;
    }
}

template <typename ValueT>
void BathRemoveMove<ValueT>::propose(unsigned rank, unsigned *cindex,
                                     unsigned *aindex)
{
    if (MODE_DEBUG) {
        verify_args(rank, cindex, aindex);
        this->sign_change_ = 0;
    }

    const Bath<ValueT> &target = *this->target_;

    // Prepare the arrays to be sorted.  Since we need to pass an continuous
    // array of block indices into the block move, the set-up is a little bit
    // evil: for cindex[i], we store the block number in cblock[i] and the
    // block position in cblock[i + max_rank_]
    for (unsigned i = 0; i != rank; ++i) {
        const BlockPosition pos = target.clookup_[cindex[i]];
        cblock_[i] = pos.block_no;
        cblock_[i + this->max_rank_] = pos.block_pos;
    }
    for (unsigned i = 0; i != rank; ++i) {
        const BlockPosition pos = target.alookup_[aindex[i]];
        ablock_[i] = pos.block_no;
        ablock_[i + this->max_rank_] = pos.block_pos;
    }

    // Sort the removals by the corresponding blocks as input for the block
    // moves. Quantum number violations are rare (we mostly insert segments),
    // therefore we do not check them immediately.  We record the number of
    // permutations for later use.
    SwapTwo swapper(this->max_rank_);
    this->num_perms_ =
            tsort(cblock_.begin(), cblock_.begin() + rank, std::less<unsigned>(), swapper) +
            tsort(ablock_.begin(), ablock_.begin() + rank, std::less<unsigned>(), swapper);

    // Determine the block boundaries
    this->ratio_ = 1.;

    unsigned start = 0;
    unsigned ibl;
    for (ibl = 0; ; ++ibl) {
        unsigned block_no = cblock_[start];

        // Search where the indices for the current block end and check if it
        // is consistent for creation and annihilation indices.
        unsigned end;
        for (end = start; end != rank; ++end) {
            if (cblock_[end] != block_no)
                break;
            if (ablock_[end] != block_no) {
                this->hard_reject_ = true;
                return;
            }
        }
        blocks_[ibl] = block_no;

        // Propose the underlying move for the current block
        BlockRemoveMove<ValueT> &block_move = this->block_moves_[block_no];
        block_move.propose(end - start, &cblock_[this->max_rank_ + start],
                           &ablock_[this->max_rank_ + start]);
        this->ratio_ *= block_move.ratio();

        if (end == rank)
            break;
        start = end;
    }
    num_blocks_ = ibl + 1;

    this->hard_reject_ = false;
    this->rank_ = rank;
    this->cindex_ = cindex;
    this->aindex_ = aindex;
}

template <typename ValueT>
void BathRemoveMove<ValueT>::accept()
{
    assert(!this->hard_reject_);

    Bath<ValueT> &target = *this->target_;

    unsigned iindex = 0;
    for (unsigned ibl = 0; ibl != num_blocks_; ++ibl) {
        // Find out the corresponding block move
        unsigned bl = blocks_[ibl];
        BlockRemoveMove<ValueT> &block_move = this->block_moves_[bl];
        const unsigned block_rank = block_move.rank();

        // Prepare the reseatings
        for (unsigned i = 0; i != block_rank; ++i) {
            // The operators to be removed are going to be replaced by the last
            // operator stored in repl for the given block.  Since we need to
            // update their lookup position, we need first to perform a reverse
            // lookup in  order to find their position in the lookup array.
            const unsigned crepl_index = target.crlookup(
                                    BlockPosition(bl, block_move.crepl()[i]));
            const unsigned arepl_index = target.arlookup(
                                    BlockPosition(bl, block_move.arepl()[i]));

            // Only store those replacements in an array and defer the actual
            // resetting (and the block move acceptance, which changes the
            // tau values) to later, otherwise we may break the lookup of the
            // next elements.
            crepl_index_[iindex] = crepl_index;
            crepl_repl_[iindex] = block_move.cblockpos()[i];
            arepl_index_[iindex] = arepl_index;
            arepl_repl_[iindex] = block_move.ablockpos()[i];
            ++iindex;
        }
    }
    assert(iindex == this->rank_);

    // Accept the block moves.
    for (unsigned ibl = 0; ibl != num_blocks_; ++ibl) {
        unsigned bl = blocks_[ibl];
        BlockRemoveMove<ValueT> &block_move = this->block_moves_[bl];

        assert(block_move.rank() != 0);
        assert(!block_move.hard_reject());
        block_move.accept();
    }

    // Now perform the re-seating of the lookups in one go.
    for (unsigned i = 0; i != this->rank_; ++i) {
        target.clookup_[crepl_index_[i]].block_pos = crepl_repl_[i];
        target.alookup_[arepl_index_[i]].block_pos = arepl_repl_[i];
    }

    // Treat the sign change:
    //  (1) We first "undo" the removal of cindex_[] and aindex_[] by permuting
    //      them to the last indices (since they are sorted, we do not need to
    //      permute them amongst themselves).
    //  (2) Now this sorting does not agree with the blocks however, so we
    //      commute through the permutation arrays R, C by first sorting the
    //      indices according to the block number they map to (already included
    //      in num_perm_ as filled in the proposal).
    //  (3) Now we would also have to account for the fact that instead of
    //      removing the last element from the corresponding blocks, we are
    //      instead replacing (rindex, aindex) with (rrepl, crepl).  The
    //      determinant stores this permutation factor in sign_change() rather
    //      than ratio(), so not including sign_change() implicitly fixes this.
    for (unsigned i = 0; i != this->rank_; ++i) {
        num_perms_ += cindex_[i] + aindex_[i];
    }
    this->sign_change_ = num_perms_ % 2 ? -1 : 1;

    // Update weight
    target.weight_ *= this->ratio_ * this->sign_change_;

    // TODO: slow for rank > 1
    for (unsigned i = this->rank_; i != 0; --i) {
        target.clookup_.erase(target.clookup_.begin() + cindex_[i - 1]);
        target.alookup_.erase(target.alookup_.begin() + aindex_[i - 1]);
    }

    if (MODE_DEBUG)
        target.verify();
}


template <typename ValueT>
BathRecompute<ValueT>::BathRecompute(Bath<ValueT> &target)
{
    for (unsigned i = 0; i != target.nblocks(); ++i) {
        block_recomps_.push_back(BlockRecompute<ValueT>(target.block(i)));
    }
}

template <typename ValueT>
void BathRecompute<ValueT>::propose()
{
    assert(block_recomps_.size() != 0);

    error_ = 0;
    for (unsigned i = 0; i != block_recomps_.size(); ++i) {
        block_recomps_[i].propose();
        error_ += block_recomps_[i].error();     // play it safe
    }
}

template <typename ValueT>
void BathRecompute<ValueT>::accept()
{
    assert(block_recomps_.size() != 0);

    for (unsigned i = 0; i != block_recomps_.size(); ++i)
        block_recomps_[i].accept();
}


// ------------------------------- ESTIMATORS ---------------------------------

template <typename ValueT>
GtauEstimator<ValueT>::GtauEstimator(const Bath<ValueT> &source, unsigned ntau_bins)
    : source_(&source),
      accum_size_(0),
      result_size_(0)
{
    for (unsigned i = 0; i != source.nblocks(); ++i) {
        this->blocks_.push_back(
                GtauBlockEstimator<ValueT>(source.blocks_[i], ntau_bins));
        this->accum_offset_.push_back(accum_size_);
        this->result_offset_.push_back(result_size_);

        accum_size_ += blocks_.rbegin()->accum_size();
        result_size_ += blocks_.rbegin()->result_size();
    }
}

template <typename ValueT>
void GtauEstimator<ValueT>::estimate(ValueT *accum, ValueT weight)
{
    assert(!blocks_.empty());
    const Bath<ValueT> &source = *this->source_;

    for (unsigned i = 0; i != source.nblocks(); ++i)
        blocks_[i].estimate(&accum[accum_offset_[i]], weight);
}

template <typename ValueT>
void GtauEstimator<ValueT>::postprocess(ValueT *result, const ValueT *accum,
                                        ValueT sum_weights)
{
    assert(!blocks_.empty());
    const Bath<ValueT> &source = *this->source_;

    for (unsigned i = 0; i != source.nblocks(); ++i)
        blocks_[i].postprocess(&result[result_offset_[i]],
                               &accum[accum_offset_[i]], sum_weights);
}


template <typename ValueT>
GiwEstimator<ValueT>::GiwEstimator(const Bath<ValueT> &source, unsigned niwf,
                                   bool use_nfft)
    : source_(&source),
      accum_size_(0),
      result_size_(0)
{
    assert(niwf > 0);
    for (unsigned i = 0; i != source.nblocks(); ++i) {
        this->blocks_.push_back(
                GiwBlockEstimator<ValueT>(source.blocks_[i], niwf, use_nfft));
        this->accum_offset_.push_back(accum_size_);
        this->result_offset_.push_back(result_size_);

        accum_size_ += blocks_.rbegin()->accum_size();
        result_size_ += blocks_.rbegin()->result_size();
    }
}

template <typename ValueT>
void GiwEstimator<ValueT>::estimate(AccumulatorT accum, ValueT weight)
{
    assert(!blocks_.empty());
    const Bath<ValueT> &source = *this->source_;

    for (unsigned i = 0; i != source.nblocks(); ++i)
        blocks_[i].estimate(&accum[accum_offset_[i]], weight);
}

template <typename ValueT>
void GiwEstimator<ValueT>::postprocess(ResultT result, AccumulatorT accum,
                                       ValueT sum_weights)
{
    assert(!blocks_.empty());
    const Bath<ValueT> &source = *this->source_;

    for (unsigned i = 0; i != source.nblocks(); ++i)
        blocks_[i].postprocess(&result[result_offset_[i]],
                               &accum[accum_offset_[i]], sum_weights);
}


template <typename ValueT>
G4iwEstimator<ValueT>::G4iwEstimator(const Bath<ValueT> &source, unsigned niwf,
                                     unsigned niwb, bool use_nfft)
    : niwf_(niwf),
      niwb_(niwb),
      source_(&source),
      plan_(source.npairs(), 2, 4 * (niwf + niwb - 1), use_nfft)
{
    assert(niwf > 0);
    assert(niwb > 0);
}

template <typename ValueT>
void G4iwEstimator<ValueT>::estimate(AccumulatorT accum, ValueT weight,
                                     const ValueT *hartree)
{
    perform_ndft_2d(hartree);
    assemble(accum, weight);
}

template <typename ValueT>
void G4iwEstimator<ValueT>::perform_ndft_2d(const ValueT *hartree)
{
    const Bath<ValueT> &source = *this->source_;

    // First, perform the Fourier transform the bath Green's function to iw
    // (M. Wallerberger, PhD thesis, eq. 3.45):  Note that we do this in the
    // impurity picture (transposition of M_ij):
    //
    //     G_AB(iv, iv') =
    //         sum_ij exp(iv tau_i - iv' tau'_j) M_ji delta(A,A_i) delta(B,B_i)
    //
    // We need to keep both frequencies here (only the summation ofver the
    // inner degrees of freedom of a diagram restores its conservation laws).
    plan_.reset();

    unsigned vec_offset = 0;
    unsigned h_offset = 0;
    for (unsigned ibl = 0; ibl != source.nblocks(); ++ibl) {
        const Block<ValueT> &block = source.block(ibl);
        const unsigned order = block.det().order();

        // convert to Eigen matrix (probably unnecessary)
        typename DeterminantTraits<ValueT>::BufferConstMap M(
                    block.det().invmat(), order, order, block.det().capacity());

        for (unsigned i = 0; i != order; ++i) {
            const BlockOperator &coper = block.copers()[i];
            const unsigned coffset = coper.block_flavour * source.nflavours();

            for (unsigned j = 0; j != order; ++j) {
                const BlockOperator &aoper = block.aopers()[j];

                plan_.add(vec_offset + coffset + aoper.block_flavour,
                          M(j, i),
                          coper.tau/(2 * source.beta()),
                          -aoper.tau/(2 * source.beta())
                          );
            }
        }
        vec_offset += block.npairs();
        h_offset += block.nflavours();
    }
    assert(vec_offset == source.npairs());
    assert(h_offset == source.nflavours());
    plan_.compute();
}

static inline
unsigned g2_index(unsigned comp, unsigned iv, unsigned ivp, unsigned ntotfreq)
{
    // Helper routine to extract components for the assembly. Since the Fourier
    // transform was done on an extended grid (only odd frequencies count)
    //
    //   G_AB(iv_m, iv_n)
    //      = (A * source.npairs() + B) * 4 * nfreq * nfreq +
    //                      (2*m + 1 + nfreq)) * 2 * nfreq + 2*n + 1 + nfreq
    //
    assert(iv && iv < ntotfreq);
    assert(ivp && ivp < ntotfreq);
    return (comp * ntotfreq + iv) * ntotfreq + ivp;
}

template <typename ValueT>
void G4iwEstimator<ValueT>::assemble(AccumulatorT accum, ValueT weight)
{
    const Bath<ValueT> &source = *this->source_;

    // Now assemble the bricks (ibid, eq. 3.46)
    //
    //     G_ABCD(iv, iv', iw) =  G_AB(iv + iw, iv) G_CD(iv', iv' + iw)
    //                                  - G_AD(iv + iw, iv' + iw) G_CB(iv', iv)
    //
    const unsigned ntotfreq = 4 * (niwf_ + niwb_ - 1);
    const unsigned iv_start = 2 * niwb_ - 1;
    const unsigned iv_stop = iv_start + 4 * niwf_;
    const int iw_start = -2 * int(niwb_) + 2;
    const int iw_stop = 2 * int(niwb_);

    unsigned comp = 0;
    for (unsigned cd = 0; cd != source.npairs(); ++cd) {
        for (unsigned ivp = iv_start; ivp != iv_stop; ivp += 2) {
            for (int iw = iw_start; iw != iw_stop; iw += 2) {
                const std::complex<double> gcd =
                        plan_.f_hat()[g2_index(cd, ivp, ivp + iw, ntotfreq)];
                for (unsigned ab = 0; ab != source.npairs(); ++ab) {
                    for (unsigned iv = iv_start; iv != iv_stop; iv += 2) {
                        const std::complex<double> gab =
                                plan_.f_hat()[g2_index(ab, iv + iw, iv, ntotfreq)];

                        accum[comp++] += weight * gab * gcd;
                    }
                }
            }
        }
    }

    for (unsigned cb = 0; cb != source.npairs(); ++cb) {
        for (unsigned iv = iv_start; iv != iv_stop; iv += 2) {
            for (unsigned ivp = iv_start; ivp != iv_stop; ivp += 2) {
                const std::complex<double> gcb =
                        plan_.f_hat()[g2_index(cb, ivp, iv, ntotfreq)];
                for (unsigned ad = 0; ad != source.npairs(); ++ad) {
                    for (int iw = iw_start; iw != iw_stop; iw += 2) {
                        const std::complex<double> gad =
                                plan_.f_hat()[g2_index(ad, iv + iw, ivp + iw,
                                                       ntotfreq)];

                        accum[comp++] += weight * gcb * gad;
                    }
                }
            }
        }
    }
    assert(comp == accum_size());
}

template <typename ValueT>
void G4iwEstimator<ValueT>::postprocess(ResultT result, ConstAccumulatorT accum,
                                        ValueT sum_weights)
{
    const Bath<ValueT> &source = *this->source_;
    const ValueT scaling = 1./sum_weights * 1./source.beta();
    const unsigned ntotb = 2 * niwb_ - 1;
    const unsigned ntotf = 2 * niwf_;

    unsigned comp = 0;

    // Transpose from (CD,iv',iw,AB,iv) to (AB,CD,iw,iv,iv')
    for (unsigned ab = 0; ab != source.npairs(); ++ab) {
        for (unsigned cd = 0; cd != source.npairs(); ++cd) {
            for (unsigned iw = 0; iw != ntotb; ++iw) {
                for (unsigned iv = 0; iv != ntotf; ++iv) {
                    for (unsigned ivp = 0; ivp != ntotf; ++ivp) {
                        unsigned loc = (((cd * ntotf + ivp) * ntotb + iw)
                                        * source.npairs() + ab) * ntotf + iv;
                        result[comp++] = accum[loc] * scaling;
                    }
                }
            }
        }
    }
    accum += comp;

    // Transpose from (CB,iv,iv',AD,iw) to (AD,CB,iw,iv,iv')
    for (unsigned ad = 0; ad != source.npairs(); ++ad) {
        for (unsigned cb = 0; cb != source.npairs(); ++cb) {
            for (unsigned iw = 0; iw != ntotb; ++iw) {
                for (unsigned iv = 0; iv != ntotf; ++iv) {
                    for (unsigned ivp = 0; ivp != ntotf; ++ivp) {
                        unsigned loc = (((cb * ntotf + iv) * ntotf + ivp)
                                        * source.npairs() + ad) * ntotb + iw;
                        result[comp++] = accum[loc] * scaling;
                    }
                }
            }
        }
    }
    assert(comp == result_size());
}
