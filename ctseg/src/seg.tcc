#ifndef _SEG_HH
#  error "Do not include this file directly, include seg.hh instead"
#endif

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "util.hh"

static unsigned popcount(uint64_t x, unsigned uptothis) {
    const uint64_t all = 0xFFFFFFFFFFFFFFFF;  // all bits set
    const uint64_t m1  = 0x5555555555555555;  // binary: 0101 ...
    const uint64_t m2  = 0x3333333333333333;  // binary: 00110011 ...
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0f;  // binary: 0000111100001111 ...
    const uint64_t h01 = 0x0101010101010101;  // sum of 256 pow 0,1,2,3...

    assert (uptothis <= 64);
    if (!uptothis)
        return 0;
    x &= all >> (64 - uptothis);       // mask out relevant bits
    x -= (x >> 1) & m1;                // contract 2 bits
    x = (x & m2) + ((x >> 2) & m2);    // contract 2 bit bins into 4 bits
    x = (x + (x >> 4)) & m4;           // etc.
    return (x * h01) >> 56;            // sum over bins
}

template <typename ValueT>
void SegmentTrace<ValueT>::cache_expterms(bool shift)
{
    const unsigned nflavours = this->nflavours_, nfock = 1 << nflavours;

    this->expterms_.reserve(nfock);
    ValueT minterm = 1e100;
    unsigned minalpha = 0;

    // TODO this is terribly inefficient
    for (unsigned alpha = 0; alpha < nfock; ++alpha) {
        ValueT currterm = 0;
        for (unsigned i = 0, imask = 1; i < nflavours; ++i, imask <<= 1) {
            if (!(alpha & imask))
                continue;
            currterm += this->energies_[i];
            for (unsigned j = 0, jmask = 1; j < nflavours; ++j, jmask <<= 1) {
                if (!(alpha & jmask))
                    continue;
                currterm += 0.5 * this->u_matrix_[i * nflavours + j];
            }
        }
        this->expterms_.push_back(-currterm);
        if (currterm < minterm) {
            minterm = currterm;
            minalpha = alpha;
        }

    }
    if (shift) {
        for (unsigned alpha = 0; alpha < nfock; ++alpha) {
            this->expterms_[alpha] += minterm;
        }
        this->e0_ = minterm;
        this->alpha0_ = minalpha;
    } else {
        this->e0_ = 0;
        this->alpha0_ = 0;
    }

    if (false) {
        for (unsigned alpha = 0; alpha < nfock; ++alpha)
            fprintf(stderr, "%d\t%g\n", alpha, this->expterms_[alpha]);
        abort();
    }
}

template <typename ValueT>
SegmentTrace<ValueT>::SegmentTrace(unsigned nflv, double beta,
                                   const ValueT *energies, const ValueT *u_matrix)
    : beta_(beta),
      nflavours_(nflv),
      nsegments_(nflv, 0),
      u_matrix_(u_matrix, u_matrix + nflv * nflv),
      energies_(energies, energies + nflv),
      expterms_(), alpha0_(), e0_(),
      timeord_sign_(), pauli_sign_(), abs_weight_(),
      empty_state_(), opers_()
{
    assert(beta > 0);

    this->cache_expterms(true);
    this->reset();
}


template <typename ValueT>
void SegmentTrace<ValueT>::reset()
{
    // create the zero operator
    this->opers_.clear();

    // reset the number of segments
    std::fill(this->nsegments_.begin(), this->nsegments_.end(), 0);

    // TODO: start at a state with minimum energy
    this->empty_state_ = 0;

    // initialise the weight
    this->timeord_sign_ = this->calc_timeord_sign();
    this->pauli_sign_ = this->calc_pauli_sign();
    this->abs_weight_ = this->calc_absweight();

    if (MODE_DEBUG)
        this->verify();
}

template <typename ValueT>
void SegmentTrace<ValueT>::verify(double tol) const
{
    try {
        if (this->opers_.size() % 2)
            throw VerificationError("operators do not come in pairs");

        // first verify the internal consistency of the thing. By keeping track
        // of the previous element, we can model the wrapping over beta
        double prevtau = 0;
        unsigned prevflv = nopers() ? opers_.rbegin()->flavour : 0;
        uint64_t prevocc_less = nopers() ? opers_.rbegin()->occ_less : 0;

        for (std::vector<LocalOper>::const_iterator curr = opers_.begin();
                curr != opers_.end(); ++curr) {

            unsigned opno = curr - this->opers_.begin();
            if (curr->tau < prevtau || curr->tau >= this->beta_) {
                throw VerificationError("local operator %d - illegal tau %g",
                                        opno, curr->tau);
            }
            if (curr->flavour >= this->nflavours_) {
                throw VerificationError("local operator %d - illegal flv %d",
                                        opno, curr->flavour);
            }
            if (bool(curr->occ_less & 1u << curr->flavour) == curr->effect) {
                throw VerificationError("local operator %d - bad effect",
                                        opno);
            }
            if ((prevocc_less ^ curr->occ_less) != 1u << prevflv) {
                throw VerificationError("local operator %d - occupactions"
                                        "mismatch left %lx vs. right %lx.",
                                        opno-1, curr->occ_less, prevocc_less);
            }

            prevtau = curr->tau;
            prevflv = curr->flavour;
            prevocc_less = curr->occ_less;
        }

        // now verify the weight
        if (this->timeord_sign_ != this->calc_timeord_sign()) {
            throw VerificationError("wrong time-ordering sign: %+d",
                                    this->timeord_sign_);
        }
        if (this->pauli_sign_ != this->calc_pauli_sign()) {
            throw VerificationError("wrong fermionic exchange sign: %+d",
                                    this->pauli_sign_);
        }
        ValueT rweight = this->calc_absweight();
        if (std::abs(this->abs_weight_ - rweight) > tol * fabs(rweight)) {
            throw VerificationError("wrong abs weight: %g (correct weight: %g)",
                                    this->abs_weight_, rweight);
        }
    } catch (const VerificationError &e) {
        fprintf(stderr, "\n\nCurrent loal trace:\n");
        dump();
        fprintf(stderr, "\nVerification Error: %s\n\n", e.what());
        throw;
    }
}

template <typename ValueT>
void SegmentTrace<ValueT>::dump_track(unsigned flavour, unsigned width) const
{
    FILE *out = stderr;

    uint64_t mask = 1 << flavour;

    // correct for the boundaries
    fputc('[', out);
    width -= 2;

    // Store some stuff we need for the printing
    unsigned caret = 0;
    unsigned crampoint = nopers() >= width ? 0 : width - nopers();
    uint64_t occ_less = nopers() ? opers_[0].occ_less : empty_state_;

    // We are printing the operators in 'mathematical' order, so the tau axis
    // goes from the right (0) to the left (beta).  This means we need to
    // iterate in reverse.
    for (std::vector<LocalOper>::const_reverse_iterator curr = opers_.rbegin();
            curr != opers_.rend(); ++curr) {

        // if we have exceeded our available space, then just end
        if (caret > crampoint && caret + 4 >= width) {
            fputs(" ...]", out);
            return;
        }

        // We want to display as many distinct operators as possible.  This
        // means that we have widen the grid where there are a lot of operators
        // present. However, this also means that we need to make sure that we
        // can fit the rest of the operators in the remaining space.
        unsigned target = (1 - curr->tau/beta_) * width;

        // Skip places if we have the room to do so
        const char fill = occ_less & mask ? '-' : ' ';
        for (; caret < target && caret < crampoint; ++caret)
            fputc(fill, out);

        // Print operators as the corresponding "slope" to the density of the
        // respective flavour.  Since the tau axis runs from the right to the
        // left, creation operators are signified by '\\'.
        const char opchar = curr->effect? '\\' : '/';
        fputc(curr->flavour == flavour ? opchar : fill, out);

        // update previous stuff
        ++caret;
        ++crampoint;
        occ_less = curr->occ_less;
    }

    // Skip the stuff behind the last operator to the boundary
    char fill = occ_less & mask ? '-' : ' ';
    for (; caret < width; ++caret)
        fputc(fill, out);

    fputc(']', out);
}

template <typename ValueT>
void SegmentTrace<ValueT>::dump(DumpParts what, unsigned width) const
{
    FILE *out = stderr;

    double scaling = (73. - this->nopers())/this->beta_;
    if (scaling < 0)
        scaling = 0;

    std::vector<ValueT> fillings(nflavours());
    if (what & SEG_TRACKS) {
        unsigned track_width = width - 1;  // newline
        if (what & SEG_FILLING) {
            track_width -= 5;
            OccupationEstimator<ValueT> occ(*this, 1);
            occ.estimate(&fillings[0], 1./beta_);
        }
        if (what & SEG_ORDER)
            track_width -= 5;

        for (unsigned flavour = 0; flavour < this->nflavours_; ++flavour) {
            dump_track(flavour, track_width);
            if (what & SEG_FILLING)
                fprintf(out, " %4.2f", fillings[flavour]);
            if (what & SEG_ORDER)
                fprintf(out, " %4d", nsegments_[flavour]);

            fputc('\n', out);
        }
    }
    if (what & SEG_INFOLINE) {
        fprintf(stderr, "nopers = %d, abs_weight = %g, T = %+d, P = %+d\n",
                this->nopers(), this->abs_weight_, this->timeord_sign_,
                this->pauli_sign_);
    }
}

template <typename ValueT>
unsigned SegmentTrace<ValueT>::find(double tau) const
{
    assert(tau >= 0 && tau < this->beta_);
    unsigned lower = 0, upper = this->nopers();

    if (!nopers())
        return 0;

    // run-of-the-mill binary search (from the left - insertion points)
    for (;;) {
        // determine centre and double lookup to get sorted array
        const unsigned centre = (lower + upper)/2;
        const double tau_centre = this->opers_[centre].tau;
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
ValueT SegmentTrace<ValueT>::hartree(unsigned npos) const
{
    assert(npos < this->nopers());

    const LocalOper *subject = &this->opers_[npos];
    // assumes that U is symmetric.
    const ValueT *urow = &this->u_matrix_[subject->flavour * this->nflavours_];
    uint64_t mask = 1;
    ValueT result = 0;
    if (subject->effect) {
        for (unsigned i = 0; i < this->nflavours_; ++i, mask <<= 1) {
            if (subject->occ_less & mask)
                result += urow[i];
        }
    } else {
        ++subject;   // to the left! to the left!
        for (unsigned i = 0; i < this->nflavours_; ++i, mask <<= 1) {
            if (!(subject->occ_less & mask))
                result += urow[i];
        }
    }
    return result;
}

template <typename ValueT>
ValueT SegmentTrace<ValueT>::calc_absweight() const
{
    if (!nopers()) {
        return exp(this->expterms_[empty_state_] * this->beta_);
    }

    ValueT action = 0;

    // take care of the wrapping around beta: because we know that the trace
    // wraps around, we can just add the tau part before beta to the first
    // operators contribution
    double prevtau = this->opers_.rbegin()->tau - this->beta_;
    for (std::vector<LocalOper>::const_iterator curr = this->opers_.begin();
            curr != this->opers_.end(); ++curr) {
        // ensures that there is no array overrun
        assert(curr->occ_less < (1u << this->nflavours_));
        // now compute the weight
        action += this->expterms_[curr->occ_less] * (curr->tau - prevtau);
        prevtau = curr->tau;
    }
    return exp(action);
}

template <typename ValueT>
int SegmentTrace<ValueT>::calc_pauli_sign() const
{
    unsigned pauli_perm = 0;

    // Do not consider guard operator in this calculation
    for (std::vector<LocalOper>::const_iterator curr = this->opers_.begin();
            curr != this->opers_.end(); ++curr) {
        pauli_perm += popcount(curr->occ_less, curr->flavour);
    }
    return pauli_perm % 2 ? -1 : 1;
}

template <typename ValueT>
int SegmentTrace<ValueT>::calc_timeord_sign() const
{
    // This clever little function in principle works with the cumulative
    // sum E of the effects e (False -> 0, True -> 1), in reverse order.
    // Since the "right" order is e' = [True, False, True, False, ...], the
    // corresponding "right" cumulative sum is E' = [1, 1, 2, 2, ...].  Now
    // any non-trivial transposition of two neighbouring elements adds or
    // subtracts 1 from a single element of E.  Therefore, the number of
    // required transpositions is the sum of |E - E'|, which mod 2 is the
    // same as the sum of E.
    unsigned ncrea = 0, timeord_perm = 0;
    for (std::vector<LocalOper>::const_reverse_iterator curr =
            this->opers_.rbegin(); curr != this->opers_.rend(); ++curr) {
        ncrea += curr->effect ? 1 : 0;
        timeord_perm += ncrea;
    }
    return timeord_perm % 2 ? -1 : 1;
}

template <typename ValueT>
double SegmentTrace<ValueT>::spacing(unsigned start_pos, double start_tau,
                                     unsigned flavour, bool effect) const
{
    assert(start_pos <= nopers());
    assert(0 <= start_tau && start_tau < this->beta_);
    assert(flavour <= nflavours_);

    typedef std::vector<LocalOper>::const_iterator opciter_t;

    const opciter_t start = this->opers_.begin() + start_pos;

    for (opciter_t curr = start; curr != this->opers_.end(); ++curr) {
        if (curr->flavour == flavour && curr->effect == effect)
            return curr->tau - start_tau;
    }
    for (opciter_t curr = this->opers_.begin(); curr != start; ++curr) {
        if (curr->flavour == flavour && curr->effect == effect)
            return curr->tau + this->beta_ - start_tau;
    }
    return this->beta_;
}

template <typename ValueT>
unsigned SegmentTrace<ValueT>::find_end(unsigned pos_start) const
{
    assert(pos_start < nopers());

    const unsigned flavour = opers_[pos_start].flavour;
    unsigned pos_end = pos_start + 1;
    for (; pos_end < nopers(); ++pos_end) {
        if (opers_[pos_end].flavour == flavour)
            return pos_end;
    }
    for (pos_end = 0; pos_end != pos_start; ++pos_end) {
        if (opers_[pos_end].flavour == flavour)
            return pos_end;
    }
    throw std::runtime_error("Segment does not end");
}

template <typename ValueT>
void SegmentTrace<ValueT>::verify_mask(unsigned nmask, const MaskPoint *mask) const
{
    if (nmask % 2 || !nmask)
        throw VerificationError("mask points do not come in pairs");

    double prevtau = 0;
    unsigned previdx = 0;
    for (const MaskPoint *prev = mask + nmask - 1, *curr = mask;
            curr != mask + nmask; prev = curr, ++curr) {
        unsigned opno = curr - mask;
        if (curr->tau < prevtau || curr->tau >= this->beta_) {
            throw VerificationError("mask point %d - illegal tau %g",
                                    opno, curr->tau);
        }
        if (curr->flavour >= this->nflavours_) {
            throw VerificationError("mask point %d - illegal flavour %d",
                                    opno, curr->flavour);
        }
        if (curr->index < previdx || curr->index > this->nopers()) {
            throw VerificationError("mask point %d - illegal index %d",
                                    opno, curr->index);
        }
        if (bool(curr->occ_less & 1u << curr->flavour) == curr->effect) {
            throw VerificationError("mask point %d - bad effect",
                                    opno);
        }
        if ((prev->occ_less ^ curr->occ_less) != 1u << prev->flavour) {
            throw VerificationError("mask point %d - mask corrupt to the left",
                                    opno);
        }
        prevtau = curr->tau;
        previdx = curr->index;
    }
}

template <typename ValueT>
ValueT SegmentTrace<ValueT>::abs_ratio(unsigned nmask, const MaskPoint *mask) const
{
    // Creating a mask is hard enough, make sure its okay before we proceed
    if (MODE_DEBUG)
        verify_mask(nmask, mask);

    // The basic idea here is the following:
    //
    //   1. `this->opers_` stores the information about the trace in a
    //      non-equidistant grid. Thus, if the k-th bit in the bitfield
    //      `this->opers_[i].occ_less` is set, the k'th flavour is occupied
    //      to the right of that operator.
    //
    //   2. `mask` is another non-equidistant grid that describes the *changes*
    //      made to the grid: if the k-th bit of `mask[i].occ_less` is set, the
    //      k-th flavour should be changed from set to unset or vice-versa for
    //      any grid points affected.  (Note that this means that a creation
    //      operator for flavour k in the mask may also change the k-th mask
    //      bit to "off" -- this marks the end of an antisegment then).
    //
    ValueT daction = 0;

    unsigned curr_idx = 0;
    uint64_t curr_occ = nopers() ? opers_[0].occ_less : empty_state_;
    double curr_tau = 0;

    for (const MaskPoint *m = mask; m != mask + nmask; ++m) {
//        printf("MASK: abs %g(%d) %c %d %lx\n",
//               m->tau, m->index, m->effect ? 'S' : 'E', m->flavour, m->occ_less);


        // First, check if the mask has any effect at all (any set bits).  If
        // not, we can skip right to the next point.  This is checked
        // explicitly because the changes will typically be very local.
        if (!m->occ_less) {
            curr_idx = m->index;
            curr_tau = m->tau;
            if (curr_idx)
                curr_occ = this->opers_[curr_idx % nopers()].occ_less;
            continue;
        }

        // Now go through all (parts of) grid segments except the last and
        // accumulate -Delta S.
        for (; curr_idx != m->index; ++curr_idx) {
            const LocalOper *curr = &this->opers_[0] + curr_idx;
            curr_occ = curr->occ_less;
            daction += (this->expterms_[curr_occ ^ m->occ_less] -
                        this->expterms_[curr_occ]) * (curr->tau - curr_tau);
            curr_tau = curr->tau;
        }

        if (curr_idx)
            curr_occ = this->opers_[curr_idx % nopers()].occ_less;

        // We have to handle the part between the last enclosed grid point and
        // the next mask point explicitly.  Note also that because we
        // initialised curr_occ to the empty state in the case of no operators,
        // this takes care of this special case.
        daction += (this->expterms_[curr_occ ^ m->occ_less] -
                    this->expterms_[curr_occ]) * (m->tau - curr_tau);
        curr_tau = m->tau;
    }

    // We are still left with the part from the last mask point onwards. This
    // is indeed just a copy of the above loop until the last operator and for
    // the mask of the *first* mask point (the mask also wraps around beta).
    const MaskPoint *m = mask;
    for (; curr_idx != nopers(); ++curr_idx) {
        const LocalOper *curr = &this->opers_[0] + curr_idx;
        curr_occ = curr->occ_less;
        daction += (this->expterms_[curr_occ ^ m->occ_less] -
                    this->expterms_[curr_occ]) * (curr->tau - curr_tau);
        curr_tau = curr->tau;
    }

    // The last part remaining now is the part from the last operator onwards
    // to beta.  Again this is just a copy from above, but with both the first
    // operators and the first mask points' masks.
    if (curr_idx)
        curr_occ = this->opers_[0].occ_less;

    daction += (this->expterms_[curr_occ ^ m->occ_less] -
                this->expterms_[curr_occ]) * (this->beta_ - curr_tau);

//    fprintf(stderr, "daction: %g; ratio: %g\n", daction, std::exp(daction));
    return std::exp(daction);
}

template <typename ValueT>
void SegmentTrace<ValueT>::insert(unsigned nmask, const MaskPoint *mask)
{
    if (MODE_DEBUG)
        verify_mask(nmask, mask);
    if (!nmask)
        return;

    // Initialise occupancies and mask to handle the wrapping around over beta
    // and (in the case of the grid) also a completely empty trace
    uint64_t curr_occ = nopers() ? this->opers_[0].occ_less : empty_state_;
    uint64_t curr_mask = mask[0].occ_less;
    unsigned curr_index = nopers();

    // Make space for new operators -- note that this changes nopers()
    opers_.resize(nopers() + nmask);

    // We now have to merge the mask into the grid.  We do this from the back
    // to the front since otherwise operators moved override operators we need
    // to move later.
    for (unsigned shift = nmask; shift != 0; --shift) {
        // Identify the next mask point to be merged into the grid.
        const MaskPoint &next_point = mask[shift - 1];

//        printf("MASK: ins %g(%d) %c %d %lx\n",
//                next_point.tau, next_point.index, next_point.effect ? 'S' : 'E',
//                next_point.flavour, next_point.occ_less);

        // For the (n-1)-th mask point, we need to shift all operators starting
        // from the insertion point (.index) up to the previous insertion point
        // by n places.  (Range denoted by start, end -- we use pointers in
        // order to more easily work with memmove.)
        //
        // The occupancies need to be changed -- however, since both grid and
        // mask record to the *right* (lower times/indices), we need to change
        // it w.r.t. to the n-th mask point (or the 0-th in the case of the
        // last point)
        LocalOper *const start = &opers_[0] + next_point.index;
        LocalOper *const end = &opers_[0] + curr_index;

        if (curr_mask) {
            for (LocalOper *curr = end - 1; curr != start - 1; --curr) {
                *(curr + shift) = *curr;
                (curr + shift)->occ_less ^= curr_mask;
            }
        } else {
            // In the case of no change we can just use optimized moving.
            memmove(start + shift, start, (end - start) * sizeof(*start));
        }

        // Update the index and occupancies of the current insertion point if
        // it has changed.
        if (next_point.index < curr_index) {
            curr_index = next_point.index;
            curr_occ = this->opers_[curr_index].occ_less;
        }

        // Insert the operator corresponding to the mask point at the proper
        // place (note that the insertion point also shifts because of the
        // insertion of further mask points to the right).
        LocalOper &insert_oper = opers_[next_point.index + shift - 1];
        insert_oper = next_point;
        insert_oper.occ_less ^= curr_occ;
        insert_oper.effect = !(insert_oper.occ_less & 1 << insert_oper.flavour);

        // Update the number of segments, but only once per segment (for its
        // starting point, say).
        if (next_point.effect)
            ++(nsegments_[next_point.flavour]);

        curr_mask = next_point.occ_less;
    }

    // Handle the operators before the first mask point.  These are not
    // shifted, but in the case of segments wrapping over beta will have
    // their occupancies changed.
    LocalOper *const start = &opers_[0];
    LocalOper *const end = &opers_[0] + mask[0].index;

    if (curr_mask) {
        for (LocalOper *curr = start; curr != end; ++curr)
            curr->occ_less ^= curr_mask;
    }
}

template <typename ValueT>
void SegmentTrace<ValueT>::remove(unsigned nmask, const MaskPoint *mask)
{
    if (MODE_DEBUG)
        verify_mask(nmask, mask);
    if (!nmask)
        return;

    // Handle the case of the removal of all operators: here, we need to store
    // the current many-body state, because we don't have any operators that
    // indicate it anymore.  W.l.o.g., we just figure out the occupation at
    // tau = 0 and store it.
    if (nopers() == nmask) {
        empty_state_ = opers_[0].occ_less ^ mask[0].occ_less;
    }

    // Removing is done in the foward direction by shifting elements backwards.
    unsigned curr_index = 0;

    for (unsigned shift = 0; shift != nmask; ++shift) {
        LocalOper *const start = &opers_[0] + curr_index;
        LocalOper *const end = &opers_[0] + mask[shift].index;

        if (mask[shift].occ_less) {
            for (LocalOper *curr = start; curr != end; ++curr) {
                curr->occ_less ^= mask[shift].occ_less;
                *(curr - shift) = *curr;
            }
        } else {
            // In the case of no change we can just use optimized moving.
            // memmove probably handles start == end explicitly ...
            memmove(start - shift, start, (end - start) * sizeof(*start));
        }

        // Update the number of segments, but only once per segment (for its
        // starting point, say).
        if (mask[shift].effect)
            --(nsegments_[mask[shift].flavour]);

        // Skip the operator to be removed and handle the next
        curr_index = mask[shift].index + 1;
    }

    // Handle the part beyond the last removal mask point:  since again the
    // mask wraps around, we use the first mask point to determine the
    // occupancy change and we move the grid backwards by the size of the whole
    // mask.
    LocalOper *const start = &opers_[0] + curr_index;
    LocalOper *const end = &opers_[0] + nopers();

    if (mask[0].occ_less) {
        for (LocalOper *curr = start; curr != end; ++curr) {
            curr->occ_less ^= mask[0].occ_less;
            *(curr - nmask) = *curr;
        }
    } else {
        // In the case of no change we can just use optimized moving.
        // memmove probably handles start == end explicitly ...
        memmove(start - nmask, start, (end - start) * sizeof(*start));
    }

    // Reduce the size of the grid
    opers_.resize(nopers() - nmask);
}


template <typename ValueT>
SegmentMove<ValueT>::SegmentMove(SegmentTrace<ValueT> &target, unsigned max_segs)
    :  target_(&target),
       mask_(2 * max_segs),
       ratio_(0)
{ }


template <typename ValueT>
void SegInsertMove<ValueT>::propose(unsigned flavour, double tau_start,
                                    double rel_length)
{
    SegmentTrace<ValueT> &target = *this->target_;

    const unsigned pos_start = target.find(tau_start);
    const bool seg_type = !(
            (target.nopers()
                 ? target.opers_[pos_start % target.nopers()].occ_less
                 : target.empty_state_
            ) & 1 << flavour);
    const double maxlen = target.spacing(pos_start, tau_start, flavour, seg_type);

    double tau_end = tau_start + rel_length * maxlen;
    const bool wraps = tau_end >= target.beta_;
    if (wraps) {
        tau_end -= target.beta_;
    }
    const unsigned pos_end = target.find(tau_end);

    this->mask_[0].tau = tau_start;
    this->mask_[0].flavour = flavour;
    this->mask_[0].effect = true;
    this->mask_[0].index = pos_start;
    this->mask_[0].occ_less = 0;

    this->mask_[1].tau = tau_end;
    this->mask_[1].flavour = flavour;
    this->mask_[1].effect = false;
    this->mask_[1].index = pos_end;
    this->mask_[1].occ_less = 1 << flavour;

    std::sort(this->mask_.begin(), this->mask_.begin() + 2,
              [](const MaskPoint &a, const MaskPoint &b) { return a.tau < b.tau; });

    this->ratio_ = target.abs_ratio(2, &this->mask_[0]);

    // XXX
    tau_start_ = tau_start;
    tau_end_ = tau_end;
    pos_start_ = pos_start;
    pos_end_ = pos_end;
    wraps_ = wraps;
    maxlen_ = maxlen;
    seg_type_ = seg_type;
}

template <typename ValueT>
void SegInsertMove<ValueT>::accept()
{
    SegmentTrace<ValueT> &target = *this->target_;

    target.insert(2, &this->mask_[0]);

    target.abs_weight_ *= this->ratio_;
    target.timeord_sign_ = target.calc_timeord_sign();
    target.pauli_sign_ = target.calc_pauli_sign();

    if (MODE_DEBUG)
        this->target_->verify();
}

template <typename ValueT>
void SegRemoveMove<ValueT>::propose(unsigned pos_start)
{
    SegmentTrace<ValueT> &target = *this->target_;

    assert(pos_start < target.nopers());

    const unsigned pos_end = target.find_end(pos_start);
    const unsigned flavour = target.opers_[pos_start].flavour;
    const bool seg_type = target.opers_[pos_start].effect;

    this->mask_[0].tau = target.opers_[pos_start].tau;   // I dont care
    this->mask_[0].flavour = flavour;
    this->mask_[0].effect = true;
    this->mask_[0].occ_less = 0;
    this->mask_[0].index = pos_start;

    this->mask_[1].tau = target.opers_[pos_end].tau;  // I dont care
    this->mask_[1].flavour = flavour;
    this->mask_[1].effect = false;
    this->mask_[1].occ_less = 1 << flavour;
    this->mask_[1].index = pos_end;

    std::sort(this->mask_.begin(), this->mask_.begin() + 2,
              [](const MaskPoint &a, const MaskPoint &b) { return a.index < b.index; });

    this->ratio_ = target.abs_ratio(2, &this->mask_[0]);

    // XXX
    tau_start_ = target.opers_[pos_start].tau;
    tau_end_ = target.opers_[pos_end].tau;
    pos_start_ = pos_start;
    pos_end_ = pos_end;
    wraps_ = tau_end_ < tau_start_;
    seg_type_ = seg_type;
    maxlen_ = target.spacing(pos_start_ + 1, tau_start_, flavour, seg_type_);
}

template <typename ValueT>
void SegRemoveMove<ValueT>::accept()
{
    SegmentTrace<ValueT> &target = *this->target_;

    target.remove(2, &this->mask_[0]);

    target.abs_weight_ *= this->ratio_;
    target.timeord_sign_ = target.calc_timeord_sign();
    target.pauli_sign_ = target.calc_pauli_sign();

    if (MODE_DEBUG)
        this->target_->verify();
}

template <typename ValueT>
unsigned OccupationEstimator<ValueT>::get_accum_size(unsigned nflavours,
                                                     unsigned order)
{
    if (order > nflavours || !order)
        throw std::runtime_error("invalid order");

    return ((order == 1 ? 1 : get_accum_size(nflavours, order - 1))
                                        * (nflavours + 1 - order))/order;
}

template <typename ValueT>
unsigned OccupationEstimator<ValueT>::get_result_size(unsigned nflavours,
                                                      unsigned order)
{
    if (order > nflavours || !order)
        throw std::runtime_error("invalid order");

    return ipow(nflavours, order);
}

template <typename ValueT>
OccupationEstimator<ValueT>::OccupationEstimator(
                        const SegmentTrace<ValueT> &source, unsigned order)
    : order_(order),
      result_size_(get_result_size(source.nflavours_, order)),
      mask_(get_accum_size(source.nflavours_, order)),
      source_(&source)
{
    // Initialise mask with the first <order> bits set
    uint64_t mask = (1 << order) - 1;
    for (std::vector<uint64_t>::iterator it = mask_.begin(); it != mask_.end();
            ++it) {
        // Compute next permutation of bits in a lexicographical order (taken
        // from <http://graphics.stanford.edu/~seander/bithacks.html>).  For
        // order 1, we generate the bit sequence 1, 10, 100, etc. and thus the
        // occupations. For order 2, we generate 11, 101, 110, 1001, etc. and
        // thus the upper triangular part of the double occupations.
        *it = mask;
        uint64_t tmp = (mask | (mask - 1)) + 1;
        mask = tmp | ((((tmp & -tmp) / (mask & -mask)) >> 1) - 1);
    }
}

template <typename ValueT>
void OccupationEstimator<ValueT>::estimate(ValueT *accum, ValueT weight)
{
    const SegmentTrace<ValueT> &source = *this->source_;

    // While in principle it would be enough to measure <n(tau = 0)>, I
    // found that it is much more accurate at litte more cost to take
    // the whole diagram into account.
    for (unsigned i = 0; i < mask_.size(); ++i) {
        double length = 0;
        if (source.opers_.empty()) {
            // The case of no operators should be handled separately.
            length = source.empty_state_ & mask_[i];
        } else {
            // Wrap the part after the last operator until beta around and add
            // it to the first operator.
            double prev_tau = source.opers_.rbegin()->tau - source.beta_;
            for (std::vector<LocalOper>::const_iterator curr =
                    source.opers_.begin(); curr!=source.opers_.end(); ++curr) {
                if ((curr->occ_less & mask_[i]) == mask_[i])
                    length += curr->tau - prev_tau;
                prev_tau = curr->tau;
            }
        }
        accum[i] += weight * length;
    }
}

template <typename ValueT>
void OccupationEstimator<ValueT>::postprocess(ValueT *result,
                                      const ValueT *accum, ValueT sum_weights)
{
    const SegmentTrace<ValueT> &source = *this->source_;
    const double scale = 1./source.beta() * 1./sum_weights;

    // Set the default to zero if applicable
    if (order_ != 1)
        std::fill(result, result + result_size_, 0);

    // Now set the element plus all permutations
    std::vector<unsigned> flv_indices(order_);
    for (unsigned accum_index = 0; accum_index != accum_size(); ++accum_index) {
        // First fill an array with the indices corresponding to the current
        // entry in a sorted fashion
        unsigned iindex = 0;
        for (unsigned bit = 0; bit != source.nflavours(); ++bit) {
            if (mask_[accum_index] & (1 << bit))
                flv_indices[iindex++] = bit;
        }
        assert(iindex == order_);

        // Now generate all permutations of those indices and fill the result
        // element.
        do {
            unsigned result_index = 0, stride = 1;
            for (unsigned i = 0; i != order_; ++i) {
                result_index += flv_indices[i] * stride;
                stride *= source.nflavours();
            }
            result[result_index] = scale * accum[accum_index];
        } while(std::next_permutation(flv_indices.begin(), flv_indices.end()));
    }
}


template <typename ValueT>
ChiiwEstimator<ValueT>::ChiiwEstimator(const SegmentTrace<ValueT> &source,
                                       unsigned niw, bool use_nfft)
    : niw_(niw),
      source_(&source),
      plan_(source.nflavours(), 1, 2 * niw, use_nfft),
      occ_(source, 1),
      occ_accumulator_(source.nflavours())
{ }

template <typename ValueT>
void ChiiwEstimator<ValueT>::estimate(AccumulatorT accum, ValueT weight)
{
    const SegmentTrace<ValueT> &source = *this->source_;

    // Reset data structures
    plan_.reset();

    // We in principle want to Fourier transform the alternating, piecewise
    // constant function n(t). This is difficult with NFFT; however, by
    // considering the derivative dn(t)/dtau, we get a set of delta spikes,
    // and the FT becomes:
    //
    //                  (  1/iw FT[ n'(t) ]            iw != 0
    //          n(w) =  <
    //                  (  sum (segment lengths)       iw == 0
    //
    // We can then rewrite the FT for iw != 0 as:
    //
    //        n(w_n) = 1/iw_n sum_j exp(i w_n t_j) n'(t_j)
    //               = 1/iw_n sum_j exp(2pi i/beta n t_j) n'(t_j)
    //               = 1/iw_n sum_j exp(2pi i n (x_j + 1/2)) n'(t_j)
    //               = (-1)^n/iw_n sum_j exp(2pi i n x_j) n'(t_j)
    //               = (-1)^n/iw_n (-1)^n NFFT[iw_n, x_j, n'(t_j)]
    //
    // where x_j the interval [0, beta) mapped to [-1/2, 1/2).
    for (std::vector<LocalOper>::const_iterator curr = source.opers_.begin();
            curr != source.opers_.end(); ++curr) {
        plan_.add(curr->flavour,                    // flavour index i
                  curr->effect ? 1 : -1,            // n'_i(t_j)
                  curr->tau/source.beta_ - .5);     // x_j
    }
    plan_.compute();

    // Note that we have to add the total length for the zero frequency,
    // because the n'/iw does not work there.
    std::fill(occ_accumulator_.begin(), occ_accumulator_.end(), 0);
    occ_.estimate(&occ_accumulator_[0], 1);

    // Note that the NFFT frequency convention differs from FFTW:
    // while FFTW frequencies are defined as [0, 1, ..., 2N-1], the NFFT
    // frequencies are instead defined as [-N,..., N-1].
    for (unsigned flv = 0; flv != source.nflavours_; ++flv) {
        plan_.f_hat()[(2*flv + 1) * niw_] = occ_accumulator_[flv];
    }

    // Now the response function, which is convolution in tau, is now just
    // given by the product  n*_i(w) n_j(w).  We can just multiply the result
    // because the factor (-1)^n cancels when multiplying n*n, and the
    // prefactor 1/(iw_n)**2 = 1/-w_n**2 can be applied at post-processing
    // since it factors out.
    for (unsigned iw = 0; iw != niw_; ++iw) {
        for (unsigned i = 0; i != source.nflavours_; ++i) {
            const std::complex<double> nistar = plan_.f_hat()[(2*i + 1) * niw_ - iw];
            for (unsigned j = 0; j <= i; ++j) {
                accum[(i*source.nflavours_ + j)*niw_ + iw] +=
                         weight * nistar * plan_.f_hat()[(2*j + 1)*niw_ + iw];
            }
        }
    }
}

template <typename ValueT>
void ChiiwEstimator<ValueT>::postprocess(ResultT result, ConstAccumulatorT accum,
                                         ValueT sum_weights)
{
    const SegmentTrace<ValueT> &source = *this->source_;
    const double scale = 1./source.beta() * 1./sum_weights;

    for (unsigned iw = 0; iw != niw_; ++iw) {
        const double iw_val = iw == 0 ? 1 : 2. * M_PI/source.beta() * iw;
        for (unsigned i = 0; i != source.nflavours_; ++i) {
            for (unsigned j = 0; j <= i; ++j) {
                std::complex<double> resval = scale/(iw_val * iw_val)
                                * accum[(i*source.nflavours_ + j)*niw_ + iw];
                result[(i*source.nflavours_ + j)*niw_ + iw] = resval;
                result[(j*source.nflavours_ + i)*niw_ + iw] = resval;
            }
        }
    }
}
