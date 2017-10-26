#ifndef _DIA_HH
#  error "Do not include this file directly, include dia.hh instead"
#endif

template <typename ValueT>
BathPosition Diagram<ValueT>::bath_for_local(unsigned opno) const
{
    assert(opno < local_.nopers());
    return bath_for_local_[opno];
}

template <typename ValueT>
void Diagram<ValueT>::verify(bool recursive, double tol) const
{
    if (recursive) {
        local_.verify();
        bath_.verify();
    }

    // Canaries in the coal mine: if these fail, we are seriously f____d
    if (bath_.nflavours() != local_.nflavours())
        throw VerificationError("inconsistent number of flavours");
    if (std::abs(bath_.beta() - local_.beta()) > tol)
        throw VerificationError("inconsistent beta");

    // Check sizes of lookup arrays
    if (bath_for_local_.size() != local_.nopers())
        throw VerificationError("local to bath mapping has invalid size");

    std::vector<unsigned> cvisit(bath_.order()), avisit(bath_.order());

    for(std::vector<BathPosition>::const_iterator it = bath_for_local_.begin();
            it != bath_for_local_.end(); ++it) {
        const unsigned local_index = it - bath_for_local_.begin();
        const unsigned bath_index = it->index;
        const LocalOper &local_oper = local_.oper(local_index);

        if (bath_index >= bath_.order()) {
            throw VerificationError("lookup %d -> %d exceeds bath order %d",
                                    local_index, bath_index, bath_.order());
        }
        if (it->effect != local_oper.effect) {
            throw VerificationError("lookup %d effect %c does not match",
                                    local_index, it->effect ? 'c' : 'a');
        }

        // track how many times each bath operator was referred to
        (local_oper.effect ? cvisit : avisit)[bath_index]++;

        // Check that the mapping array maps the correct local to the correct
        // bath index by comparing their characteristics
        const BathOperator &bath_oper = local_oper.effect ?
                            bath_.coper(bath_index) : bath_.aoper(bath_index);

        if (local_oper.flavour != bath_oper.flavour) {
            throw VerificationError("lookup %d(%d) -> %d(%d) flavour mismatch",
                                    local_index, local_oper.flavour,
                                    bath_index, bath_oper.flavour);
        }
        if (std::abs(local_oper.tau - bath_oper.tau) > tol) {
            throw VerificationError("lookup %d(%f) -> %d(%f) tau mismatch",
                                    local_index, local_oper.tau,
                                    bath_index, bath_oper.tau);
        }
    }

    // Verify that each bath operator is referred to exactly once
    for(std::vector<unsigned>::const_iterator it = cvisit.begin();
            it != cvisit.end(); ++it) {
        if (*it != 1) {
            throw VerificationError("bath cindex %d occurs %d != 1 times",
                                    it - cvisit.begin(), *it);
        }
    }
    for(std::vector<unsigned>::const_iterator it = avisit.begin();
            it != avisit.end(); ++it) {
        if (*it != 1) {
            throw VerificationError("bath aindex %d occurs %d != 1 times",
                                    it - avisit.begin(), *it);
        }
    }
}

template <typename ValueT>
void Diagram<ValueT>::dump(unsigned what, unsigned width) const
{
    FILE *out = stderr;

    local_.dump(static_cast<DumpParts>(SEG_ALL ^ SEG_INFOLINE), width);
    fprintf(out, "k = %d, wloc = %g, wbath = %g, Tloc = %+d, Tbath = %+d\n",
            bath_.order(), local_.weight(), bath_.weight(), local_.timeord_sign(),
            bath_.calc_perm_sign());

}

template <typename ValueT>
void DiaInsertMove<ValueT>::propose(unsigned flavour,
                                    double tau_start, double len_share)
{
    this->local_move_.propose(flavour, tau_start, len_share);
    if (this->local_move_.hard_reject())
        return;

    cflavour_[0] = flavour;
    aflavour_[0] = flavour;

    if (this->local_move_.seg_type()) {
        // segment
        ctau_[0] = this->local_move_.tau_start();
        atau_[0] = this->local_move_.tau_end();
    } else {
        // anti-segment
        ctau_[0] = this->local_move_.tau_end();
        atau_[0] = this->local_move_.tau_start();
    }

    this->bath_move_.propose(1, &ctau_[0], &cflavour_[0], &atau_[0], &aflavour_[0]);
    // hard reject and ratio are implied ...
}

template <typename ValueT>
void DiaInsertMove<ValueT>::accept()
{
    // accept sub-moves
    this->local_move_.accept();
    this->bath_move_.accept();

    // update the mappings
    std::vector<BathPosition> &mapping = this->target_->bath_for_local_;

    // update the bath indices (the image side of the lookup)
    for(std::vector<BathPosition>::iterator it = mapping.begin();
            it != mapping.end(); ++it) {
        const unsigned insertion_index = it->effect ?
                        this->bath_move_.cindex()[0] : this->bath_move_.aindex()[0];
        if (it->index >= insertion_index)
            ++(it->index);
    }

    // insert the corresponding elements into the array:
    // first we have to translate creation/annihilation operator to start and
    // end of the (anti-)segment
    BathPosition start_index, end_index;
    if (this->local_move_.seg_type()) {
        start_index.effect = true;
        start_index.index = this->bath_move_.cindex()[0];
        end_index.effect = false;
        end_index.index = this->bath_move_.aindex()[0];
    } else {
        start_index.effect = false;
        start_index.index = this->bath_move_.aindex()[0];
        end_index.effect = true;
        end_index.index = this->bath_move_.cindex()[0];
    }

    // Afterwards, we have to determine the order in the trace and make room
    // for the elements and set them
    if (this->local_move_.wraps()) {
        unsigned indices[2] = {this->local_move_.pos_end(),
                               this->local_move_.pos_start()};
        insert_many(mapping, 2, indices);
        mapping[this->local_move_.pos_end()] = end_index;
        mapping[this->local_move_.pos_start() + 1] = start_index;
    } else {
        unsigned indices[2] = {this->local_move_.pos_start(),
                               this->local_move_.pos_end()};
        insert_many(mapping, 2, indices);
        mapping[this->local_move_.pos_start()] = start_index;
        mapping[this->local_move_.pos_end() + 1] = end_index;
    }

    if (MODE_DEBUG)
        this->target_->verify();
}

template <typename ValueT>
void DiaRemoveMove<ValueT>::propose(unsigned pos_start)
{
    Diagram<ValueT> &target = *this->target_;
    this->local_move_.propose(pos_start);
    if (this->local_move_.hard_reject())
        return;

    BathPosition cpos, apos;
    if (this->local_move_.seg_type()) {
        cpos = target.bath_for_local_[this->local_move_.pos_start()];
        apos = target.bath_for_local_[this->local_move_.pos_end()];
    } else {
        cpos = target.bath_for_local_[this->local_move_.pos_end()];
        apos = target.bath_for_local_[this->local_move_.pos_start()];
    }
    assert(cpos.effect);
    assert(!apos.effect);

    cindex_[0] = cpos.index;
    aindex_[0] = apos.index;

    this->bath_move_.propose(1, &cindex_[0], &aindex_[0]);
    // hard reject and ratio are implied ...
}

template <typename ValueT>
void DiaRemoveMove<ValueT>::accept()
{
    std::vector<BathPosition> &mapping = this->target_->bath_for_local_;

    // update the bath indices (the image side of the lookup)
    for(std::vector<BathPosition>::iterator it = mapping.begin();
            it != mapping.end(); ++it) {
        const unsigned insertion_index = it->effect ?
                        this->bath_move_.cindex()[0] : this->bath_move_.aindex()[0];
        if (it->index > insertion_index)
            --(it->index);
    }

    // Afterwards, we have to determine the order in the trace and delete
    // them in order
    if (this->local_move_.wraps()) {
        unsigned indices[2] = {this->local_move_.pos_end(),
                               this->local_move_.pos_start()};
        remove_many(mapping, 2, indices);
    } else {
        unsigned indices[2] = {this->local_move_.pos_start(),
                               this->local_move_.pos_end()};
        remove_many(mapping, 2, indices);
    }

    // accept sub-moves
    this->local_move_.accept();
    this->bath_move_.accept();
    if (MODE_DEBUG)
        this->target_->verify();
}

template <typename ValueT>
void fill_hartree_buffer(std::vector< std::vector<ValueT> > &hartree_buffer,
                         const Diagram<ValueT> &source)
{
    assert(hartree_buffer.size() == source.bath().nblocks());

    // Zero out buffers before use
    for (unsigned i = 0; i != source.bath().nblocks(); ++i) {
        // the +1 is there to make sure that we can take the pointer even for
        // a zero-sized array
        hartree_buffer[i].resize(source.bath().block(i).order() + 1);
        std::fill(hartree_buffer[i].begin(), hartree_buffer[i].end(), 0);
    }

    // Fill buffers with corresponding elements
    for (unsigned i = 0; i != source.local().nopers(); ++i) {
        BathPosition pos = source.bath_for_local(i);

        // we only need it in front of creation operators
        if (!pos.effect)
            continue;

        // map the bath index to a block and an index within that block to
        // align the (local) Hartree diagram properly for the use of the (bath)
        // Gtau estimator
        BlockPosition bpos = source.bath().clookup(pos.index);

        hartree_buffer[bpos.block_no][bpos.block_pos]
                                           += source.local().hartree(i);

        // Make sure we found the right operator
        if (MODE_DEBUG) {
            const LocalOper &local_op = source.local().oper(i);
            const BlockOperator &bath_op =
                    source.bath().block(bpos.block_no).copers()[bpos.block_pos];
            if(std::abs(local_op.tau - bath_op.tau) > 1e-8)
                throw VerificationError("tau values do not match");
        }
    }
}


template <typename ValueT>
GSigmatauEstimator<ValueT>::GSigmatauEstimator(const Diagram<ValueT> &source,
                                               unsigned ntau_bins)
    : source_(&source),
      accum_size_(0),
      result_size_(0),
      hartree_buffer_(source.bath().nblocks())
{
    assert(ntau_bins > 0);
    for (unsigned i = 0; i != source.bath().nblocks(); ++i) {
        this->blocks_.push_back(
                GtauBlockEstimator<ValueT>(source.bath().block(i), ntau_bins));
        this->accum_offset_.push_back(accum_size_);
        this->result_offset_.push_back(result_size_);

        accum_size_ += blocks_.rbegin()->accum_size();
        result_size_ += blocks_.rbegin()->result_size();
    }
}

template <typename ValueT>
void GSigmatauEstimator<ValueT>::estimate(ValueT *accum, ValueT weight)
{
    assert(!blocks_.empty());
    const Diagram<ValueT> &source = *this->source_;

    fill_hartree_buffer(hartree_buffer_, source);
    for (unsigned i = 0; i != source.bath().nblocks(); ++i)
        blocks_[i].estimate(&accum[accum_offset_[i]], weight,
                            &hartree_buffer_[i][0]);
}

template <typename ValueT>
void GSigmatauEstimator<ValueT>::postprocess(ValueT *result, const ValueT *accum,
                                             ValueT sum_weights)
{
    assert(!blocks_.empty());
    const Diagram<ValueT> &source = *this->source_;

    for (unsigned i = 0; i != source.bath().nblocks(); ++i)
        blocks_[i].postprocess(&result[result_offset_[i]],
                               &accum[accum_offset_[i]], sum_weights);
}


template <typename ValueT>
GSigmaiwEstimator<ValueT>::GSigmaiwEstimator(const Diagram<ValueT> &source,
                                             unsigned niwf, bool use_nfft)
    : source_(&source),
      accum_size_(0),
      result_size_(0),
      hartree_buffer_(source.bath().nblocks())
{
    assert(niwf > 0);
    for (unsigned i = 0; i != source.bath().nblocks(); ++i) {
        this->blocks_.push_back(
            GiwBlockEstimator<ValueT>(source.bath().block(i), niwf, use_nfft));
        this->accum_offset_.push_back(accum_size_);
        this->result_offset_.push_back(result_size_);

        accum_size_ += blocks_.rbegin()->accum_size();
        result_size_ += blocks_.rbegin()->result_size();
    }
}

template <typename ValueT>
void GSigmaiwEstimator<ValueT>::estimate(AccumulatorT accum, ValueT weight)
{
    assert(!blocks_.empty());
    const Diagram<ValueT> &source = *this->source_;

    fill_hartree_buffer(hartree_buffer_, source);
    for (unsigned i = 0; i != source.bath().nblocks(); ++i)
        blocks_[i].estimate(&accum[accum_offset_[i]], weight,
                            &hartree_buffer_[i][0]);
}

template <typename ValueT>
void GSigmaiwEstimator<ValueT>::postprocess(ResultT result,
                                        AccumulatorT accum, ValueT sum_weights)
{
    assert(!blocks_.empty());
    const Diagram<ValueT> &source = *this->source_;

    for (unsigned i = 0; i != source.bath().nblocks(); ++i)
        blocks_[i].postprocess(&result[result_offset_[i]],
                               &accum[accum_offset_[i]], sum_weights);
}
