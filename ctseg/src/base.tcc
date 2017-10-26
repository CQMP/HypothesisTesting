VerificationError::VerificationError(const char *format, ...) {
    my_what_str_ = new char[200];
    va_list argp;
    va_start(argp, format);
    vsnprintf(my_what_str_, 200, format, argp);
    va_end(argp);
    fprintf(stderr, "\n=== Verification failure: %s\n", my_what_str_);
}

template <typename T>
void insert_many(std::vector<T> &vec, unsigned npos, const unsigned *pos)
{
    unsigned last_pos = vec.size();
    vec.resize(vec.size() + npos);
    for (unsigned shift = npos; shift; --shift) {
        unsigned next_pos = pos[shift - 1];
        assert(next_pos <= last_pos);

        T *oper = &vec[0] + next_pos;
        const unsigned move_count = last_pos - next_pos;
        memmove(oper + shift, oper, move_count * sizeof(*oper));
        if (move_count)
            oper[shift - 1] = oper[shift];
        last_pos = next_pos;
    }
}

template <typename T>
void remove_many(std::vector<T> &vec, unsigned npos, const unsigned *pos)
{
    const unsigned size = vec.size();
    unsigned this_pos = pos[0];
    for (unsigned shift = 1; shift < npos; ++shift) {
        unsigned next_pos = pos[shift];
        assert(next_pos > this_pos);

        ++this_pos;         // set it *after* the element to delete
        T *oper = &vec[0] + this_pos;
        memmove(oper - shift, oper, (next_pos - this_pos) * sizeof(*oper));
        this_pos = next_pos;
    }
    if (npos) {
        ++this_pos;
        T *oper = &vec[0] + this_pos;
        memmove(oper - npos, oper, (size - this_pos)*sizeof(*oper));
    }
    vec.resize(size - npos);
}

template <typename ValueT>
void Accumulator<ValueT>::mean(ValueT *result, unsigned result_size)
{
    assert(result_size == size());
    for (unsigned i = 0; i != size(); ++i) {
        result[i] = sum_[i] / count_;
    }
}

template <typename ValueT>
void Accumulator<ValueT>::reset()
{
    std::fill(sum_.begin(), sum_.end(), 0);
    count_ = 0;
}

template <typename ValueT>
void HistogramAccumulator<ValueT>::sum_add(unsigned index, unsigned value,
                                           ValueT weight)
{
    if (value >= histo_.size()/size_)
        histo_.resize((value + 1) * size_);
    histo_[value * size_ + index] += weight;
}

template <typename ValueT>
void HistogramAccumulator<ValueT>::mean(ValueT *result, unsigned size)
{
    assert(size == size_);
    for (unsigned i = 0; i != size_; ++i) {
        ValueT mean_val = 0;
        for (unsigned v = 0; v != histo_.size()/size_; ++v)
            mean_val += ValueT(v) * histo_[v * size_ + i];
        result[i] = mean_val / count_;
    }
}

template <typename ValueT>
void HistogramAccumulator<ValueT>::reset()
{
    std::fill(histo_.begin(), histo_.end(), 0);
    count_ = 0;
}
