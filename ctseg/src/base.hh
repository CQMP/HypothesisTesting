#ifndef _BASE_HH
#define _BASE_HH

#include <stdexcept>

#include <cstdarg>
#include <cstdio>
#include <cassert>

#include <vector>

#if __cplusplus >= 201103L
    #define DATA(vec) ((vec).data())
#else
    #define DATA(vec) (assert((vec).size()) && &(vec)[0])
#endif /* __cplusplus >= 201103L */

struct LossOfPrecision : public std::exception { };

class VerificationError
{
public:
    VerificationError(const char *format, ...);

    ~VerificationError() { delete [] my_what_str_; }

    const char *what() const { return my_what_str_; }

private:
    char *my_what_str_;
};

template <typename SourceT, typename ValueT>
class Derived
{
    Derived(SourceT &source, unsigned size);

    const SourceT &source() { return source_; }

    unsigned size() const { return data_.size(); }

    const ValueT *data() const { return &data_[0]; }

    ValueT *data() { return &data_[0]; }

protected:
    std::vector<ValueT> data_;
    const SourceT &source_;
};

//void write_data(int file_descr, unsigned num_rows, unsigned num_cols, double *data);
//void read_data(int file_descr, unsigned num_)

template <typename T>
void insert_many(std::vector<T> &vec, unsigned npos, const unsigned *pos);

template <typename T>
void remove_many(std::vector<T> &vec, unsigned npos, const unsigned *pos);

template <typename ValueT = double>
class Accumulator
{
public:
    Accumulator(unsigned size)
        : count_(0), sum_(size, 0)
    { }

    void sum_add(unsigned index, ValueT value, ValueT weight = 1.)
    { sum_[index] += value * weight; }

    void count_add(ValueT weight) { count_ += weight; }

    void mean(ValueT *result, unsigned size);

    unsigned size() const { return sum_.size(); }

    ValueT count() const { return count_; }

    const ValueT *sum() const { return sum_.size() ? &sum_[0] : NULL; }

    void reset();


protected:
    ValueT count_;
    std::vector<ValueT> sum_;
};

template <typename ValueT = double>
class HistogramAccumulator
{
public:
    HistogramAccumulator(unsigned size)
        : size_(size), count_(0), histo_(0, 0)
    { }

    void sum_add(unsigned index, unsigned value, ValueT weight = 1.);

    void count_add(ValueT weight) { count_ += weight; }

    void mean(ValueT *result, unsigned size);

    unsigned size() const { return size_; }

    unsigned value_capacity() const { return histo_.size()/size_; }

    ValueT count() const { return count_; }

    const ValueT *histo() const { return histo_.size() ? &histo_[0] : NULL; }

    void reset();

protected:
    unsigned size_;
    ValueT count_;
    std::vector<ValueT> histo_;
};

template <typename TargetT>
class MoveGenerator
{
public:
    typedef TargetT Target;

public:
    MoveGenerator() : target_(NULL) { }

    MoveGenerator(TargetT &target) : target_(&target) { }

    const TargetT &target() const { return *target_; }

    TargetT &target() { return *target_; }

protected:
    TargetT *target_;
};

#ifndef SWIG
    #include "base.tcc"
#endif

#endif /* _BASE_HH */
