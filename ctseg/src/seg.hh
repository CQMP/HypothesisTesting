#ifndef _SEG_HH
#define _SEG_HH

#include "base.hh"
#include "nfft.hh"

#include <vector>
#include <complex>

// Handle uint64_t for C++ versions before 2011

#if __cplusplus >= 201103L
#include <cstdint>
#else
#include <climits>
#if ULONG_MAX != 0xFFFFFFFFFFFFFFFF
    #error "Expecting unsigned long to be 64-bit or enable C++11"
#endif
typedef unsigned long uint64_t;
#endif /* __cplusplus >= 201103L */

// Forward declarations

struct LocalOper;
template<typename ValueT> class SegmentTrace;
template<typename ValueT> class SegInsertMove;
template<typename ValueT> class SegRemoveMove;

template<typename ValueT> class OccupationEstimator;
template<typename ValueT> class ChiiwEstimator;

// Actual declarations

struct LocalOper
{
    double tau;
    unsigned flavour;
    bool effect;
    uint64_t occ_less;
};

struct MaskPoint
    : public LocalOper
{
    unsigned index;
};

enum DumpParts
{
    SEG_TRACKS = 1,
    SEG_ORDER = 2,
    SEG_FILLING = 4,
    SEG_INFOLINE = 8,
    SEG_ALL = 15
};

template <typename ValueT = double>
class SegmentTrace
{
public:
    typedef ValueT Value;

    static const bool DEBUG_RATIO = false;
    static const bool DEBUG_DUMP = false;
    static const bool DEBUG_OPERIW = false;

public:
    SegmentTrace(unsigned nflavours, double beta, const ValueT *energies,
                 const ValueT *u_matrix);

    void verify(double tol = 1e-10) const;

    void reset();

    void dump(DumpParts what = SEG_ALL, unsigned width = 80) const;

    unsigned find(double tau) const;

    ValueT hartree(unsigned npos) const;

    ValueT calc_absweight() const;

    int calc_timeord_sign() const;

    int calc_pauli_sign() const;

    unsigned nopers() const { return opers_.size(); }

    double beta() const { return beta_; }

    unsigned nflavours() const { return nflavours_; }

    unsigned npairs() const { return nflavours_ * nflavours_; }

    unsigned nsegments(unsigned flavour) const { return nsegments_[flavour]; }

    double spacing(unsigned start_pos, double start_tau, unsigned flavour,
                   bool effect) const;

    unsigned find_end(unsigned start_pos) const;

    void verify_mask(unsigned nmask, const MaskPoint *mask) const;

    ValueT abs_ratio(unsigned nmask, const MaskPoint *mask) const;

    void insert(unsigned nmask, const MaskPoint *mask);

    void remove(unsigned nmask, const MaskPoint *mask);

    ValueT weight() const { return abs_weight_ * timeord_sign_ * pauli_sign_; }

    LocalOper oper(unsigned index) const { return opers_[index]; }

    int timeord_sign() const { return timeord_sign_; }

    int pauli_sign() const { return pauli_sign_; }

protected:
    void cache_expterms(bool shift);

    void dump_track(unsigned flavour, unsigned width) const;

    double beta_;
    unsigned nflavours_;
    std::vector<unsigned> nsegments_;
    std::vector<ValueT> u_matrix_, energies_, expterms_;
    unsigned alpha0_;
    ValueT e0_;

    int timeord_sign_;
    int pauli_sign_;
    ValueT abs_weight_;

    uint64_t empty_state_;
    std::vector<LocalOper> opers_;

    friend class SegInsertMove<ValueT>;
    friend class SegRemoveMove<ValueT>;
    friend class OccupationEstimator<ValueT>;
    friend class ChiiwEstimator<ValueT>;
};

template <typename ValueT = double>
class SegmentMove
{
public:
    SegmentMove(SegmentTrace<ValueT> &target, unsigned max_segs = 1);

    const MaskPoint *mask() { return &mask_[0]; }

    const SegmentTrace<ValueT> &target() const { return *target_; }

    ValueT ratio() const { return ratio_; }

    bool hard_reject() const { return false; }

protected:
    SegmentTrace<ValueT> *target_;
    std::vector<MaskPoint> mask_;
    ValueT ratio_;
};

template <typename ValueT = double>
class SegInsertMove
        : public SegmentMove<ValueT>
{
public:
    SegInsertMove(SegmentTrace<ValueT> &target)
        : SegmentMove<ValueT>(target)
    { }

    void propose(unsigned flavour, double tau_start, double rel_length);

    void accept();

    unsigned pos_start() const { return pos_start_; }

    unsigned pos_end() const { return pos_end_; }

    double tau_start() const { return tau_start_; }

    double tau_end() const { return tau_end_; }

    bool seg_type() const { return seg_type_; }

    bool wraps() const { return wraps_; }

    double maxlen() const { return maxlen_; }

protected:
    unsigned flavour, pos_start_, pos_end_;
    double tau_start_, tau_end_, maxlen_;
    bool seg_type_, wraps_;
};

template <typename ValueT = double>
class SegRemoveMove
        : public SegmentMove<ValueT>
{
public:
    SegRemoveMove(SegmentTrace<ValueT> &target)
        : SegmentMove<ValueT>(target)
    { }

    void propose(unsigned pos_start);

    void accept();

    unsigned pos_start() const { return pos_start_; }

    unsigned pos_end() const { return pos_end_; }

    double tau_start() const { return tau_start_; }

    double tau_end() const { return tau_end_; }

    bool seg_type() const { return seg_type_; }

    bool wraps() const { return wraps_; }

    double maxlen() const { return maxlen_; }

protected:
    unsigned flavour, pos_start_, pos_end_;
    double tau_start_, tau_end_, maxlen_;
    bool seg_type_, wraps_;
};

template <typename ValueT = double>
class SegRecompute
{
public:
    SegRecompute(SegmentTrace<ValueT> &target) { }

    void propose() { }

    void accept() { }

    double error() const { return 0; }
};


template <typename ValueT = double>
class OccupationEstimator
{
public:
    static unsigned get_accum_size(unsigned nflavours, unsigned order);

    static unsigned get_result_size(unsigned nflavours, unsigned order);

public:
    OccupationEstimator(const SegmentTrace<ValueT> &source, unsigned order);

    const SegmentTrace<ValueT> &source() const { return *source_; }

    void estimate(ValueT *accum, ValueT weight);

    void postprocess(ValueT *result, const ValueT *accum, ValueT sum_weights);

    unsigned accum_size() const { return mask_.size(); }

    unsigned result_size() const { return result_size_; }

    unsigned order() const { return order_; }

    const std::vector<uint64_t> &mask() const { return mask_; }

protected:
    unsigned order_;
    unsigned result_size_;
    std::vector<uint64_t> mask_;
    const SegmentTrace<ValueT> *source_;
};

template <typename ValueT = double>
class ChiiwEstimator
{
public:
    typedef const std::complex<double>* ConstAccumulatorT;
    typedef std::complex<double>* AccumulatorT;
    typedef std::complex<double>* ResultT;

public:
    ChiiwEstimator(const SegmentTrace<ValueT> &source, unsigned niw, bool use_nfft);

    void estimate(ValueT weight);

    const SegmentTrace<ValueT> &source() const { return *source_; }

    void estimate(AccumulatorT accum, ValueT weight);

    void postprocess(ResultT result, ConstAccumulatorT accum, ValueT sum_weights);

    unsigned accum_size() const { return result_size(); }

    unsigned result_size() const { return source().npairs() * niw(); }

    unsigned niw() const { return niw_; }

protected:
    unsigned niw_;
    const SegmentTrace<ValueT> *source_;
    AdjointNDFT plan_;
    OccupationEstimator<ValueT> occ_;
    std::vector<ValueT> occ_accumulator_;
};

#ifndef SWIG
#   include "seg.tcc"
#endif

#endif /* _LOCAL_HH */
