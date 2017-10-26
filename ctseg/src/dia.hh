#ifndef _DIA_HH
#define _DIA_HH

#include "seg.hh"
#include "bath.hh"

#include <vector>

// Forward declarations

template <typename ValueT> class Diagram;
template <typename LocalMoveT, typename BathMoveT, typename ValueT> class DiaMove;
template <typename ValueT> class DiaInsertMove;
template <typename ValueT> class DiaRemoveMove;

template <typename ValueT> class SignEstimator;
template <typename ValueT> class OrderEstimator;

template <typename BlockEstimatorT> class GSigmaEstimator;
template <typename ValueT> class GSigmatauEstimator;
template <typename ValueT> class GSigmaiwEstimator;

// Actual declarations

struct BathPosition
{
    bool effect;
    unsigned index;
};

template <typename ValueT = double>
class Diagram {
public:
    typedef ValueT Value;

public:
    Diagram(const SegmentTrace<ValueT> &local, const Bath<ValueT> &bath)
        : local_(local), bath_(bath) {}

    void verify(bool recursive=false, double tol=1e-6) const;

    void dump(unsigned what = 1, unsigned width = 80) const;

    ValueT weight() const { return local_.weight() * bath_.weight(); }

    ValueT sign() const { return weight() > 0 ? 1 : -1; }

    const SegmentTrace<ValueT> &local() const { return local_; }

    SegmentTrace<ValueT> &local() { return local_; }

    const Bath<ValueT> &bath() const { return bath_; }

    Bath<ValueT> &bath() { return bath_; }

    double beta() const { return local_.beta(); }

    unsigned nflavours() const { return local_.nflavours(); }

    unsigned order() const { return bath_.order(); }

    BathPosition bath_for_local(unsigned opno) const;

protected:
    SegmentTrace<double> local_;
    Bath<double> bath_;

    std::vector<BathPosition> bath_for_local_;

    friend class DiaInsertMove<ValueT>;
    friend class DiaRemoveMove<ValueT>;

    friend class SignEstimator<ValueT>;
    friend class OrderEstimator<ValueT>;

    friend class GSigmaEstimator< GtauBlockEstimator<ValueT> >;
    friend class GSigmatauEstimator<ValueT>;

    friend class GSigmaEstimator< GiwBlockEstimator<ValueT> >;
    friend class GSigmaiwEstimator<ValueT>;
};

template <typename LocalMoveT, typename BathMoveT, typename ValueT>
class DiaMove
{
public:
    DiaMove(Diagram<ValueT> &target, unsigned max_rank)
        : target_(&target),
          local_move_(target.local()),  // FIXME local rank
          bath_move_(target.bath(), max_rank)
    { }

    const LocalMoveT &local_move() const { return local_move_; }

    const BathMoveT &bath_move() const { return bath_move_; }

    ValueT ratio() const { return local_move_.ratio() * bath_move_.ratio(); }

    bool hard_reject() const
    {
        return local_move_.hard_reject() || bath_move_.hard_reject();
    }

protected:
    Diagram<ValueT> *target_;
    LocalMoveT local_move_;
    BathMoveT bath_move_;
};

template <typename ValueT = double>
class DiaInsertMove
        : public DiaMove<SegInsertMove<ValueT>, BathInsertMove<ValueT>, ValueT>
{
public:
    DiaInsertMove(Diagram<ValueT> &target, unsigned max_rank)
        : DiaMove<SegInsertMove<ValueT>, BathInsertMove<ValueT>,
                  ValueT>(target, max_rank),
          ctau_(max_rank, -1),
          atau_(max_rank, -1),
          cflavour_(max_rank, -1),
          aflavour_(max_rank, -1)
    { }

    void propose(unsigned flavour, double tau_start, double len_share);

    void accept();

public:
    std::vector<double> ctau_, atau_;
    std::vector<unsigned> cflavour_, aflavour_;
};

template <typename ValueT = double>
class DiaRemoveMove
        : public DiaMove<SegRemoveMove<ValueT>, BathRemoveMove<ValueT>, ValueT>
{
public:
    DiaRemoveMove(Diagram<ValueT> &target, unsigned max_rank)
        : DiaMove<SegRemoveMove<ValueT>, BathRemoveMove<ValueT>,
                  ValueT>(target, max_rank),
          cindex_(max_rank),
          aindex_(max_rank)
    { }

    void propose(unsigned pos_start);

    void accept();

public:
    std::vector<unsigned> cindex_, aindex_;
};


template <typename ValueT = double>
class DiaRecompute
{
public:
    DiaRecompute(Diagram<ValueT> &target)
        : local_recomp_(target.local()),
          bath_recomp_(target.bath())
    { }

    void propose()
    {
        local_recomp_.propose();
        bath_recomp_.propose();
    }

    void accept()
    {
        local_recomp_.accept();
        bath_recomp_.accept();
    }

    double error() const
    {
        return local_recomp_.error() + bath_recomp_.error();
    }

protected:
    SegRecompute<ValueT> local_recomp_;
    BathRecompute<ValueT> bath_recomp_;
};


template <typename ValueT = double>
class SignEstimator
{
public:
    SignEstimator() { }

    void estimate(ValueT *accum, ValueT weight)
    {
        // This is a little bit of a special case as this is the only estimator
        // that in a sense does not include sgn(w) in the normalisation.
        accum[0] += weight;
        accum[1] += std::abs(weight);
    }

    void postprocess(ValueT *result, const ValueT *accum, ValueT sum_weights)
    {
        assert(std::abs(accum[0]/sum_weights - 1) < 1e-10);
        result[0] = accum[0]/accum[1];
    }

    unsigned accum_size() const { return 2; }

    unsigned result_size() const { return 1; }
};

template <typename ValueT = double>
class OrderEstimator
{
public:
    OrderEstimator(const Diagram<ValueT> &source, unsigned order_limit)
        : source_(&source),
          order_limit_(order_limit)
    { }

    const Diagram<ValueT> &source() const { return *source_; }

    void estimate(ValueT *accum, ValueT weight)
    {
        for (unsigned i = 0; i != source().nflavours(); ++i) {
            unsigned nseg = source().local().nsegments(i);
            if (nseg < order_limit_)
                accum[i * order_limit_ + nseg] += weight;
        }
    }

    void postprocess(ValueT *result, const ValueT *accum, ValueT sum_weights)
    {
        for (unsigned i = 0; i != accum_size(); ++i)
            result[i] = accum[i] / sum_weights;
    }

    unsigned accum_size() const { return result_size(); }

    unsigned result_size() const { return source().nflavours() * order_limit_; }

    unsigned order_limit() const { return order_limit_; }

protected:
    const Diagram<ValueT> *source_;
    unsigned order_limit_;
};

template <typename ValueT>
class GSigmatauEstimator
{
public:
    GSigmatauEstimator(const Diagram<ValueT> &source, unsigned ntau_bins);

    const Diagram<ValueT> &source() const { return *source_; }

    void estimate(ValueT *accum, ValueT weight);

    void postprocess(ValueT *result, const ValueT *accum, ValueT sum_weights);

    unsigned accum_size() const { return accum_size_; }

    unsigned result_size() const { return result_size_; }

    unsigned ntau() const { return blocks_[0].ntau(); }

protected:
    const Diagram<ValueT> *source_;
    std::vector< GtauBlockEstimator<ValueT> > blocks_;
    std::vector<unsigned> accum_offset_, result_offset_;
    unsigned accum_size_, result_size_;
    std::vector< std::vector<ValueT> > hartree_buffer_;
};

template <typename ValueT>
class GSigmaiwEstimator
{
public:
    typedef const std::complex<double>* ConstAccumulatorT;
    typedef std::complex<double>* AccumulatorT;
    typedef std::complex<double>* ResultT;

public:
    GSigmaiwEstimator(const Diagram<ValueT> &source, unsigned niwf,
                      bool use_nfft);

    const Diagram<ValueT> &source() const { return *source_; }

    void estimate(AccumulatorT accum, ValueT weight);

    void postprocess(ResultT result, AccumulatorT accum, ValueT sum_weights);

    unsigned accum_size() const { return accum_size_; }

    unsigned result_size() const { return result_size_; }

    unsigned niwf() const { return blocks_[0].niwf(); }

protected:
    const Diagram<ValueT> *source_;
    std::vector< GiwBlockEstimator<ValueT> > blocks_;
    std::vector<unsigned> accum_offset_, result_offset_;
    unsigned accum_size_, result_size_;
    std::vector< std::vector<ValueT> > hartree_buffer_;
};

#ifndef SWIG
    #include "dia.tcc"
#endif

#endif /* _DIA_HH */
