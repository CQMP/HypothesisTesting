/**
 * Module for storing bath blocks/non-interacting diagrams (determinants).
 *
 * Author: Markus Wallerberger
 */
#ifndef _BLOCK_HH
#define _BLOCK_HH

#include "util.hh"
#include "det.hh"
#include "dndft.hh"

#include <vector>

// Forward declarations

template <typename ValueT> class IHybFunc;
template <typename HybFuncT> class HybFuncAdapter;

template <typename ValueT> class LinHybFunc;
template <typename ValueT> class Block;

template <typename DetMoveT> class BlockMove;
template <typename ValueT> class BlockAppendMove;
template <typename ValueT> class BlockRemoveMove;
template <typename ValueT> class BlockSetMove;

template <typename ValueT> class GtauBlockEstimator;
template <typename ValueT> class GiwBlockEstimator;

template <typename ValueT>
void swap(Block<ValueT> &left, Block<ValueT> &right);


// Declarations

/**
 * Annotates a block row/column with the corresponding operator data.
 */
struct BlockOperator
{

    BlockOperator();

    template <typename ValueT>
    BlockOperator(const Block<ValueT> &block, double tau,
                  unsigned block_flavour, bool effect, unsigned block_no,
                  unsigned move_pos);

    double tau;
    unsigned block_flavour;
    double slot;

    // TODO: this is actually needed for the insertion sorting only.
    unsigned block_no;
    unsigned move_pos;
};

template <typename T>
struct HybFuncTraits
{
//    typedef void ValueType;
};

template <typename ValueT>
struct SamplingPoint {
    ValueT base, incr;
};



/**
 * Stores a (anti-)periodic propagator function discretised in tau.
 *
 * Stores a non-interacting propagator `G(i, j, tau)` depending on two flavour
 * indices `i, j = 0,1, ..., nflavours-1`, and an imaginary time argument,
 * `tau in [-beta, beta]`, which is (anti-)periodic in tau:
 *
 *                  G(i, j, tau - beta) = sign * G(i, j, tau)
 *
 * `G` is given on a regular grid `tau[i] = i * beta/(nvalues - 1)`, including
 * both endpoints at `0` and `beta`.  When extracting values, the class
 * performs a linear interpolation between the grid points.
 *
 * See also: Block<ValueT>::slot()
 */
template <typename ValueT = double>
class PolyHybFunc
{
public:
    static PolyHybFunc linear(unsigned nflavours, unsigned nvalues_per_pair,
                              const ValueT *values, double beta, int sign);

public:
    /** Create new instance from the values on the grid points */
    PolyHybFunc(const ValueT *values, unsigned nflavours, unsigned nbins,
                unsigned order, double beta, int sign);

    /** Return multiple values */
    void values(ValueT *buffer, unsigned ncoper, const BlockOperator *coper,
                unsigned naoper, const BlockOperator *aoper) const;

    /** Return number of flavours */
    unsigned nflavours() const { return nflavours_; }

    /** Return number of grid points minus 1 */
    unsigned nbins() const { return nbins_; }

    /** Return inverse temperature associated with the propagator */
    double beta() const { return beta_; }

    /** Return -1 for fermionic, +1 for bosonic propagators */
    int sign() const { return sign_; }

    /** Return polynomial */
    const PiecewisePolynomial<ValueT> &poly() const { return values_; }

protected:
    /** Return value of G at given point, employing linear interpolation */
    ValueT value(BlockOperator coper, BlockOperator aoper) const;

    int sign_;
    double beta_;
    unsigned nflavours_, nbins_;

    PiecewisePolynomial<ValueT> values_;
};

template <typename ValueT>
struct HybFuncTraits< PolyHybFunc<ValueT> >
{
    typedef ValueT ValueType;
};

template <typename ValueT>
HybFuncAdapter< PolyHybFunc<ValueT> > make_adapter(const PolyHybFunc<ValueT> &x)
{
    return HybFuncAdapter< PolyHybFunc<ValueT> >(x);
}



template <typename ValueT = double>
struct IHybFunc
{
    virtual double beta() const = 0;

    virtual unsigned nflavours() const = 0;

    virtual int sign() const = 0;

    virtual void values(
                ValueT *buffer, unsigned ncoper, const BlockOperator *copers,
                unsigned naoper, const BlockOperator *aopers) const = 0;

    virtual IHybFunc *clone() const = 0;

    virtual ~IHybFunc() { }
};

template <typename ValueT>
struct HybFuncTraits< IHybFunc<ValueT> >
{
    typedef ValueT ValueType;
};


template <typename T>
struct HybFuncAdapter
    : public IHybFunc<typename HybFuncTraits<T>::ValueType>,
      public T
{
    typedef typename HybFuncTraits<T>::ValueType ValueT;

    HybFuncAdapter() : T() { }

    HybFuncAdapter(const T &other) : T(other) { }

    double beta() const { return T::beta(); }

    unsigned nflavours() const { return T::nflavours(); }

    int sign() const { return T::sign(); }

    void values(ValueT *buffer, unsigned ncoper, const BlockOperator *copers,
                unsigned naoper, const BlockOperator *aopers) const
    {
        T::values(buffer, ncoper, copers, naoper, aopers);
    }

    HybFuncAdapter *clone() const { return new HybFuncAdapter(*this); }

    ~HybFuncAdapter() { }
};

template <typename T>
struct HybFuncTraits< HybFuncAdapter<T> >
{
    typedef typename HybFuncTraits<T>::ValueType ValueType;
};


/**
 * Determinant of a matrix composed of non-interacting propagators.
 *
 * This class models a n-by-n matrix `M`, generated by some non-interacting
 * propagator `G`, evaluated for a set of `n` creation vertices (ctau,
 * cflavour) and annihilation vertices (atau, aflavour), as follows:
 *
 *          M[i, j] = G(cflavour[i], aflavour[j], ctau[i] - atau[j])
 *
 * The class in fact stores the determinant of M in `weight` and the inverse
 * of `M` in `det().invmat()`. The bath block can be updated by `rank-k`
 * updates that are mapped to the corresponding `rank-k` updates of the
 * underlying `Determinant` instance.
 *
 * For update performance, the vertices need not and will in general *not* be
 * sorted by their tau value, thus the determinant differs from the usual
 * time-ordered non-interacting expectation value by a sign.  This is because
 * the block is embedded in a bath, which needs to maintain a permutation
 * array anyway to translate the block to the tau-ordered structure.
 */
template <typename ValueT = double>
class Block
{
public:
    typedef ValueT Value;

public:
    /** Create empy block without any flavours */
    Block();

    /** Create new block instance for a given propagator */
    Block(const IHybFunc<ValueT> &hybfunc);

    /** Constructs block from another one */
    Block(const Block &other);

#if __cplusplus >= 201103L
    /** Constructs block from rvalue expression */
    Block(Block &&other) : Block() { swap(*this, other); }
#endif

    /** Assignment of block from other block */
    Block &operator=(Block other) { swap(*this, other); return *this; }

    /** Return number of flavours in the block */
    unsigned nflavours() const { return nflavours_; }

    /** Return number of possible flavour pairs in the block */
    unsigned npairs() const { return nflavours() * nflavours(); }

    /** Return inverse temperature associated with the block propagator */
    double beta() const { return beta_; }

    /** Verify internal consistency of the block */
    void verify(bool recursive = false, double tol = 1e-6) const;

    /** Return the weight, i.e., the determinant of the underlying matrix */
    ValueT weight() const { return det_.weight(); }

    /** Return the bath diagram order, i.e., rows/column of matrix */
    unsigned order() const { return det_.order(); }

    /**
     * Return the "slot" associated with a vertex.
     *
     * Slots are basically precomputation of indices in a propagator array.
     * The main idea is to map `(cindex, aindex, ctau, atau)` to a difference
     * `(cslot - aslot) * num_bins`.
     */
    double slot(double tau, unsigned block_flavour, bool effect) const;

    /** Return the maximum value of slot differences */
    double slot_max() const { return 2 * nflavours() * nflavours(); }

    /** Return associated generating propagator function */
    const IHybFunc<ValueT> &hybrfunc() const { return *hybrfunc_; }

    /** Return matrix recomputed from scratch using the stored vertices */
    std::vector<ValueT> calc_hyb_matrix() const;

    /** Return underlying Determinant instance, allowing access to inv(M) */
    Determinant<ValueT> &det() { return det_; }

    /** Return underlying Determinant instance, allowing access to inv(M) */
    const Determinant<ValueT> &det() const { return det_; }

    /** Return the creation operators/row vertices */
    const std::vector<BlockOperator> &copers() const { return copers_; }

    /** Return the annihilation operators/column vertices */
    const std::vector<BlockOperator> &aopers() const { return aopers_; }

    friend void swap<>(Block &left, Block &right);

    ~Block() { delete hybrfunc_; }

protected:
    unsigned nflavours_;
    double beta_;
    IHybFunc<ValueT> *hybrfunc_;
    Determinant<ValueT> det_;
    std::vector<BlockOperator> copers_, aopers_;

    friend class BlockAppendMove<ValueT>;
    friend class BlockRemoveMove<ValueT>;
    friend class BlockSetMove<ValueT>;
    friend class GtauBlockEstimator<ValueT>;
    friend class GiwBlockEstimator<ValueT>;
};

/**
 * Base class for bath block moves.
 *
 * These moves basically a thin wrapper over a corresponding determinant move.
 * As such, they are usually trying to follow the structure and limitations of
 * these moves.
 */
template <typename DetMoveT>
class BlockMove
{
public:
    typedef typename DetMoveT::Value Value;

public:
    /** Create a new block move bound to a target */
    BlockMove(Block<Value> &target, unsigned max_rank);

    /** Return underlying move on the determinant */
    const DetMoveT &det_move() const { return det_move_; }

    /** Return target of the move */
    const Block<Value> &target() const { return *target_; }

    /** Return determinant ratio, up to additional sign change */
    Value ratio() const { return det_move_.ratio(); }

    /** Block moves will never be hard rejected */
    Value hard_reject() const { return det_move_.hard_reject(); }

    /** Return rank of the update (number of creation/ann vertices changed) */
    unsigned rank() const { return det_move_.rank(); }

protected:
    Block<Value> *target_;
    DetMoveT det_move_;
};

/**
 * Generate moves that append k vertices to the determinant.
 */
template <typename ValueT = double>
class BlockAppendMove
        : public BlockMove< DetAppendMove<ValueT> >
{
public:
    typedef ValueT Value;

public:
    BlockAppendMove(Block<ValueT> &target, unsigned max_rank);

    BlockAppendMove(const BlockAppendMove &other);

    ~BlockAppendMove();

    void propose(unsigned rank, BlockOperator *coper, BlockOperator *aoper);

    void accept();

protected:
    void reserve();

    unsigned max_order_, capacity_;
    BlockOperator *coper_, *aoper_;
    ValueT *newrow_buffer_, *newcol_buffer_, *newstar_buffer_;
};

/**
 * Generate move that remove k vertices from the determinant
 *
 * Note that as with the underlying determinants, these actually first replace
 * the to-be-removed vertices with the last ones in the determinant and then
 * just shrink the underlying matrix.
 *
 * The indices need to be unique, but need not be sorted.
 */
template <typename ValueT = double>
class BlockRemoveMove
        : public BlockMove< DetRemoveMove<ValueT> >
{
public:
    typedef ValueT Value;

public:
    BlockRemoveMove(Block<ValueT> &target, unsigned max_rank)
        : BlockMove< DetRemoveMove<ValueT> >(target, max_rank)
    { }

    void propose(unsigned rank, unsigned *cblockpos, unsigned *ablockpos);

    void accept();

    const unsigned *cblockpos() const { return this->det_move_.rowno(); }

    const unsigned *crepl() const { return this->det_move_.rowrepl(); }

    const unsigned *ablockpos() const { return this->det_move_.colno(); }

    const unsigned *arepl() const { return this->det_move_.colrepl(); }

    int perm_sign() const { return this->det_move_.perm_sign(); }
};

/**
 * Generate moves that replaces all block operators.
 */
template <typename ValueT = double>
class BlockSetMove
        : public BlockMove< DetSetMove<ValueT> >
{
public:
    typedef ValueT Value;

public:
    BlockSetMove(Block<ValueT> &target, unsigned max_rank)
        : BlockMove< DetSetMove<ValueT> >(target, max_rank),
          coper_(NULL),
          aoper_(NULL),
          new_hybr_(1)
    { }

    void propose(unsigned order, const BlockOperator *coper,
                 const BlockOperator *aoper);

    void accept();

protected:
    const BlockOperator *coper_, *aoper_;
    std::vector<ValueT> new_hybr_;
};


template <typename ValueT = double>
class BlockRecompute
{
public:
    BlockRecompute(Block<ValueT> &target) : move_(target, 0) { }

    void propose();

    void accept() { move_.accept(); }

    double error() const { return std::abs(move_.ratio() - 1); }

    const BlockSetMove<ValueT> &move() const { return move_; }

protected:
    BlockSetMove<ValueT> move_;
};


/**
 * Estimates the dressed propagator from the determinant.
 *
 * Computes and accumulates the dressed propagator for one block:
 *
 *     G(cflv, aflv, ctau - atau) = sum_ij M(j, i) d(cflv - cflavour[i])
 *                 * d(aflv - aflavour[j]) d(ctau - atau - ctau[i] - atau[j])
 *
 * binned into discrete tau intervals (for speed, a double interval from -beta
 * to +beta is used).  This estimator scales as `O(N**2)` and is thus fastest,
 * but one will usually require a significant oversampling factor.
 */
template <typename ValueT = double>
class GtauBlockEstimator
{
public:
    typedef ValueT Value;
    typedef Block<ValueT> SourceT;

public:
    GtauBlockEstimator(const SourceT &source, unsigned ntau_bins);

    const SourceT &source() const { return *source_; }

    void estimate(ValueT *accum, ValueT weight, const ValueT *hartree = NULL);

    void postprocess(ValueT *result, const ValueT *accum, ValueT sum_weights);

    unsigned accum_size() const { return 2 * result_size(); }

    unsigned result_size() const { return source().npairs() * ntau_; }

    unsigned ntau() const { return ntau_; }

protected:
    unsigned ntau_;
    const SourceT *source_;
};

/**
 * Estimates the dressed propagator from the determinant.
 *
 * Computes and accumulates the dressed propagator for one block:
 *
 *     G(cflv, aflv, i w) = sum_ij M(j, i) d(cflv - cflavour[i])
 *                 * d(aflv - aflavour[j]) exp(i w (ctau[i] - atau[j]))
 *
 * direct in fermionic Matsubara frequencies `iw`, by performing a
 * non-equidistant discrete Fourier transform.  This estimator scales as
 * `O(N**2 log(k))` for the fast Fourier transform and as `O(N**2 k)`
 * otherwise.
 */
template <typename ValueT = double>
class GiwBlockEstimator
{
public:
    typedef const std::complex<double>* ConstAccumulatorT;
    typedef std::complex<double>* AccumulatorT;
    typedef std::complex<double>* ResultT;

public:
    GiwBlockEstimator(const Block<ValueT> &source, unsigned niwf, bool use_nfft);

    const Block<ValueT> &source() const { return *source_; }

    void estimate(AccumulatorT accum, ValueT weight, const ValueT *hartree = NULL);

    void postprocess(ResultT result, AccumulatorT accum, ValueT sum_weights);

    unsigned accum_size() const { return this->source_->npairs() * plan_.ngrid(); }

    unsigned result_size() const { return this->source_->npairs() * 2 * niwf_; }

    unsigned niwf() const { return niwf_; }

protected:
    unsigned niwf_;
    const Block<ValueT> *source_;
    DeferredAdjointNDFT<GaussianWindow> plan_;
    std::vector< std::complex<double> > f_hat_;
};

#ifndef SWIG
    #include "block.tcc"
#endif

#endif /* _BLOCK_HH */
