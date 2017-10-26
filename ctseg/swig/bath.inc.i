// Interface file for bath classes
//
// Author: Markus Wallerberger

%{
#define SWIG_FILE_WITH_INIT
#include "../src/bath.hh"
%}

%include "std_vector.i"

%define BATH_VEC_INSTANTIATE(suffix, ValueT...)
namespace std {
    // to get around the fact that Block<> is not default-constructible
    %ignore vector< Block<ValueT> >::vector(size_type);
    %ignore vector< Block<ValueT> >::resize;

    // make vector known so we can use it in transferring blocks
    %template(BlockVector ## suffix) vector< Block<ValueT> >;
}
%enddef

BATH_VEC_INSTANTIATE(D, double)

struct BlockOperator
{
    double tau;
    unsigned block_flavour;
    double slot;
};

template <typename ValueT = double>
struct IHybFunc
{
    virtual void values(
                ValueT *buffer, unsigned ncoper, const BlockOperator *copers,
                unsigned naoper, const BlockOperator *aopers) const = 0;

    virtual IHybFunc *clone() const = 0;

    virtual ~IHybFunc();
};

template <typename ValueT = double>
class PolyHybFunc
{
public:
    %apply (ValueT *IN_ARRAY3, int DIM1, int DIM2, int DIM3) {
        (ValueT *values, int rows, int cols, int npoints)
    }
    %extend {
        static PolyHybFunc<ValueT> linear(ValueT *values, int rows, int cols,
                                          int npoints, double beta, int sign)
        {
            if (rows != cols)
                throw std::invalid_argument("Nfl x Nfl x Nb x ord expected");
            return PolyHybFunc<ValueT>::linear(rows, npoints, values, beta, sign);
        }
    }

    %apply (ValueT *IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4) {
        (ValueT *values, int rows, int cols, int nbins, int order)
    }
    %extend {
        PolyHybFunc(ValueT *values, int rows, int cols, int nbins, int order,
                    double beta, int sign)
        {
            if (rows != cols) {
                PyErr_Format(PyExc_ValueError, "Nfl x Nfl x Nb x ord expected");
                return NULL;
            }

            // The SWIG convention is for the constructor to return a new
            // object rather than just initialising it.
            return new PolyHybFunc<ValueT>(values, rows, nbins, order, beta, sign);
        }
    }
};


template <typename ValueT = double>
class Block
{
public:
    %extend {
        Block(const PolyHybFunc<ValueT> &hybf) {
            return new Block<ValueT>(make_adapter(hybf));
        }
    }

    void verify(bool recursive = false, double tol = 1e-6) const;

    double slot(double tau, unsigned flavour, bool effect) const;

    std::vector<ValueT> calc_hyb_matrix() const;
};

template <typename DetMoveT>
class BlockMove
{
public:
    typedef typename DetMoveT::Value Value;

public:
    BlockMove(Block<Value> &target, unsigned max_rank);
};

template <typename ValueT = double>
class BlockAppendMove
        : public BlockMove< DetAppendMove<ValueT> >
{
public:
    typedef ValueT Value;

public:
    BlockAppendMove(Block<ValueT> &target, unsigned max_rank);

    void propose(unsigned rank, BlockOperator *coper, BlockOperator *aoper);

    void accept();
};

template <typename ValueT = double>
class BlockRemoveMove
        : public BlockMove< DetRemoveMove<ValueT> >
{
public:
    typedef ValueT Value;

public:
    BlockRemoveMove(Block<ValueT> &target, unsigned max_rank);

    %exception propose {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (unsigned *IN_ARRAY1, int DIM1) {
        (unsigned *cblockpos, int ncblockpos),
        (unsigned *ablockpos, int nablockpos)
    }
    %extend {
        void unsafe_propose(unsigned *cblockpos, int ncblockpos,
                            unsigned *ablockpos, int nablockpos)
        {
            if (ncblockpos != nablockpos) {
                PyErr_Format(PyExc_ValueError, "Matrix must remain square");
                return;
            }
            // Verification is performed by the det level
            self->propose(ncblockpos, cblockpos, ablockpos);
        }
    }

    void accept();

    %extend {
        void _get_cblockpos(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->cblockpos());
            *DIM1 = self->rank();
        }
        void _get_ablockpos(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->ablockpos());
            *DIM1 = self->rank();
        }
        void _get_crepl(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->crepl());
            *DIM1 = self->rank();
        }
        void _get_arepl(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->arepl());
            *DIM1 = self->rank();
        }
    }

    %rename(_get_perm_sign) perm_sign;
    int perm_sign() const;

    %pythoncode %{
        __swig_getmethods__["cblockpos"] = _get_cblockpos
        if _newclass: cblockpos = property(_get_cblockpos)

        __swig_getmethods__["ablockpos"] = _get_ablockpos
        if _newclass: ablockpos = property(_get_ablockpos)

        __swig_getmethods__["crepl"] = _get_crepl
        if _newclass: crepl = property(_get_crepl)

        __swig_getmethods__["arepl"] = _get_arepl
        if _newclass: arepl = property(_get_arepl)

        __swig_getmethods__["perm_sign"] = _get_perm_sign
        if _newclass: perm_sign = property(_get_perm_sign)

        # C++ code steals a reference to this ... so we need to make sure
        # Python does not destroy the arrays by holding a reference in the
        # Python part of the class.
        def propose(self, cblockpos, ablockpos):
            self._ref_cblockpos = cblockpos
            self._ref_ablockpos = ablockpos
            self.unsafe_propose(cblockpos, ablockpos)
    %}
};

template <typename ValueT = double>
class BlockSetMove
        : public BlockMove< DetSetMove<ValueT> >
{
public:
    typedef ValueT Value;

public:
    BlockSetMove(Block<ValueT> &target, unsigned max_rank);

    void propose(unsigned order, BlockOperator *coper, BlockOperator *aoper);

    void accept();
};

struct BlockPosition
{
    unsigned block_no, block_pos;
};

struct BathOperator
{
    double tau;
    unsigned flavour;
};

template <typename ValueT = double>
class Bath
{
public:
    typedef ValueT Value;

public:
    Bath(const std::vector<Block<ValueT> > &blocks);

    Block<ValueT> &block(unsigned i);

    unsigned flavour_offset(unsigned i) const;

    unsigned pair_offset(unsigned i) const;

    BathOperator coper(unsigned index) const;

    BathOperator aoper(unsigned index) const;

    const BlockPosition &clookup(unsigned i);

    const BlockPosition &alookup(unsigned i);

    void verify(bool recursive = false, double tol = 1e-6) const;

    unsigned find(double tau, bool effect) const;

    int calc_perm_sign() const;
};

template <typename BlockMoveT>
class BathMove
{
public:
    typedef typename BlockMoveT::Value Value;

public:
    BathMove(Bath<Value> &target, unsigned max_rank);

    const BlockMoveT &block_move(unsigned i) const;
};

template <typename ValueT = double>
class BathInsertMove
        : public BathMove< BlockAppendMove<ValueT> >
{
public:
    typedef ValueT Value;

public:
    BathInsertMove(Bath<ValueT> &target, unsigned max_rank);

    %exception propose {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (double *IN_ARRAY1, int DIM1) {
        (double *ctau, int nctau),
        (double *atau, int natau)
    }
    %apply (unsigned *IN_ARRAY1, int DIM1) {
        (unsigned *cflavour, int ncflavour),
        (unsigned *aflavour, int naflavour)
    }
    %extend {
        void unsafe_propose(double *ctau, int nctau, unsigned *cflavour,
                            int ncflavour, double *atau, int natau,
                            unsigned *aflavour, int naflavour)
        {
            if (nctau != ncflavour || nctau != natau || nctau != naflavour) {
                PyErr_Format(PyExc_ValueError, "All arrays must be same size");
                return;
            }
            try {
                self->verify_args(nctau, ctau, cflavour, atau, aflavour);
            } catch(const VerificationError &e) {
                PyErr_Format(PyExc_ValueError, "%s", e.what());
                return;
            }
            self->propose(nctau, ctau, cflavour, atau, aflavour);
        }
    }

    void accept();

    %extend {
        void _get_cindex(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->cindex());
            *DIM1 = self->rank();
        }
        void _get_aindex(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->aindex());
            *DIM1 = self->rank();
        }
    }

    %pythoncode %{
        __swig_getmethods__["cindex"] = _get_cindex
        if _newclass: cindex = property(_get_cindex)

        __swig_getmethods__["aindex"] = _get_aindex
        if _newclass: aindex = property(_get_aindex)

        # C++ code steals a reference to this ... so we need to make sure
        # Python does not destroy the arrays by holding a reference in the
        # Python part of the class.
        def propose(self, ctau, cflavour, atau, aflavour):
            self._ref_ctau = ctau
            self._ref_cflavour = cflavour
            self._ref_atau = atau
            self._ref_aflavour = aflavour
            self.unsafe_propose(ctau, cflavour, atau, aflavour)
    %}
};

template <typename ValueT = double>
class BathRemoveMove
        : public BathMove< BlockRemoveMove<ValueT> >
{
public:
    typedef ValueT Value;

public:
    BathRemoveMove(Bath<ValueT> &target, unsigned max_rank);

    %exception propose {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (unsigned *IN_ARRAY1, int DIM1) {
        (unsigned *cindex, int ncindex),
        (unsigned *aindex, int naindex)
    }
    %extend {
        void unsafe_propose(unsigned *cindex, int ncindex, unsigned *aindex,
                            int naindex)
        {
            if (ncindex != naindex) {
                PyErr_Format(PyExc_ValueError, "Matrix must remain square");
                return;
            }
            try {
                self->verify_args(ncindex, cindex, aindex);
            } catch(const VerificationError &e) {
                PyErr_Format(PyExc_ValueError, "%s", e.what());
                return;
            }
            self->propose(ncindex, cindex, aindex);
        }
    }

    void accept();

    %extend {
        void _get_cindex(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->cindex());
            *DIM1 = self->rank();
        }
        void _get_aindex(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->aindex());
            *DIM1 = self->rank();
        }
    }

    %pythoncode %{
        __swig_getmethods__["cindex"] = _get_cindex
        if _newclass: cindex = property(_get_cindex)

        __swig_getmethods__["aindex"] = _get_aindex
        if _newclass: aindex = property(_get_aindex)

        # C++ code steals a reference to this ... so we need to make sure
        # Python does not destroy the arrays by holding a reference in the
        # Python part of the class.
        def propose(self, cindex, aindex):
            self._ref_cindex = cindex
            self._ref_aindex = aindex
            self.unsafe_propose(cindex, aindex)
    %}
};

template <typename ValueT = double>
class GtauBlockEstimator
{
public:
    typedef ValueT Value;
    typedef Block<ValueT> SourceT;

public:
    GtauBlockEstimator(const Block<ValueT> &source, unsigned ntau_bins);

    %exception estimate {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (ValueT *INPLACE_ARRAY1, int DIM1) { (ValueT *accum, int naccum) }
    %extend {
        void estimate(ValueT *accum, int naccum, ValueT weight)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            self->estimate(accum, weight);
        }
    }
    %clear (ValueT *accum, int naccum);

    %exception postprocess {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (ValueT **ARGOUTVIEWM_ARRAY1, int *DIM1) { (ValueT **result, int *nresult) }
    %apply (ValueT *IN_ARRAY1, int DIM1) { (ValueT *accum, int naccum) }
    %extend {
        void postprocess(ValueT **result, int *nresult,
                         ValueT *accum, int naccum, ValueT sum_weights)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            *nresult = self->result_size();
            *result = (ValueT *)malloc(*nresult * sizeof(**result));
            self->postprocess(*result, accum, sum_weights);
        }
    }
    %clear (ValueT *result, int nresult);
    %clear (ValueT *accum, int naccum);

    %pythoncode %{
        @property
        def accum_dtype(self): return float

        @property
        def result_shape(self): return (self.source.npairs, self.ntau)
    %}
};


template <typename ValueT = double>
class GiwBlockEstimator
{
public:
    typedef const std::complex<double>* ConstAccumulatorT;
    typedef std::complex<double>* AccumulatorT;
    typedef std::complex<double>* ResultT;

public:
    GiwBlockEstimator(const Block<ValueT> &source, unsigned niwf, bool use_nfft);

    %exception estimate {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> *INPLACE_ARRAY1, int DIM1) { (std::complex<double> *accum, int naccum) }
    %extend {
        void estimate(std::complex<double> *accum, int naccum, ValueT weight)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            self->estimate(accum, weight);
        }
    }
    %clear (std::complex<double> *accum, int naccum);

    %exception postprocess {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> **ARGOUTVIEWM_ARRAY1, int *DIM1) { (std::complex<double> **result, int *nresult) }
    %apply (std::complex<double> *IN_ARRAY1, int DIM1) { (std::complex<double> *accum, int naccum) }
    %extend {
        void postprocess(std::complex<double> **result, int *nresult,
                         std::complex<double> *accum, int naccum, ValueT sum_weights)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            *nresult = self->result_size();
            *result = (std::complex<double> *)malloc(*nresult * sizeof(**result));
            self->postprocess(*result, accum, sum_weights);
        }
    }
    %clear (std::complex<double> *result, int nresult);
    %clear (std::complex<double> *accum, int naccum);

    %pythoncode %{
        @property
        def accum_dtype(self): return complex

        @property
        def result_shape(self): return (self.source.npairs, 2 * self.niwf)
    %}
};


template <typename ValueT = double>
class GtauEstimator
{
public:
    GtauEstimator(const Bath<ValueT> &source, unsigned ntau_bins);

    %exception estimate {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (ValueT *INPLACE_ARRAY1, int DIM1) { (ValueT *accum, int naccum) }
    %extend {
        void estimate(ValueT *accum, int naccum, ValueT weight)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            self->estimate(accum, weight);
        }
    }
    %clear (ValueT *accum, int naccum);

    %exception postprocess {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (ValueT **ARGOUTVIEWM_ARRAY1, int *DIM1) { (ValueT **result, int *nresult) }
    %apply (ValueT *IN_ARRAY1, int DIM1) { (ValueT *accum, int naccum) }
    %extend {
        void postprocess(ValueT **result, int *nresult,
                         ValueT *accum, int naccum, ValueT sum_weights)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            *nresult = self->result_size();
            *result = (ValueT *)malloc(*nresult * sizeof(**result));
            self->postprocess(*result, accum, sum_weights);
        }
    }
    %clear (ValueT *result, int nresult);
    %clear (ValueT *accum, int naccum);

    %pythoncode %{
        @property
        def accum_dtype(self): return float

        @property
        def result_shape(self): return (self.source.npairs, self.ntau)
    %}
};


template <typename ValueT = double>
class GiwEstimator
{
public:
    typedef const std::complex<double>* ConstAccumulatorT;
    typedef std::complex<double>* AccumulatorT;
    typedef std::complex<double>* ResultT;

public:
    GiwEstimator(const Bath<ValueT> &source, unsigned niwf, bool use_nfft);

    %exception estimate {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> *INPLACE_ARRAY1, int DIM1) { (std::complex<double> *accum, int naccum) }
    %extend {
        void estimate(std::complex<double> *accum, int naccum, ValueT weight)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            self->estimate(accum, weight);
        }
    }
    %clear (std::complex<double> *accum, int naccum);

    %exception postprocess {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> **ARGOUTVIEWM_ARRAY1, int *DIM1) { (std::complex<double> **result, int *nresult) }
    %apply (std::complex<double> *IN_ARRAY1, int DIM1) { (std::complex<double> *accum, int naccum) }
    %extend {
        void postprocess(std::complex<double> **result, int *nresult,
                         std::complex<double> *accum, int naccum, ValueT sum_weights)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            *nresult = self->result_size();
            *result = (std::complex<double> *) malloc(*nresult * sizeof(**result));
            self->postprocess(*result, accum, sum_weights);
        }
    }
    %clear (std::complex<double> *result, int nresult);
    %clear (std::complex<double> *accum, int naccum);

    %pythoncode %{
        @property
        def accum_dtype(self): return complex

        @property
        def result_shape(self): return (self.source.npairs,  2 * self.niwf)
    %}
};


template <typename ValueT = double>
class G4iwEstimator
{
public:
    typedef const std::complex<double>* ConstAccumulatorT;
    typedef std::complex<double>* AccumulatorT;
    typedef std::complex<double>* ResultT;

public:
    G4iwEstimator(const Bath<ValueT> &source, unsigned niwf, unsigned niwb,
                  bool use_nfft);

    %exception estimate {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> *INPLACE_ARRAY1, int DIM1) { (std::complex<double> *accum, int naccum) }
    %extend {
        void estimate(std::complex<double> *accum, int naccum, ValueT weight)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            self->estimate(accum, weight);
        }
    }
    %clear (std::complex<double> *accum, int naccum);

    %exception postprocess {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> **ARGOUTVIEWM_ARRAY1, int *DIM1) {
        (std::complex<double> **result, int *nresult)
    }
    %apply (std::complex<double> *IN_ARRAY1, int DIM1) {
        (std::complex<double> *accum, int naccum)
    }
    %extend {
        void postprocess(std::complex<double> **result, int *nresult,
                         std::complex<double> *accum, int naccum, ValueT sum_weights)
        {
            if (naccum != (int)self->accum_size()) {
                PyErr_Format(PyExc_ValueError, "Invalid accumulator size");
                return;
            }
            *nresult = self->result_size();
            *result = (std::complex<double> *) malloc(*nresult * sizeof(**result));
            self->postprocess(*result, accum, sum_weights);
        }
    }
    %clear (std::complex<double> *result, int nresult);
    %clear (std::complex<double> *accum, int naccum);

    %pythoncode %{
        @property
        def accum_dtype(self): return complex

        @property
        def result_shape(self):
            return (2, self.source.npairs, self.source.npairs,
                    2 * self.niwb - 1, 2 * self.niwf, 2 * self.niwf)
    %}
};

%define BATH_INSTANTIATE(suffix, ValueT...)

%attribute(PolyHybFunc<ValueT>, unsigned, nflavours, nflavours)
%attribute(PolyHybFunc<ValueT>, unsigned, nbins, nbins)
%attribute(PolyHybFunc<ValueT>, double, beta, beta)
%attribute(PolyHybFunc<ValueT>, int, sign, sign)
%attribute2(PolyHybFunc<ValueT>, PiecewisePolynomial<ValueT>, poly, poly)
%template(PolyHybFunc ## suffix) PolyHybFunc<ValueT>;

%attribute(IHybFunc<ValueT>, unsigned, nflavours, nflavours)
%attribute(IHybFunc<ValueT>, double, beta, beta)
%attribute(IHybFunc<ValueT>, int, sign, sign)
%template(IHybFunc ## suffix) IHybFunc<ValueT>;

%attribute(Block<ValueT>, unsigned, nflavours, nflavours)
%attribute(Block<ValueT>, unsigned, npairs, npairs)
%attribute(Block<ValueT>, double, beta, beta)
%attribute(Block<ValueT>, ValueT, weight, weight)
%attribute(Block<ValueT>, unsigned, order, order)
%attribute2(Block<ValueT>, IHybFunc<ValueT>, hybrfunc, hybrfunc)
%attribute2(Block<ValueT>, Determinant<ValueT>, det, det)
%template(Block ## suffix) Block<ValueT>;

%attribute2(BlockAppendMove<ValueT>, const DetAppendMove<ValueT>, det_move, det_move)
%attribute2(BlockAppendMove<ValueT>, const Block<ValueT>, target, target)
%attribute(BlockAppendMove<ValueT>, ValueT, ratio, ratio)
%attribute(BlockAppendMove<ValueT>, bool, hard_reject, hard_reject)
%attribute(BlockAppendMove<ValueT>, unsigned, rank, rank)
%template(BlockAppendMoveBase ## suffix) BlockMove< DetAppendMove<ValueT> >;
%template(BlockAppendMove ## suffix) BlockAppendMove<ValueT>;

%attribute2(BlockRemoveMove<ValueT>, const DetRemoveMove<ValueT>, det_move, det_move)
%attribute2(BlockRemoveMove<ValueT>, const Block<ValueT>, target, target)
%attribute(BlockRemoveMove<ValueT>, ValueT, ratio, ratio)
%attribute(BlockRemoveMove<ValueT>, bool, hard_reject, hard_reject)
%attribute(BlockRemoveMove<ValueT>, unsigned, rank, rank)
%template(BlockRemoveMoveBase ## suffix) BlockMove< DetRemoveMove<ValueT> >;
%template(BlockRemoveMove ## suffix) BlockRemoveMove<ValueT>;

%attribute2(BlockSetMove<ValueT>, const DetSetMove<ValueT>, det_move, det_move)
%attribute2(BlockSetMove<ValueT>, const Block<ValueT>, target, target)
%attribute(BlockSetMove<ValueT>, ValueT, ratio, ratio)
%attribute(BlockSetMove<ValueT>, bool, hard_reject, hard_reject)
%attribute(BlockSetMove<ValueT>, unsigned, rank, rank)
%template(BlockSetMoveBase ## suffix) BlockMove< DetSetMove<ValueT> >;
%template(BlockSetMove ## suffix) BlockSetMove<ValueT>;

%attribute(Bath<ValueT>, unsigned, nflavours, nflavours)
%attribute(Bath<ValueT>, unsigned, nblocks, nblocks)
%attribute(Bath<ValueT>, unsigned, npairs, npairs)
%attribute(Bath<ValueT>, double, beta, beta)
%attribute(Bath<ValueT>, ValueT, weight, weight)
%attribute(Bath<ValueT>, unsigned, order, order)
%template(Bath ## suffix) Bath<ValueT>;

%attribute2(BathMove< BlockAppendMove<ValueT> >, const Bath<ValueT>, target, target)
%attribute(BathMove< BlockAppendMove<ValueT> >, ValueT, ratio, ratio)
%attribute(BathMove< BlockAppendMove<ValueT> >, bool, hard_reject, hard_reject)
%template(BathInsertMoveBase ## suffix) BathMove< BlockAppendMove<ValueT> >;

%attribute2(BathMove< BlockRemoveMove<ValueT> >, const Bath<ValueT>, target, target)
%attribute(BathMove< BlockRemoveMove<ValueT> >, ValueT, ratio, ratio)
%attribute(BathMove< BlockRemoveMove<ValueT> >, bool, hard_reject, hard_reject)
%template(BathRemoveMoveBase ## suffix) BathMove< BlockRemoveMove<ValueT> >;

%template(BathInsertMove ## suffix) BathInsertMove<ValueT>;
%template(BathRemoveMove ## suffix) BathRemoveMove<ValueT>;

%attribute(GtauBlockEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(GtauBlockEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(GtauBlockEstimator<ValueT>, unsigned, ntau, ntau);
%attribute2(GtauBlockEstimator<ValueT>, Block<ValueT>, source, source);
%template(GtauBlockEstimator ## suffix) GtauBlockEstimator<ValueT>;

%attribute(GtauEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(GtauEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(GtauEstimator<ValueT>, unsigned, ntau, ntau);
%attribute2(GtauEstimator<ValueT>, Bath<ValueT>, source, source);
%template(GtauEstimator ## suffix) GtauEstimator<ValueT>;

%attribute(GiwBlockEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(GiwBlockEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(GiwBlockEstimator<ValueT>, unsigned, niwf, niwf);
%attribute2(GiwBlockEstimator<ValueT>, Block<ValueT>, source, source);
%template(GiwBlockEstimator ## suffix) GiwBlockEstimator<ValueT>;

%attribute(GiwEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(GiwEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(GiwEstimator<ValueT>, unsigned, niwf, niwf);
%attribute2(GiwEstimator<ValueT>, Bath<ValueT>, source, source);
%template(GiwEstimator ## suffix) GiwEstimator<ValueT>;

%attribute(G4iwEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(G4iwEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(G4iwEstimator<ValueT>, unsigned, niwf, niwf);
%attribute(G4iwEstimator<ValueT>, unsigned, niwb, niwb);
%attribute2(G4iwEstimator<ValueT>, Bath<ValueT>, source, source);
%template(G4iwEstimator ## suffix) G4iwEstimator<ValueT>;

%enddef

BATH_INSTANTIATE(D, double)
