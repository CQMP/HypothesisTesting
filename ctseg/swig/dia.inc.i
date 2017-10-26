%{
#include "../src/dia.hh"
%}

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
    Diagram(const SegmentTrace<double> &local, const Bath<double> &bath);

    void verify(bool recursive=false, double tol=1e-6) const;

    void dump(unsigned what = 1, unsigned width = 80) const;

    BathPosition bath_for_local(unsigned opno) const;
};

template <typename LocalMoveT, typename BathMoveT, typename ValueT = double>
class DiaMove
{
public:
    DiaMove(Diagram<ValueT> &target, unsigned max_rank);
};

template <typename ValueT = double>
class DiaInsertMove
        : public DiaMove<SegInsertMove<ValueT>, BathInsertMove<ValueT>, ValueT>
{
public:
    DiaInsertMove(Diagram<ValueT> &target, unsigned max_rank);

    void propose(unsigned flavour, double tau_start, double len_share);

    void accept();
};

template <typename ValueT = double>
class DiaRemoveMove
        : public DiaMove<SegRemoveMove<ValueT>, BathRemoveMove<ValueT>, ValueT>
{
public:
    DiaRemoveMove(Diagram<ValueT> &target, unsigned max_rank);

    void propose(unsigned pos_start);

    void accept();
};

template <typename ValueT = double>
class DiaRecompute
{
public:
    DiaRecompute(Diagram<ValueT> &target);

    void propose();

    void accept();
};

template <typename ValueT = double>
class SignEstimator
{
public:
    SignEstimator();

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
    %clear (ValueT *accum, int naccum);
    %clear (ValueT *result, int nresult);

    %pythoncode %{
        @property
        def accum_dtype(self): return float

        @property
        def result_shape(self): return ()
    %}
};

template <typename ValueT = double>
class OrderEstimator
{
public:
    OrderEstimator(const Diagram<ValueT> &source, unsigned order_limit);

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
    %clear (ValueT *accum, int naccum);
    %clear (ValueT *result, int nresult);

    %pythoncode %{
        @property
        def accum_dtype(self): return float

        @property
        def result_shape(self): return (self.source.nflavours, self.order_limit)
    %}
};

template <typename ValueT = double>
class GSigmatauEstimator
{
public:
    GSigmatauEstimator(const Diagram<ValueT> &source, unsigned ntau_bins);

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
    %clear (ValueT *accum, int naccum);
    %clear (ValueT *result, int nresult);

    %pythoncode %{
        @property
        def accum_dtype(self): return float

        @property
        def result_shape(self): return (self.source.bath.npairs, self.ntau)
    %}
};


template <typename ValueT = double>
class GSigmaiwEstimator
{
public:
    typedef const std::complex<double>* ConstAccumulatorT;
    typedef std::complex<double>* AccumulatorT;
    typedef std::complex<double>* ResultT;

public:
    GSigmaiwEstimator(const Diagram<ValueT> &source, unsigned niwf, bool use_nfft);

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
        def result_shape(self): return (self.source.bath.npairs, 2 * self.niwf)
    %}
};


%define DIA_INSTANTIATE(suffix, ValueT...)

%attribute(Diagram<ValueT>, ValueT, weight, weight)
%attribute(Diagram<ValueT>, ValueT, sign, sign)
%attribute2(Diagram<ValueT>, SegmentTrace<ValueT>, local, local)
%attribute2(Diagram<ValueT>, Bath<double>, bath, bath)
%attribute(Diagram<ValueT>, double, beta, beta)
%attribute(Diagram<ValueT>, unsigned, nflavours, nflavours)
%attribute(Diagram<ValueT>, unsigned, order, order)
%template(Diagram ## suffix) Diagram<ValueT>;

%attribute2(
    %arg(DiaMove<SegInsertMove<ValueT>, BathInsertMove<ValueT>, ValueT>),
    SegInsertMove<ValueT>, local_move, local_move)
%attribute2(
    %arg(DiaMove<SegInsertMove<ValueT>, BathInsertMove<ValueT>, ValueT>),
    BathInsertMove<ValueT>, bath_move, bath_move)
%attribute(
    %arg(DiaMove<SegInsertMove<ValueT>, BathInsertMove<ValueT>, ValueT>),
    ValueT, ratio, ratio)
%attribute(
    %arg(DiaMove<SegInsertMove<ValueT>, BathInsertMove<ValueT>, ValueT>),
    bool, hard_reject, hard_reject)
%template(_DiaInsertMoveBase ## suffix)
    DiaMove<SegInsertMove<ValueT>, BathInsertMove<ValueT>, ValueT>;

%attribute2(
    %arg(DiaMove<SegRemoveMove<ValueT>, BathRemoveMove<ValueT>, ValueT>),
    SegRemoveMove<ValueT>, local_move, local_move)
%attribute2(
    %arg(DiaMove<SegRemoveMove<ValueT>, BathRemoveMove<ValueT>, ValueT>),
    BathRemoveMove<ValueT>, bath_move, bath_move)
%attribute(
    %arg(DiaMove<SegRemoveMove<ValueT>, BathRemoveMove<ValueT>, ValueT>),
    ValueT, ratio, ratio)
%attribute(
    %arg(DiaMove<SegRemoveMove<ValueT>, BathRemoveMove<ValueT>, ValueT>),
    bool, hard_reject, hard_reject)
%template(_DiaRemoveMoveBase ## suffix)
    DiaMove<SegRemoveMove<ValueT>, BathRemoveMove<ValueT>, ValueT>;

%template(DiaInsertMove ## suffix) DiaInsertMove<ValueT>;
%template(DiaRemoveMove ## suffix) DiaRemoveMove<ValueT>;

%attribute(DiaRecompute<ValueT>, double, error, error)
%template(DiaRecompute ## suffix) DiaRecompute<ValueT>;

%attribute(SignEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(SignEstimator<ValueT>, unsigned, result_size, result_size);
%template(SignEstimator ## suffix) SignEstimator<ValueT>;

%attribute(OrderEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(OrderEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(OrderEstimator<ValueT>, unsigned, order_limit, order_limit);
%attribute2(OrderEstimator<ValueT>, Diagram<ValueT>, source, source);
%template(OrderEstimator ## suffix) OrderEstimator<ValueT>;

%attribute(GSigmatauEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(GSigmatauEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(GSigmatauEstimator<ValueT>, unsigned, ntau, ntau);
%attribute2(GSigmatauEstimator<ValueT>, Diagram<ValueT>, source, source);
%template(GSigmatauEstimator ## suffix) GSigmatauEstimator<ValueT>;

%attribute(GSigmaiwEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(GSigmaiwEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(GSigmaiwEstimator<ValueT>, unsigned, niwf, niwf);
%attribute2(GSigmaiwEstimator<ValueT>, Diagram<ValueT>, source, source);
%template(GSigmaiwEstimator ## suffix) GSigmaiwEstimator<ValueT>;

%enddef

DIA_INSTANTIATE(D, double)
