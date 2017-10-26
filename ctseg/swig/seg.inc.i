%{
#include "../src/seg.hh"
%}

struct LocalOper
{
    double tau;
    unsigned flavour;
    bool effect;
    uint64_t occ_less;
};

enum DumpParts {
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
    %apply (int DIM1, ValueT *IN_ARRAY1) {(int edim, const ValueT *energies)}
    %apply (int DIM1, int DIM2, ValueT *IN_ARRAY2) {
                                (int urows, int ucols, const ValueT *umatrix)};
    %extend {
        SegmentTrace(double beta, int edim, const ValueT *energies,
                     int urows, int ucols, const ValueT *umatrix)
        {
            if (urows != ucols) {
                PyErr_Format(PyExc_ValueError, "u_matrix must be square");
                return NULL;
            }
            if (edim != urows) {
                PyErr_Format(PyExc_ValueError, "u_matrix vs. energies");
                return NULL;
            }

            // The SWIG convention is for the constructor to return a new
            // object rather than just initialising it.
            return new SegmentTrace<ValueT>(edim, beta, energies, umatrix);
        }
    }
    %clear (int edim, const ValueT *energies);
    %clear (int urows, int ucols, const ValueT *umatrix);

    void verify(double tol = 1e-10) const;

    void reset();

    void dump(DumpParts what = SEG_ALL, unsigned width = 80) const;

    unsigned find(double tau) const;

    ValueT hartree(unsigned npos) const;

    ValueT calc_absweight() const;

    int calc_timeord_sign() const;

    unsigned nsegments(unsigned flavour) const;

    double spacing(unsigned start_pos, double start_tau, unsigned flavour,
                   bool effect) const;

    LocalOper oper(unsigned index) const;
};

template <typename ValueT = double>
class SegmentMove
{
public:
    SegmentMove(SegmentTrace<ValueT> &target, unsigned max_segs = 1);

    const MaskPoint *mask();

    %rename(_get_ratio) ratio;
    ValueT ratio() const;

    %rename(_get_hard_reject) hard_reject;
    bool hard_reject() const;

    %pythoncode %{
        __swig_getmethods__["ratio"] = _get_ratio
        if _newclass: ratio = property(_get_ratio)

        __swig_getmethods__["hard_reject"] = _get_hard_reject
        if _newclass: hard_reject = property(_get_hard_reject)
    %}
};

template <typename ValueT = double>
class SegInsertMove
        : public SegmentMove< ValueT >
{
public:
    SegInsertMove(SegmentTrace<ValueT> &target);

    void propose(unsigned flavour, double tau_start, double len_share);

    void accept();

    %rename(_get_pos_start) pos_start;
    unsigned pos_start() const;

    %rename(_get_pos_end) pos_end;
    unsigned pos_end() const;

    %rename(_get_tau_start) tau_start;
    double tau_start() const;

    %rename(_get_tau_end) tau_end;
    double tau_end() const;

    %rename(_get_seg_type) seg_type;
    bool seg_type() const;

    %rename(_get_wraps) wraps;
    bool wraps() const;

    %rename(_get_maxlen) maxlen;
    double maxlen() const;

    %pythoncode %{
        __swig_getmethods__["pos_start"] = _get_pos_start
        if _newclass: pos_start = property(_get_pos_start)

        __swig_getmethods__["pos_end"] = _get_pos_end
        if _newclass: pos_end = property(_get_pos_end)

        __swig_getmethods__["tau_start"] = _get_tau_start
        if _newclass: tau_start = property(_get_tau_start)

        __swig_getmethods__["tau_end"] = _get_tau_end
        if _newclass: tau_end = property(_get_tau_end)

        __swig_getmethods__["seg_type"] = _get_seg_type
        if _newclass: seg_type = property(_get_seg_type)

        __swig_getmethods__["wraps"] = _get_wraps
        if _newclass: wraps = property(_get_wraps)

        __swig_getmethods__["maxlen"] = _get_maxlen
        if _newclass: maxlen = property(_get_maxlen)
    %}
};

template <typename ValueT = double>
class SegRemoveMove
        : public SegmentMove< ValueT >
{
public:
    SegRemoveMove(SegmentTrace<ValueT> &target);

    void propose(unsigned pos_start);

    void accept();

    %rename(_get_pos_start) pos_start;
    unsigned pos_start() const;

    %rename(_get_pos_end) pos_end;
    unsigned pos_end() const;

    %rename(_get_tau_start) tau_start;
    double tau_start() const;

    %rename(_get_tau_end) tau_end;
    double tau_end() const;

    %rename(_get_seg_type) seg_type;
    bool seg_type() const;

    %rename(_get_wraps) wraps;
    bool wraps() const;

    %rename(_get_maxlen) maxlen;
    double maxlen() const;

    %pythoncode %{
        __swig_getmethods__["pos_start"] = _get_pos_start
        if _newclass: pos_start = property(_get_pos_start)

        __swig_getmethods__["pos_end"] = _get_pos_end
        if _newclass: pos_end = property(_get_pos_end)

        __swig_getmethods__["tau_start"] = _get_tau_start
        if _newclass: tau_start = property(_get_tau_start)

        __swig_getmethods__["tau_end"] = _get_tau_end
        if _newclass: tau_end = property(_get_tau_end)

        __swig_getmethods__["seg_type"] = _get_seg_type
        if _newclass: seg_type = property(_get_seg_type)

        __swig_getmethods__["wraps"] = _get_wraps
        if _newclass: wraps = property(_get_wraps)

        __swig_getmethods__["maxlen"] = _get_maxlen
        if _newclass: maxlen = property(_get_maxlen)
    %}
};


template <typename ValueT = double>
class OccupationEstimator
{
public:
    static unsigned get_accum_size(unsigned nflavours, unsigned order);

    static unsigned get_result_size(unsigned nflavours, unsigned order);

public:
    OccupationEstimator(const SegmentTrace<ValueT> &source, unsigned order);

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
        def result_shape(self): return (self.source.nflavours,) * self.order
    %}
};


template <typename ValueT = double>
class ChiiwEstimator
{
public:
    ChiiwEstimator(const SegmentTrace<ValueT> &source, unsigned niwf, bool use_nfft);

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
            return (self.source.nflavours, self.source.nflavours, self.niw)
    %}
};


%define SEG_INSTANTIATE(suffix, ValueT...)

%attribute(SegmentTrace<ValueT>, unsigned, nopers, nopers)
%attribute(SegmentTrace<ValueT>, unsigned, nflavours, nflavours)
%attribute(SegmentTrace<ValueT>, double, beta, beta)
%attribute(SegmentTrace<ValueT>, ValueT, weight, weight)
%attribute(SegmentTrace<ValueT>, int, timeord_sign, timeord_sign)
%template(SegmentTrace ## suffix) SegmentTrace<ValueT>;

%template(SegMove ## suffix) SegmentMove<ValueT>;
%template(SegInsertMove ## suffix) SegInsertMove<ValueT>;
%template(SegRemoveMove ## suffix) SegRemoveMove<ValueT>;

%attribute(OccupationEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(OccupationEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(OccupationEstimator<ValueT>, unsigned, order, order);
%attribute2(OccupationEstimator<ValueT>, SegmentTrace<ValueT>, source, source);
%template(OccupationEstimator ## suffix) OccupationEstimator<ValueT>;

%attribute(ChiiwEstimator<ValueT>, unsigned, accum_size, accum_size);
%attribute(ChiiwEstimator<ValueT>, unsigned, result_size, result_size);
%attribute(ChiiwEstimator<ValueT>, unsigned, niw, niw);
%attribute2(ChiiwEstimator<ValueT>, SegmentTrace<ValueT>, source, source);
%template(ChiiwEstimator ## suffix) ChiiwEstimator<ValueT>;

%enddef

SEG_INSTANTIATE(D, double)
