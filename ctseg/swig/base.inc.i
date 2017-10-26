%{
#include "../src/base.hh"
%}

%pythoncode %{
    import numpy as _np
%}

%exception verify
{
    try {
        $action
    } catch (VerificationError &e) {
        PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
        return NULL;
    }
}

class VerificationError
{
public:
    VerificationError(const char *format, ...);

    ~VerificationError();

    const char *what() const;
};

%pythoncode %{
    class TransformedAccumulator:
        def __init__(self, inner, func):
            self.inner = inner
            self.func = func

        def sum_add(self, index, value, weight=1.):
            self.inner.sum_add(index, value, weight)

        def count_add(self, weight):
            self.inner.count_add(weight)

        def reset(self):
            self.inner.reset()

        @property
        def count(self):
            return self.self.inner._get_count()

        @property
        def sum(self):
            return self.func(self.inner._get_sum())

        @property
        def mean(self):
            return self.func(self.inner._get_mean())

    class BlockedAccumulator:
        def __init__(self, est):
            self.inner = [est.block(i).accumulator for i in range(est.nblocks)]

        @property
        def mean(self):
            return _np.concatenate([acc.mean for acc in self.inner], 0)

        def reset(self):
            for acc in self.inner:
                acc.reset()
%}

template <typename ValueT = double>
class Accumulator
{
public:
    Accumulator(unsigned size);

    void sum_add(unsigned index, ValueT value, ValueT weight = 1.);

    void count_add(ValueT weight);

    %rename (_get_mean_wrap) mean;
    %apply (ValueT *ARGOUT_ARRAY1, int DIM1) {(ValueT *result, unsigned size)};
    void mean(ValueT *result, unsigned size);
    %clear (ValueT *result, unsigned size);

    %rename(_get_size) size;
    unsigned size() const;

    %rename(_get_count) count;
    ValueT count() const;

    %extend {
        void _get_sum(ValueT **ARGOUTVIEW_ARRAY1, int *DIM1) {
            // Mapping function to the numpy.i interface
            *ARGOUTVIEW_ARRAY1 = const_cast<ValueT *>(self->sum());
            *DIM1 = self->size();
        }
    }

    void reset();

    %pythoncode %{
        def _get_mean(self):
            return self._get_mean_wrap(self._get_size())

        __swig_getmethods__["sum"] = _get_sum
        if _newclass: sum = property(_get_sum)

        __swig_getmethods__["mean"] = _get_mean
        if _newclass: mean = property(_get_mean)

        __swig_getmethods__["size"] = _get_size
        if _newclass: size = property(_get_size)

        __swig_getmethods__["count"] = _get_count
        if _newclass: count = property(_get_count)

        data = sum
    %}
};

template <typename ValueT = double>
class HistogramAccumulator
{
public:
    HistogramAccumulator(unsigned size);

    void sum_add(unsigned index, unsigned value, ValueT weight = 1.);

    void count_add(ValueT weight);

    %rename (_get_mean_wrap) mean;
    %apply (ValueT *ARGOUT_ARRAY1, int DIM1) {(ValueT *result, unsigned size)};
    void mean(ValueT *result, unsigned size);
    %clear (ValueT *result, unsigned size);

    %extend {
        void _get_histo_flat(ValueT **ARGOUTVIEW_ARRAY2, int *DIM1,
                             int *DIM2) {
            // Mapping function to the numpy.i interface
            *ARGOUTVIEW_ARRAY2 = const_cast<ValueT *>(self->histo());
            *DIM1 = self->value_capacity();
            *DIM2 = self->size();
        }
    }

    %rename(_get_size) size;
    unsigned size() const;

    %rename(_get_count) count;
    ValueT count() const;

    void reset();

    %pythoncode %{
        def _get_mean(self):
            return self._get_mean_wrap(self._get_size())

        __swig_getmethods__["histo"] = _get_histo_flat
        if _newclass: histo = property(_get_histo_flat)

        __swig_getmethods__["mean"] = _get_mean
        if _newclass: mean = property(_get_mean)

        __swig_getmethods__["size"] = _get_size
        if _newclass: size = property(_get_size)

        __swig_getmethods__["count"] = _get_count
        if _newclass: count = property(_get_count)

        data = mean
    %}
};


%define BASE_INSTANTIATE(suffix, ValueT...)

%template(Accumulator ## suffix) Accumulator<ValueT>;
%template(HistogramAccumulator ## suffix) HistogramAccumulator<ValueT>;

%enddef

BASE_INSTANTIATE(D, double)
BASE_INSTANTIATE(Z, std::complex<double>)
