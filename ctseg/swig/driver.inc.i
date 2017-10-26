%{
#include "../src/driver.hh"
%}

%include "std_string.i"

%template(AccumulatorU) Accumulator<unsigned>;

class DefaultRandom {
public:
    static const unsigned ENTROPY;

    DefaultRandom(unsigned seed=0);

    void seed(unsigned seed);

    double operator() ();

    unsigned raw();
};

template <typename ValueT, typename Random=DefaultRandom>
class Driver {
public:
    Driver(const Diagram<ValueT> &current, const Random &rand = Random());

    void sweep(unsigned steps);

    %rename(_get_current) current;
    Diagram<ValueT> &current();

    %rename(_get_random) random;
    Random &random() { return random_; }

    %rename(_get_record_rates) record_rates;
    bool record_rates() const;

    %rename(_set_record_rates) record_rates;
    void record_rates(bool rec);

    %rename(_get_rates_wrap) rates;
    Accumulator<unsigned> &rates();

    %rename(_get_record_timings) record_timings;
    bool record_timings() const;

    %rename(_set_record_timings) record_timings;
    void record_timings(bool rec);

    %rename(_get_timings_wrap) timings;
    Accumulator<unsigned> &timings();

    %pythoncode %{
        def _get_rates(self):
            def fold(val):
                return val.reshape(-1, 3)
            return TransformedAccumulator(self._get_rates_wrap(), fold)

        def _get_timings(self):
            def fold(val):
                return val.reshape(-1, 3)
            return TransformedAccumulator(self._get_timings_wrap(), fold)

        __swig_getmethods__["record_rates"] = _get_record_rates
        __swig_setmethods__["record_rates"] = _set_record_rates
        if _newclass: record_rates = property(_get_record_rates, _set_record_rates)

        __swig_getmethods__["rates"] = _get_rates
        if _newclass: rates = property(_get_rates)

        __swig_getmethods__["record_timings"] = _get_record_timings
        __swig_setmethods__["record_timings"] = _set_record_timings
        if _newclass: record_timings = property(_get_record_timings, _set_record_timings)

        __swig_getmethods__["timings"] = _get_timings
        if _newclass: timings = property(_get_timings)

        __swig_getmethods__["current"] = _get_current
        if _newclass: current = property(_get_current)

        __swig_getmethods__["random"] = _get_random
        if _newclass: random = property(_get_random)
    %}
};

%template(DriverD) Driver<double>;
