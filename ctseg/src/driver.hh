#ifndef _DRIVER_HH
#define _DRIVER_HH

#include "dia.hh"
#include "random.hh"

#include <string>

template <typename ValueT, typename Random=DefaultRandom>
class Driver {
public:
    Driver(const Diagram<ValueT> &current, const Random &rand = Random());

    void sweep(unsigned steps);

    Diagram<ValueT> &current() { return current_; }

    const Diagram<ValueT> &current() const { return current_; }

    bool record_rates() const { return record_rates_; }

    void record_rates(bool rec) { record_rates_ = rec; }

    const Accumulator<unsigned> &rates() const { return rates_; };

    Accumulator<unsigned> &rates() { return rates_; };

    bool record_timings() const { return record_timings_; }

    void record_timings(bool rec) { record_timings_ = rec; }

    const Accumulator<unsigned> &timings() const { return timings_; };

    Accumulator<unsigned> &timings() { return timings_; };

    Random &random() { return random_; }

    const Random &random() const { return random_; }

protected:
    Diagram<ValueT> current_;
    DiaInsertMove<ValueT> insert_move_;
    DiaRemoveMove<ValueT> remove_move_;

    Random random_;
    bool record_rates_, record_timings_;
    Accumulator<unsigned> rates_, timings_;
};

#ifndef SWIG
#  include "driver.tcc"
#endif

#endif /* _DRIVER_HH */
