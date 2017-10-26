#ifndef _DRIVER_HH
#  error "Do not include this file directly, include driver.hh instead"
#endif

template <typename ValueT, typename Random>
Driver<ValueT, Random>::Driver(const Diagram<ValueT> &current, const Random &rand)
    : current_(current),
      insert_move_(current_, 1),
      remove_move_(current_, 1),
      random_(rand),
      record_rates_(false), record_timings_(false),
      rates_(6), timings_(4)
{ }

template <typename ValueT, typename Random>
void Driver<ValueT, Random>::sweep(unsigned steps)
{
    const double beta = current_.local().beta();
    const unsigned nflavours = current_.local().nflavours();

    for (unsigned i = 0; i < steps; ++i) {
        // number of segments/antisegments
        const unsigned nsegments = current_.local().nopers();

        double rate = 0;
        unsigned imove_type = 2 * random_();
        // MoveType &move_type = moves_[imove_type];

        if (imove_type == 0) {
            // ---- insertion move ----
            const unsigned flavour = nflavours * random_();
            const double tau_begin = beta * random_();
            const double len_share = random_();

            insert_move_.propose(flavour, tau_begin, len_share);

            if (insert_move_.hard_reject()) {
                // hard reject
                if (record_rates_)
                    rates_.sum_add(3 * imove_type + 0, 1);
            } else {
                rate = insert_move_.ratio();
                rate *= (beta * insert_move_.local_move().maxlen() * nflavours)/(nsegments + 2);
                if (random_() > std::abs(rate)) {
                    // soft reject
                    if (record_rates_)
                        rates_.sum_add(3 * imove_type + 1, 1);
                } else {
                    // accept
                    insert_move_.accept();
                    if (record_rates_)
                        rates_.sum_add(3 * imove_type + 2, 1);
                }
            }
        } else {
            // ---- removal move ----
            if (nsegments == 0) {
                if (record_rates_)
                    timings_.sum_add(3 * imove_type + 0, 1);
            } else {
                const unsigned start_pos = nsegments * random_();

                remove_move_.propose(start_pos);

                if (remove_move_.hard_reject()) {
                    // hard reject
                    if (record_rates_)
                        rates_.sum_add(3 * imove_type + 0, 1);
                } else {
                    rate = remove_move_.ratio();
                    rate *= nsegments/(beta * remove_move_.local_move().maxlen() * nflavours);
                    if (random_() > std::abs(rate)) {
                        // soft reject
                        if (record_rates_)
                            rates_.sum_add(3 * imove_type + 1, 1);
                    } else {
                        // accept
                        remove_move_.accept();
                        if (record_rates_)
                            rates_.sum_add(3 * imove_type + 2, 1);
                    }
                }
            }
        }
    }
    if (record_rates_)
        rates_.count_add(steps);
}
