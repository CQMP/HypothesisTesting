
#include "updates.hh"
#include "lattice.hh"
#include "chi.hh"

#include <cassert>
#include <vector>

#include <ostream>
#include <fstream>
#include <cmath>

#include <iostream>
#include <ctime>

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#include <alps/params.hpp>
#include <boost/random.hpp>

namespace brandom = boost::random;
namespace aaccum = alps::accumulators;

class IsingSimulation
        : public alps::mcbase
{
public:
    IsingSimulation(parameters_type const &params, std::size_t seed_offset)
        : alps::mcbase(params, seed_offset)
    {
        lattice_ = SquareLattice(params["rows"], params["columns"],
                                 params["boundary"], params["buggy"]);
        chi_est_ = ChiEstimator(lattice_);

        temp_ = params["temperature"];
        j_ = params["j"];
        updates_ = params["updates"];
        flip_ = SpinFlipSweeper(lattice_, random.engine(), temp_, j_);
        wolff_ = WolffUpdate(lattice_, random.engine(), temp_, j_);

        unsigned batches = params["batches"];
        sweeps_total_ = params["sweeps"];
        sweeps_done_ = -double(params["thermalization"]) * sweeps_total_;
        get_chi_ = params["getchi"];

        measurements << aaccum::FullBinningAccumulator<double>("m", batches)
                     << aaccum::FullBinningAccumulator<double>("m2", batches)
                     << aaccum::FullBinningAccumulator<double>("m4", batches);

        if (get_chi_) {
            measurements << aaccum::FullBinningAccumulator<std::vector<double> >
                                ("chi", batches);
        }

        // Tallys
        verbose_ = params["verbose"];
        elapsed_ = 0;
    }

    virtual void update()
    {
        // decide which sort of update to do
        switch(updates_) {
        case 0:
            flip_.sweep();
            break;
        case 1:
            wolff_.sweep();
            break;
        default:
            throw std::runtime_error("Unknown update scheme");
        }

        // elapsed
        if (verbose_ && clock() - elapsed_ > CLOCKS_PER_SEC) {
            if (verbose_ & 2)
                std::cerr << '\n' << lattice_;
            if (verbose_ & 1)
                std::cerr << "Sweep " << sweeps_done_ << " of " << sweeps_total_;
            std::cerr << std::endl;
            elapsed_ = clock();
        }
    }

    // This collects the measurements at each MC step.
    virtual void measure()
    {
        // Increase the count
        ++sweeps_done_;

        // Do not measure in thermalisation phase
        if (sweeps_done_ <= 0)
            return;

        double mcurr = 1. * lattice_.sum() / (lattice_.rows() * lattice_.cols());

        measurements["m"] << mcurr;
        measurements["m2"] << (mcurr *= mcurr);
        measurements["m4"] << (mcurr *= mcurr);
        if (get_chi_)
            measurements["chi"] << chi_est_.estimate();
    };

    virtual double fraction_completed() const
    {
        return sweeps_done_ < 0 ? 0 : 1. * sweeps_done_/sweeps_total_;
    }

    void evaluate_results(alps::accumulators::result_set &res,
                          std::string result_file) const
    {
        res["U4"] = 1.0 - res["m4"] / (3 * res["m2"] * res["m2"]);

        std::cerr << res << std::endl;
//        std::cerr << "Racc =" << 1.0 * flip_.accepts()/flip_.proposals() << std::endl;
        alps::hdf5::archive(result_file, "w")["/"] << res;
    }

// The internal state of our simulation
private:
    SquareLattice lattice_;
    ChiEstimator chi_est_;

    int updates_;
    SpinFlipSweeper flip_;
    WolffUpdate wolff_;

    double temp_, j_;
    long sweeps_total_, sweeps_done_;
    bool get_chi_;

    clock_t elapsed_;
    int verbose_;
};

int main(int argc, char *argv[])
{
    alps::params p(argc, (const char **)argv);
    alps::mcbase::define_parameters(p);
    p.define<size_t>("walltime", 0, "Walltime limit for the run (0=none)");
    p.define<int>("rows", "number of rows in lattice");
    p.define<int>("columns", "number of columns in lattice");
    p.define<int>("boundary", 1, "1=periodic, -1=antiperiodic, 0=hard");
    p.define<bool>("buggy", false, "introduce nasty bug");
    p.define<int>("updates", 0, "0=typewriter, 1=Wolff");
    p.define<double>("temperature", "temperature (usually in units of -J)");
    p.define<double>("j", 1, "nearest-neighbour coupling");
    p.define<int>("batches", 128, "number of batches for binning");
    p.define<long>("sweeps", "number of Monte-Carlo sweeps");
    p.define<double>("thermalization", 0.1, "share of discarded sweeps");
    p.define<bool>("getchi", true, "measure correlation function");
    p.define<int>("verbose", 3, "print status report");
    p.define<std::string>("result", "result.h5", "HDF5 file for storing results");
    if (p.help_requested(std::cerr))
        return 1;

    IsingSimulation sim(p, 0);
    sim.run(alps::stop_callback((size_t)p["walltime"]));
    alps::results_type<IsingSimulation>::type res = alps::collect_results(sim);
    sim.evaluate_results(res, p["result"]);
    return 0;
}
