/*
 * chi.hh
 *
 *  Created on: Jul 6, 2017
 *      Author: mwallerb
 */

#ifndef CHI_HH_
#define CHI_HH_

#include <vector>
#include <complex>
#include <algorithm>

#include <fftw3.h>

#include "lattice.hh"

inline std::complex<double> sqnorm(std::complex<double> x) {
    return x * std::conj(x);
}

#include <iostream>

class ChiEstimator
{
public:
    ChiEstimator()
        : target_(NULL),
          plan_(NULL),
          rplan_(NULL)
    { }

    ChiEstimator(const SquareLattice &target)
        : target_(&target),
          spin_in_(target.rows() * target.cols(), 0.),
          spin_out_(target.rows() * target.cols(), 0.)
    {
        init_plans();
    }

    ChiEstimator(const ChiEstimator &other)
        : target_(other.target_),
          spin_in_(other.spin_in_),
          spin_out_(other.spin_out_)
    {
        init_plans();
    }

    ChiEstimator &operator=(const ChiEstimator &other)
    {
        destroy_plans();
        target_ = other.target_;
        spin_in_ = other.spin_in_;
        spin_out_ = other.spin_out_;
        init_plans();
        return *this;
    }

    ~ChiEstimator() { destroy_plans(); }

    const std::vector<double> &estimate()
    {
        // copy to spins w/o the borders
        const double norm = spin_out_.size(); // * spin_out_.size();
        for (int i = 0; i != target_->rows(); ++i)
            for (int j = 0; j != target_->cols(); ++j)
                spin_in_[i * target_->cols() + j] = target_->get(i, j) / norm;

        // Do FT
        fftw_execute(plan_);

        // Self-convolution amounts to squaring the FT
        std::transform(spin_out_.begin(), spin_out_.end(), spin_out_.begin(),
                       sqnorm);

        // Do inverse FT
        fftw_execute(rplan_);

        // Return convoluted stuff
        return spin_in_;
    }

    unsigned accum_size() const { return spin_out_.size(); }

protected:
    void init_plans()
    {
        if (target_ == NULL) {
            plan_ = NULL;
            rplan_ = NULL;
        } else {
            plan_ = fftw_plan_dft_r2c_2d(target_->rows(), target_->cols(),
                                 &spin_in_[0],
                                 reinterpret_cast<fftw_complex*>(&spin_out_[0]),
                                 FFTW_MEASURE);
            rplan_ = fftw_plan_dft_c2r_2d(target_->rows(), target_->cols(),
                                 reinterpret_cast<fftw_complex*>(&spin_out_[0]),
                                 &spin_in_[0],
                                 FFTW_MEASURE);
        }
    }

    void destroy_plans()
    {
        if (target_ == NULL)
            return;

        fftw_destroy_plan(plan_);
        fftw_destroy_plan(rplan_);
    }

    const SquareLattice *target_;
    std::vector<double> spin_in_;
    std::vector< std::complex<double> > spin_out_;
    fftw_plan plan_, rplan_;
};



#endif /* CHI_HH_ */
