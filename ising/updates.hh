/*
 * updates.hh
 *
 *  Created on: Jul 7, 2017
 *      Author: markus
 */
#ifndef UPDATES_HH_
#define UPDATES_HH_

#include "lattice.hh"

#include <cassert>
#include <vector>
#include <stack>
#include <cmath>

#include <iostream>
#include <boost/random.hpp>

namespace brandom = boost::random;

class SpinFlipSweeper
{
public:
    SpinFlipSweeper()
        : target_(NULL),
          rng_(NULL)
    { }

    SpinFlipSweeper(SquareLattice &target, brandom::mt19937 &rng,
                     double temp, double j)
        : target_(&target),
          rng_(&rng),
          die01_(0, 1),
          accepts_(0),
          proposals_(0)
    {
        for (int sumn = -4; sumn <= 4; sumn += 2) {
            double delta_e = -4 * j * sumn/2;
            ratios_[sumn/2 + 2] = std::exp(delta_e/temp);
        }
    }

    void sweep()
    {
        SquareLattice &lattice = *target_;
        for (int i = 0; i != lattice.rows(); ++i) {
            for (int j = 0; j != lattice.cols(); ++j) {
                char spin = lattice.get(i, j);
                int sum_nn = spin * lattice.sum_nn(i, j);
                double ratio = ratios_[sum_nn/2 + 2];

                if(ratio >= 1 || die01_(*rng_) < ratio) {
                    lattice.set(i, j, -spin);
                    ++accepts_;
                }
            }
        }
        proposals_ += lattice.rows() * lattice.cols();
    }

    long accepts() const { return accepts_; }

    long proposals() const { return proposals_; }

private:
    SquareLattice *target_;
    brandom::mt19937 *rng_;
    brandom::uniform_real_distribution<double> die01_;
    double ratios_[5];
    long accepts_, proposals_;
};

class WolffUpdate
{
public:
    WolffUpdate()
        : target_(NULL),
          rng_(NULL)
    { }

    WolffUpdate(SquareLattice &target, brandom::mt19937 &rng,
                double temp, double j)
        : target_(&target),
          rng_(&rng),
          die01_(0, 1),
          dierow_(0, target.rows()-1),
          diecol_(0, target.cols()-1)
    {
        if (j < 0)
            throw std::runtime_error("Cannot do Wolff for AFM case");

        pext_ = 1 - std::exp(-2 * j/temp);
    }

    void sweep()
    {
        SquareLattice &lattice = *target_;

        // step 1: select spin at random
        int i = dierow_(*rng_);
        int j = diecol_(*rng_);
        char spin = lattice.get(i, j);
        lattice.set(i, j, -spin);
        cand_.push(std::make_pair(i, j));

        // step 2: probabilistically try to grow the cluster in all directions
        while (!cand_.empty()) {
            int i = cand_.top().first, j = cand_.top().second;
            cand_.pop();
            try_extend(i-1, j, spin);
            try_extend(i+1, j, spin);
            try_extend(i, j-1, spin);
            try_extend(i, j+1, spin);
        }
    }

private:
    inline void try_extend(int i, int j, char spin)
    {
        SquareLattice &lattice = *target_;
        if (lattice.get(i, j) != spin)
            return;
        if (die01_(*rng_) >= pext_)
            return;

        i = (i + lattice.rows()) % lattice.rows();
        j = (j + lattice.cols()) % lattice.cols();
        lattice.set(i, j, -spin);
        cand_.push(std::make_pair(i, j));
    }

    SquareLattice *target_;
    brandom::mt19937 *rng_;
    brandom::uniform_real_distribution<double> die01_;
    brandom::uniform_int_distribution<int> dierow_, diecol_;

    double pext_;
    std::stack< std::pair<int, int> > cand_;
};



#endif /* UPDATES_HH_ */
