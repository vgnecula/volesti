// Use forward-mode automatic differentiation using Autodiff Library

// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou
// Copyright (c) 2022-2022 Zhang zhuyan

// Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.
// Contributed and/or modified by Zhang zhuyan, as part of Google Summer of Code 2020 program.

// Licensed under GNU LGPL.3, see LICENCE file

// task number 1 use array and fix the problem first done
// difference between EigenArray and EigenVector

#include <iostream>
#include <cmath>
#include <functional>
#include <vector>
#include <unistd.h>
#include <string>
#include <typeinfo>

#include "Eigen/Eigen"

#include "ode_solvers/ode_solvers.hpp"
#include "ode_solvers/oracle_autodiff_functors.hpp"
#include "random.hpp"
#include "random/uniform_int.hpp"
#include "random/normal_distribution.hpp"
#include "random/uniform_real_distribution.hpp"
#include "random_walks/random_walks.hpp"
#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_balls.hpp"
#include "generators/known_polytope_generators.h"
#include "readData.h"
#include "diagnostics/diagnostics.hpp"
#include "cartesian_geom/autopoint.h"

template <typename NT>
void run_main()
{
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point> Hpolytope;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;
    typedef AutoDiffFunctor::GradientFunctor<Point> NegativeGradientFunctor;
    typedef AutoDiffFunctor::FunctionFunctor<Point> NegativeLogprobFunctor;
    typedef LeapfrogODESolver<Point, NT, Hpolytope, NegativeGradientFunctor> Solver;
    typedef typename Hpolytope::MT MT;
    typedef typename Hpolytope::VT VT;
    AutoDiffFunctor::parameters<NT> params;
    params.data = readMatrix<NT>("data.txt");
    NegativeGradientFunctor F(params);
    NegativeLogprobFunctor f(params);
    RandomNumberGenerator rng(1);
    unsigned int dim = 2;

    HamiltonianMonteCarloWalk::parameters<NT, NegativeGradientFunctor> hmc_params(F, dim);
    hmc_params.eta=0.00005;// working learning rate for this specific example
    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;

    Hpolytope P = generate_cube<Hpolytope>(dim, false);

    Point x0 = -0.25 * Point::all_ones(dim);

    // In the first argument put in the address of an H-Polytope
    // for truncated sampling and NULL for untruncated
    HamiltonianMonteCarloWalk::Walk<Point, Hpolytope, RandomNumberGenerator, NegativeGradientFunctor, NegativeLogprobFunctor, Solver>
        hmc(&P, x0, F, f, hmc_params);
    int n_samples = 50000; // Half will be burned
    int max_actual_draws = n_samples / 2;
    unsigned int min_ess = 0;
    MT samples;
    samples.resize(dim, max_actual_draws);

    for (int i = 0; i < n_samples - max_actual_draws; i++) {
        hmc.apply(rng, 3);
    }
    start = std::chrono::high_resolution_clock::now();
    std::cerr << (long)std::chrono::duration_cast<std::chrono::microseconds>(start - stop).count();
    for (int i = 0; i < max_actual_draws; i++) {
        std::cout << hmc.x.getCoefficients().transpose() << std::endl;
        hmc.apply(rng, 3);
        samples.col(i) = hmc.x.getCoefficients();
    }
    stop = std::chrono::high_resolution_clock::now();
    long ETA = (long)std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cerr << "total time taken " << ETA << std::endl;
    std::cerr << std::endl;
    print_diagnostics<NT, VT, MT>(samples, min_ess, std::cerr);
    std::cerr << "min ess " << min_ess << "us" << std::endl;
    std::cerr << "Average time per sample: " << ETA / max_actual_draws << "us" << std::endl;
    std::cerr << "Average time per independent sample: " << ETA / min_ess << "us" << std::endl;
    std::cerr << "Average number of reflections: " << (1.0 * hmc.solver->num_reflections) / hmc.solver->num_steps << std::endl;
    std::cerr << "Step size (final): " << hmc.solver->eta << std::endl;
    std::cerr << "Discard Ratio: " << hmc.discard_ratio << std::endl;
    std::cerr << "Average Acceptance Probability: " << exp(hmc.average_acceptance_log_prob) << std::endl;
    std::cerr << "PSRF: " << multivariate_psrf<NT, VT, MT>(samples) << std::endl;
    std::cerr << std::endl;
}
using TT = double;
typedef Eigen::Matrix<TT,Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;
typename autopoint<TT>::FT pdf_(const autopoint<TT> &x, const Eigen::Matrix<TT, Eigen::Dynamic, 1> &data_) {
    // define your function here,
    autopoint<TT> data_auto = autopoint(data_);
    autopoint<TT> result = (((-0.5 * 100 * (data_auto - x.getCoefficients()[0]).pow(2)).exp() + (-0.5 * 100 * (data_auto - x.getCoefficients()[1]).pow(2)).exp())).log();

    auto y = (result * -1.0).sum();
    return y;
}

template <>
std::function<typename autopoint<TT>::FT(const autopoint<TT> &, const EigenMatrix&)> AutoDiffFunctor::FunctionFunctor_internal<TT>::pdf = pdf_;

int main() {
    run_main<double>();
    return 0;
}
