// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2024 Vissarion Fisikopoulos
// Copyright (c) 2018-2024 Apostolos Chalkis
// Copyright (c) 2024 Vladimir Necula

// Contributed and/or modified by Vladimir Necula, as part of Google Summer of
// Code 2024 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "generators/known_polytope_generators.h"
#include "random_walks/random_walks.hpp"
#include "generators/h_polytopes_generator.h"
#include "volume/volume_cooling_nonspherical_gaussians_crhmc.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include "misc/misc.h"
#include "preprocess/crhmc/crhmc_problem.h"
#include "preprocess/max_inscribed_ellipsoid.hpp"
#include "preprocess/inscribed_ellipsoid_rounding.hpp"

const unsigned int FIXED_SEED = 42;


int main() {

    typedef double NT;
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt11213b, NT, FIXED_SEED> RandomNumberGenerator;
    typedef HPolytope<Point> Polytope;
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT     NT;
    typedef typename Polytope::VT  VT;
    typedef typename Polytope::MT  MT;
    typedef typename NonSphericalGaussianFunctor::FunctionFunctor<Point>    Func;
    typedef typename NonSphericalGaussianFunctor::GradientFunctor<Point>    Grad;
    typedef typename NonSphericalGaussianFunctor::HessianFunctor<Point>     Hess;
    typedef typename NonSphericalGaussianFunctor::parameters<NT, Point>     func_params;

    typedef crhmc_input<MT, Point, Func, Grad, ZeroFunctor<Point>> Input;
    //typedef crhmc_input<MT, Point, Func, Grad, Hess> Input;   
    typedef crhmc_problem<Point, Input> CrhmcProblem;   

    typedef ImplicitMidpointODESolver<Point, NT, CrhmcProblem, Grad, 8> Solver;

    typedef typename CRHMCWalk::template Walk
            <
                    Point,
                    CrhmcProblem,
                    RandomNumberGenerator,
                    Grad,
                    Func,
                    Solver
            > CRHMCWalkType;

    typedef typename CRHMCWalk::template parameters
            <
                    NT,
                    Grad
            > crhmc_walk_params;

    int walk_len = 10;
    NT e = 0.1;
    RandomNumberGenerator rng;

    // Generate a random H-polytope with dimension 10 and 20 hyperplanes
    Polytope randomPoly = random_hpoly<Polytope, RandomNumberGenerator>(3, 30, FIXED_SEED);
    std::cout << "Random Poly \n";
    randomPoly.print();

    Polytope P(randomPoly.dimension(), randomPoly.get_mat(), randomPoly.get_vec());
    Polytope newPin(randomPoly.dimension(), randomPoly.get_mat(), randomPoly.get_vec());
    unsigned int n = P.dimension();
    unsigned int m = P.num_of_hyperplanes();
    
    //compute inscribed ellipsoid
    NT tol = std::pow(10, -6.0), reg = std::pow(10, -4.0);
    unsigned int maxiter = 100;
    P.normalize();
    VT x0 = compute_feasible_point(P.get_mat(), P.get_vec());
    auto ellipsoid_result = compute_inscribed_ellipsoid<MT, EllipsoidType::MAX_ELLIPSOID>(P.get_mat(), P.get_vec(), x0, maxiter, tol, reg);

    // extract the covariance matrix and the center of the ellipsoid
    MT inv_covariance_matrix = std::get<0>(ellipsoid_result); //this is the covariance to use in the telescopic product
    VT center = std::get<1>(ellipsoid_result);
    MT covariance_matrix = inv_covariance_matrix.inverse(); 

    newPin.shift(center); //we shift the initial polytope so that the origin is the center of the gaussians
    P.shift(center);

    // we apply the rounding transformation
    Eigen::LLT<MT> lltOfA(covariance_matrix);
    auto L = lltOfA.matrixL();
    P.linear_transformIt(L);
    // Initialize the gaussian_annealing_parameters struct
    non_gaussian_annealing_parameters<NT> parameters(P.dimension());

    // Initialization for the schedule annealing
    std::vector<NT> a_vals;
    NT ratio = parameters.ratio;
    NT C = parameters.C;
    unsigned int N = parameters.N;

    auto ball = P.ComputeInnerBall();
    P.shift(ball.first.getCoefficients()); // when using max_ellipsoid for rounding this center is the origin, but if we use other covariances this is different than the origin
    get_first_gaussian(P, parameters.frac, ball.second, e, a_vals); // the function included from volume_cooling_gaussians.hpp (spherical gaussians)


    NT a_stop = 0.0;

    if (a_vals[0]<a_stop) a_vals[0] = a_stop;

    NT initial_eta;
    Point start_point;
/*
#ifdef VOLESTI_DEBUG
    P_copy.print();
    std::cout << "first \n";
    P_copy.InnerBall().first.print();
    std::cout << "second \n" << P_copy.InnerBall().second << "\n";
    std::cout<<"-----------------------------------------------------\n\n";
    Pin_copy.print();
    std::cout << "first \n";
    Pin_copy.InnerBall().first.print();
    std::cout << "second \n" << Pin_copy.InnerBall().second << "\n";
#endif
*/
    int dimension = newPin.dimension();
    func_params initial_f_params = func_params(Point(dimension), a_vals[0], -1, inv_covariance_matrix);
    Func initial_f(initial_f_params);
    Grad initial_g(initial_f_params);
    Hess initial_h(initial_f_params);
    ZeroFunctor<Point> zerof;

    Input initial_input = convert2crhmc_input<Input, Polytope, Func, Grad, ZeroFunctor<Point>>(newPin, initial_f, initial_g, zerof);
    CrhmcProblem initial_problem = CrhmcProblem(initial_input);

    Input initial_input_2 = convert2crhmc_input<Input, Polytope, Func, Grad, ZeroFunctor<Point>>(P, initial_f, initial_g, zerof);
    CrhmcProblem initial_problem_2 = CrhmcProblem(initial_input_2);

    std::cout << "\n Initial problem terminate Pin Copy: " << initial_problem.terminate << "\n\n";
    std::cout << "\n Initial problem terminate P Copy: " << initial_problem_2.terminate << "\n\n";
    return 0;
}
