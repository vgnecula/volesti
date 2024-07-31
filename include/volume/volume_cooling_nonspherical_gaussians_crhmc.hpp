// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2024 Vissarion Fisikopoulos
// Copyright (c) 2018-2024 Apostolos Chalkis
// Copyright (c) 2024 Vladimir Necula

// Contributed and/or modified by Vladimir Necula, as part of Google Summer of
// Code 2024 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_COOLING_NON_SPHERICAL_GAUSSIANS_CRHMC_HPP
#define VOLUME_COOLING_NON_SPHERICAL_GAUSSIANS_CRHMC_HPP

// #define VOLESTI_DEBUG

#include "volume/volume_cooling_gaussians.hpp"
#include "preprocess/crhmc/crhmc_problem.h"
#include "preprocess/max_inscribed_ellipsoid.hpp"
#include "preprocess/inscribed_ellipsoid_rounding.hpp"

////////////////////////////// Algorithms

// Gaussian Anealling

// Compute the first variance a_0 for the starting gaussian
template <typename Polytope, typename NT, typename MT>
void get_first_gaussian(Polytope& P,
                        NT const& frac,
                        NT const& error,
                        MT const& inv_covariance_matrix,
                        std::vector<NT>& a_vals)
{
    NT tol = std::is_same<float, NT>::value ? 0.001 : 0.0000001;
    std::vector<Eigen::Matrix<NT, Eigen::Dynamic, 1>> dists = P.get_dists();
    NT lower = 0.0;
    NT upper = 1.0;

    // Compute an upper bound for a_0
    unsigned int i;
    const unsigned int maxiter = 10000;
    for (i = 1; i <= maxiter; ++i)
    {
        NT sum = 0.0;
        for (const auto& dist_vector : dists)
        {
            MT scaled_inv_covariance = upper * inv_covariance_matrix;
            NT mahalanobis_dist = std::sqrt(dist_vector.transpose() * scaled_inv_covariance * dist_vector);
            sum += std::exp(-0.5 * std::pow(mahalanobis_dist, 2.0))
                   / (std::pow(2.0 * M_PI, P.dimension() / 2.0) * std::sqrt(scaled_inv_covariance.determinant()));
        }
        if (sum > frac * error)
        {
            upper *= 10;
        } else {
            break;
        }
    }

    if (i == maxiter) {
#ifdef VOLESTI_DEBUG
        std::cout << "Cannot obtain sharp enough starting Gaussian" << std::endl;
#endif
        return;
    }

    // get a_0 with binary search
    while (upper - lower > tol) {
        NT mid = (upper + lower) / 2.0;
        NT sum = 0.0;
        for (const auto& dist_vector : dists)
        {
            MT scaled_inv_covariance = mid * inv_covariance_matrix;
            NT mahalanobis_dist = std::sqrt(dist_vector.transpose() * scaled_inv_covariance * dist_vector);
            sum += std::exp(-0.5 * std::pow(mahalanobis_dist, 2.0))
                   / (std::pow(2.0 * M_PI, P.dimension() / 2.0) * std::sqrt(scaled_inv_covariance.determinant()));
        }
        if (sum < frac * error)
        {
            upper = mid;
        }
        else {
            lower = mid;
        }
    }

    NT a_0 = (upper + lower) / NT(2.0);
    a_vals.push_back(a_0);;
}


// Compute a_{i+1} when a_i is given
template
<
    typename CRHMCWalkType,
    typename crhmc_walk_params,
    int simdLen,
    typename Grad,
    typename Func,
    typename CrhmcProblem,
    typename Polytope,
    typename Point,
    typename NT,
    typename MT,
    typename RandomNumberGenerator
>
NT get_next_gaussian(Polytope& P,
                    Point &p,
                    NT const& a,
                    const unsigned int &N,
                    const NT &ratio,
                    const NT &C,
                    const unsigned int& walk_length,
                    RandomNumberGenerator& rng,
                    Grad& g,
                    Func& f,
                    crhmc_walk_params& parameters,
                    CrhmcProblem& problem,
                    CRHMCWalkType& crhmc_walk,
                    MT const& inv_covariance_matrix) 
{
    NT last_a = a;
    NT last_ratio = 0.1;
    NT k = 1.0;
    const NT tol = 0.00001;
    bool done = false;
    std::vector<NT> fn(N, NT(0.0));
    std::list<Point> randPoints;

    // sample N points
    PushBackWalkPolicy push_back_policy;
    bool raw_output = false;
    typedef CrhmcRandomPointGenerator<CRHMCWalkType> CRHMCRandomPointGenerator;

    CRHMCRandomPointGenerator::apply(problem, p, N, walk_length, randPoints,
                                    push_back_policy, rng, g, f, parameters, crhmc_walk, simdLen, raw_output);

    while (!done) {
        NT new_a = last_a * std::pow(ratio, k);

        auto fnit = fn.begin();
        for (auto pit = randPoints.begin(); pit != randPoints.end(); ++pit, fnit++) {
            *fnit = eval_exp(*pit, inv_covariance_matrix, new_a, last_a);
        }

        std::pair<NT, NT> mv = get_mean_variance(fn);


        // Compute a_{i+1}
        if (mv.second / (mv.first * mv.first) >= C || mv.first / last_ratio < 1.0 + tol) {
            if (k != 1.0) {
                k = k / 2;
            }
            done = true;
        } else {
            k = 2 * k;
        }

        last_ratio = mv.first;
    }

    // Return the new a value as a scalar
    return last_a * std::pow(ratio, k);
}


// Compute the sequence of non spherical gaussians
template<
    int simdLen,
    typename Polytope,
    typename NT,
    typename MT,
    typename RandomNumberGenerator
>
void compute_annealing_schedule(Polytope& P,
                                NT const& ratio,
                                NT const& C,
                                NT const& frac,
                                unsigned int const& N,
                                unsigned int const& walk_length,
                                NT const& error,
                                std::vector<NT>& a_vals,
                                MT const& inv_covariance_matrix,
                                RandomNumberGenerator& rng) 
{
    typedef typename Polytope::PointType Point;
    
    typedef typename NonSphericalGaussianFunctor::FunctionFunctor<Point>    Func;
    typedef typename NonSphericalGaussianFunctor::GradientFunctor<Point>    Grad;
    typedef typename NonSphericalGaussianFunctor::HessianFunctor<Point>     Hess;
    typedef typename NonSphericalGaussianFunctor::parameters<NT, Point>     func_params;

    typedef crhmc_input<MT, Point, Func, Grad, Hess> Input;
    typedef crhmc_problem<Point, Input> CrhmcProblem;   

    typedef ImplicitMidpointODESolver<Point, NT, CrhmcProblem, Grad, simdLen> Solver;

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

    typedef CrhmcRandomPointGenerator<CRHMCWalkType> CRHMCRandomPointGenerator;


    // Compute the first gaussian
    // This uses the function from the standard volume_cooling_gaussians.hpp
    get_first_gaussian(P, frac, error, inv_covariance_matrix, a_vals);

#ifdef VOLESTI_DEBUG
    std::cout << "first gaussian computed " << a_vals[0] << std::endl;
#endif

    NT a_stop = 0.0;
    const NT tol = 0.001;
    unsigned int it = 0;
    unsigned int n = P.dimension();
    const unsigned int totalSteps = ((int)150/((1.0 - frac) * error)) + 1;

    if (a_vals[0]<a_stop) a_vals[0] = a_stop;

#ifdef VOLESTI_DEBUG
    std::cout << "Computing the sequence of gaussians..\n" << std::endl;
#endif

    while (true) {

        
        NT curr_fn = 0;
        NT curr_its = 0;
        auto steps = totalSteps;

        //TODO: potential problem creation and preprocessing optimization

        // Create the CRHMC problem for this variance
        int dimension = P.dimension();
        func_params f_params = func_params(Point(dimension), a_vals[it], 1, inv_covariance_matrix);
        
        Func f(f_params);
        Grad g(f_params);
        Hess h(f_params);
        
        Input input = convert2crhmc_input<Input, Polytope, Func, Grad, Hess>(P, f, g, h);
        
        typedef crhmc_problem<Point, Input> CrhmcProblem;
        CrhmcProblem problem = CrhmcProblem(input);
        
        Point p = Point(problem.center);
        
        if(problem.terminate) { return; }
        
        problem.options.simdLen = simdLen;
        crhmc_walk_params params(input.df, p.dimension(), problem.options);
        
        if (input.df.params.eta > 0) {
            params.eta = input.df.params.eta;
        }

        // Create the walk object for this problem
        CRHMCWalkType walk = CRHMCWalkType(problem, p, input.df, input.f, params);

        // Compute the next gaussian
        NT next_a = get_next_gaussian<CRHMCWalkType, crhmc_walk_params, simdLen, Grad, Func, CrhmcProblem>(
            P, p, a_vals[it], N, ratio, C, walk_length, rng, g, f, params, problem, walk, inv_covariance_matrix);

#ifdef VOLESTI_DEBUG
std::cout << "Next Gaussian " << next_a << std::endl;
#endif

        // Compute some ratios to decide if this is the last gaussian
        for (unsigned int j = 0; j < steps; j++) 
        {
            walk.template apply(rng, walk_length);
            p = walk.getPoint();
            curr_its += 1.0;    
            curr_fn += eval_exp(p, inv_covariance_matrix, next_a, a_vals[it]);
        }

#ifdef VOLESTI_DEBUG
    std::cout<<"Ratio = "<< curr_fn / curr_its <<"\n"<<std::endl;
#endif
        // Remove the last gaussian.
        // Set the last a_i equal to zero
        if (next_a > 0 && curr_fn / curr_its > (1.0 + tol)) 
        {
            a_vals.push_back(next_a);
            it++;
        } else if (next_a <= 0) 
        {
            a_vals.push_back(a_stop);
            it++;
            break;
        } else {
            a_vals[it] = a_stop;
            break;
        }
    }
}


template <typename NT>
struct non_gaussian_annealing_parameters
{
    non_gaussian_annealing_parameters(unsigned int d)
        : frac(0.1)
        , ratio(NT(1) - NT(1) / NT(d))
        , C(NT(2))
        , N(500 * ((int) C) + ((int) (d * d)))
        , W(6 * d * d + 800)
    {}

    NT frac;
    NT ratio;
    NT C;
    unsigned int N;
    unsigned int W;
};


template
<
    typename Polytope,
    typename RandomNumberGenerator,
    typename WalkTypePolicy = CRHMCWalk,
    int simdLen = 8
>
double non_spherical_crhmc_volume_cooling_gaussians(Polytope& Pin,
                                RandomNumberGenerator& rng,
                                double const& error = 0.1,
                                unsigned int const& walk_length = 1)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT     NT;
    typedef typename Polytope::VT  VT;
    typedef typename Polytope::MT  MT;
    typedef typename NonSphericalGaussianFunctor::FunctionFunctor<Point>    Func;
    typedef typename NonSphericalGaussianFunctor::GradientFunctor<Point>    Grad;
    typedef typename NonSphericalGaussianFunctor::HessianFunctor<Point>     Hess;
    typedef typename NonSphericalGaussianFunctor::parameters<NT, Point>     func_params;

    typedef crhmc_input<MT, Point, Func, Grad, Hess> Input;
    typedef crhmc_problem<Point, Input> CrhmcProblem;   

    typedef ImplicitMidpointODESolver<Point, NT, CrhmcProblem, Grad, simdLen> Solver;

    typedef typename WalkTypePolicy::template Walk
            <
                    Point,
                    CrhmcProblem,
                    RandomNumberGenerator,
                    Grad,
                    Func,
                    Solver
            > CRHMCWalkType;

    typedef typename WalkTypePolicy::template parameters
          <
                  NT,
                  Grad
          > crhmc_walk_params;

    typedef CrhmcRandomPointGenerator<CRHMCWalkType> RandomPointGenerator;

    auto P(Pin); //copy and work with P because we are going to shift
    unsigned int n = P.dimension();
    unsigned int m = P.num_of_hyperplanes();
    
    // Get the matrix A and vector b of the polytope
    MT A = P.get_mat();
    VT b = P.get_vec();

    // get the inverse covariance matrix of themax inscribed ellipsoid
    Point q(P.dimension());
    NT tol = std::pow(10, -6.0), reg = std::pow(10, -4.0);
    VT x0 = q.getCoefficients();
    unsigned int maxiter = 100;
    auto ellipsoid_result = max_inscribed_ellipsoid<MT>(A, b, x0, maxiter, tol, reg);

    // Extract the covariance matrix and center of the ellipsoid
    MT inv_covariance_matrix = std::get<0>(ellipsoid_result);
    Point center(std::get<1>(ellipsoid_result));


#ifdef VOLESTI_DEBUG
    std::cout<<"\n\nExtracted the covariance matrix...\n"<<std::endl;
    std::cout<< inv_covariance_matrix <<std::endl;
#endif

    // do the polytope rounding
    auto rounding_result = inscribed_ellipsoid_rounding<MT, VT, NT>(P, center);
    MT T = std::get<0>(rounding_result);
    VT shift = std::get<1>(rounding_result);
    NT round_val = std::get<2>(rounding_result);

    // Modify P
    P.shift(shift);
    P.linear_transformIt(T);

    // Initialize the gaussian_annealing_parameters struct
    non_gaussian_annealing_parameters<NT> parameters(P.dimension());


    // Computing the sequence of gaussians
#ifdef VOLESTI_DEBUG
    std::cout<<"\n\nComputing annealing...\n"<<std::endl;
    double tstart2 = (double)clock()/(double)CLOCKS_PER_SEC;
#endif

    // Initialization for the schedule annealing
    std::vector<NT> a_vals;
    NT ratio = parameters.ratio;
    NT C = parameters.C;
    unsigned int N = parameters.N;

    compute_annealing_schedule<simdLen>(P, ratio, C, parameters.frac, N, walk_length, error, a_vals, inv_covariance_matrix, rng);


#ifdef VOLESTI_DEBUG
    std::cout<<"All the variances of schedule_annealing computed in = "
            << (double)clock()/(double)CLOCKS_PER_SEC-tstart2<<" sec"<<std::endl;
    auto j=0;
    for (auto avalIt = a_vals.begin(); avalIt!=a_vals.end(); avalIt++, j++){
        std::cout<<"a_"<<j<<" = "<<*avalIt<<" ";
    }
    std::cout<<std::endl<<std::endl;
#endif

    // Initialization for the approximation of the ratios
    unsigned int W = parameters.W;
    unsigned int mm = a_vals.size()-1;
    std::vector<NT> last_W2(W,0);
    std::vector<NT> fn(mm,0);
    std::vector<NT> its(mm,0);
    VT lamdas;
    lamdas.setZero(m);
    
    NT vol = std::pow(M_PI / a_vals[0], NT(n) / 2.0);

    unsigned int i=0;

    typedef typename std::vector<NT>::iterator viterator;
    viterator itsIt = its.begin();
    auto avalsIt = a_vals.begin();
    viterator minmaxIt;


#ifdef VOLESTI_DEBUG
    std::cout<<"volume of the first gaussian = "<<vol<<"\n"<<std::endl;
    std::cout<<"computing ratios..\n"<<std::endl;
#endif

    //iterate over the number of ratios
    for (viterator fnIt = fn.begin();
         fnIt != fn.end();
         fnIt++, itsIt++, avalsIt++, i++)
    {
        //initialize convergence test
        bool done = false;
        NT curr_eps = error/std::sqrt((NT(mm)));
        NT min_val = std::numeric_limits<NT>::min();
        NT max_val = std::numeric_limits<NT>::max();
        unsigned int min_index = W-1;
        unsigned int max_index = W-1;
        unsigned int index = 0;
        unsigned int min_steps = 0;
        std::vector<NT> last_W = last_W2;

        // Set the radius for the ball walk
        //creating the walk object
        int dimension = P.dimension();
        func_params f_params = func_params(Point(dimension), *avalsIt, 1, inv_covariance_matrix);
        
        Func f(f_params);
        Grad g(f_params);
        Hess h(f_params);

        //create the crhmc problem
        Input input = convert2crhmc_input<Input, Polytope, Func, Grad, Hess>(P, f, g, h);

        typedef crhmc_problem<Point, Input> CrhmcProblem;
        CrhmcProblem problem = CrhmcProblem(input);

        Point p = Point(problem.center);

        if(problem.terminate){return 0;}

        problem.options.simdLen=simdLen;
        crhmc_walk_params params(input.df, p.dimension(), problem.options);

        if (input.df.params.eta > 0) {
            params.eta = input.df.params.eta;
        }

        CRHMCWalkType walk = CRHMCWalkType(problem, p, input.df, input.f, params);

        while (!done || (*itsIt) < min_steps)
        {
            walk.template apply(rng, walk_length);
            p = walk.getPoint();
            *itsIt = *itsIt + 1.0;
            
            *fnIt += eval_exp(p, inv_covariance_matrix, *(avalsIt+1), *avalsIt);

            NT val = (*fnIt) / (*itsIt);

            last_W[index] = val;
            if (val <= min_val)
            {
                min_val = val;
                min_index = index;
            } else if (min_index == index)
            {
                minmaxIt = std::min_element(last_W.begin(), last_W.end());
                min_val = *minmaxIt;
                min_index = std::distance(last_W.begin(), minmaxIt);
            }

            if (val >= max_val)
            {
                max_val = val;
                max_index = index;
            } else if (max_index == index)
            {
                minmaxIt = std::max_element(last_W.begin(), last_W.end());
                max_val = *minmaxIt;
                max_index = std::distance(last_W.begin(), minmaxIt);
            }

            if ((max_val - min_val) / max_val <= curr_eps / 2.0)
            {
                done = true;
            }

            index = index % W + 1;
            if (index == W) index = 0;
        }
#ifdef VOLESTI_DEBUG
        std::cout << "ratio " << i << " = " << (*fnIt) / (*itsIt)
                  << " N_" << i << " = " << *itsIt << std::endl;
#endif
        vol *= ((*fnIt) / (*itsIt));
    }

#ifdef VOLESTI_DEBUG
        NT sum_of_steps = 0.0;
        for(viterator it = its.begin(); it != its.end(); ++it) {
            sum_of_steps += *it;
        }
        auto steps= int(sum_of_steps);
        std::cout<<"\nTotal number of steps = "<<steps<<"\n"<<std::endl;
#endif

    return vol;
}

#endif // VOLUME_COOLING_NON_SPHERICAL_GAUSSIANS_CRHMC_HPP
