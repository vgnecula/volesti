#ifndef VOLUME_COOLING_NON_SPHERICAL_GAUSSIANS_CRHMC_HPP
#define VOLUME_COOLING_NON_SPHERICAL_GAUSSIANS_CRHMC_HPP

#define VOLESTI_DEBUG

#include "volume/volume_cooling_gaussians.hpp"
#include "preprocess/crhmc/crhmc_problem.h"

////////////////////////////// Algorithms

// Gaussian Anealling

// Compute the first variance a_0 for the starting gaussian
template <typename Polytope, typename NT>
void get_first_gaussian(Polytope& P,
                        NT const& frac,
                        NT const& chebychev_radius,
                        NT const& error,
                        std::vector<std::vector<NT>>& a_vals)
{
    NT tol = std::is_same<float, NT>::value ? 0.001 : 0.0000001;

    std::vector<Eigen::Matrix<NT, Eigen::Dynamic, 1>> dists = P.get_dists_non(chebychev_radius);
    std::vector<NT> lower(P.dimension(), 0.0);
    std::vector<NT> upper(P.dimension(), 1.0);

    // Compute an upper bound for a_0
    unsigned int i;
    const unsigned int maxiter = 10000;
    for (i = 1; i <= maxiter; ++i)
    {
        NT sum = 0.0;
        for (const auto& dist_vector : dists)
        {
            Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>::Zero(P.dimension(), P.dimension());
            for (unsigned int j = 0; j < P.dimension(); j++)
                covariance_matrix(j, j) = 1.0 / (2.0 * upper[j]);
            
            NT mahalanobis_dist = std::sqrt(dist_vector.transpose() * covariance_matrix.inverse() * dist_vector);
            sum += std::exp(-0.5 * std::pow(mahalanobis_dist, 2.0))
                / (std::pow(2.0 * M_PI, P.dimension() / 2.0) * std::sqrt(covariance_matrix.determinant()));
        }
        
        if (sum > frac * error)
        {
            for (unsigned int j = 0; j < P.dimension(); j++)
                upper[j] = upper[j] * 10;
        }
        else
            break;
    }

    if (i == maxiter) {
#ifdef VOLESTI_DEBUG
        std::cout << "Cannot obtain sharp enough starting Gaussian" << std::endl;
#endif
        return;
    }

    // get a_0 with binary search
    while (true) {
        bool converged = true;
        for (unsigned int j = 0; j < P.dimension(); j++) 
        {
            if (upper[j] - lower[j] > tol) {
                converged = false;
                break;
            }
        }
        if (converged)
            break;

        std::vector<NT> mid(P.dimension());
        for (unsigned int j = 0; j < P.dimension(); j++)
            mid[j] = (upper[j] + lower[j]) / 2.0;

        NT sum = 0.0;
        for (auto it = dists.begin(); it != dists.end(); it++) 
        {
            Eigen::Matrix<NT, Eigen::Dynamic, 1> dist_vector(*it);
            Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>::Zero(P.dimension(), P.dimension());
            
            for (unsigned int j = 0; j < P.dimension(); j++) 
                covariance_matrix(j, j) = 1.0 / (2.0 * mid[j]);

            NT mahalanobis_dist = std::sqrt(dist_vector.transpose() * covariance_matrix.inverse() * dist_vector);
            sum += std::exp(-0.5 * std::pow(mahalanobis_dist, 2.0))
                   / (std::pow(2.0 * M_PI, P.dimension() / 2.0) * std::sqrt(covariance_matrix.determinant()));
        }
        if (sum < frac * error) 
        {
            for (unsigned int j = 0; j < P.dimension(); j++)
                upper[j] = mid[j];
        } 
        else {
            for (unsigned int j = 0; j < P.dimension(); j++)
                lower[j] = mid[j];
        }
    }

    std::vector<NT> a_vals_0(P.dimension());
    for (unsigned int j = 0; j < P.dimension(); j++)
        a_vals_0[j] = (upper[j] + lower[j]) / NT(2.0);

    a_vals.push_back(a_vals_0);
}


// Compute a_{i+1} when a_i is given
template<
    typename WalkType,
    typename walk_params,
    typename RandomPointGenerator,
    int simdLen,
    typename Grad,
    typename Func,
    typename CrhmcProblem,
    typename Polytope,
    typename Point,
    typename NT,
    typename RandomNumberGenerator
>
std::vector<NT> get_next_gaussian(
    Polytope& P,
    Point &p,
    std::vector<NT> const& a,
    const unsigned int &N,
    const NT &ratio,
    const NT &C,
    const unsigned int& walk_length,
    RandomNumberGenerator& rng,
    Grad& g,
    Func& f,
    walk_params& parameters,
    CrhmcProblem& problem,
    WalkType& crhmc_walk
) {
    std::vector<NT> last_a = a;
    NT last_ratio = 0.1;
    // k is needed for the computation of the next variance a_{i+1} = a_i * (1-1/d)^k
    NT k = 1.0;
    const NT tol = 0.00001;
    bool done = false;
    std::vector<NT> fn(N, NT(0.0));
    std::list<Point> randPoints;
    typedef typename std::vector<NT>::iterator viterator;

    // sample N points
    PushBackWalkPolicy push_back_policy;
    bool raw_output = false;
    RandomPointGenerator::apply(problem, p, N, walk_length, randPoints, push_back_policy, rng, g, f, parameters, crhmc_walk, simdLen, raw_output);

    while (!done) {
        std::vector<NT> new_a(last_a.size());
        for (unsigned int j = 0; j < last_a.size(); j++) {
            new_a[j] = last_a[j] * std::pow(ratio, k);
        }

        auto fnit = fn.begin();
        for (auto pit = randPoints.begin(); pit != randPoints.end(); ++pit, ++fnit) {
            Eigen::Matrix<NT, Eigen::Dynamic, 1> dist_vector = (*pit).getCoefficients();
            Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>::Zero(P.dimension(), P.dimension());
            for (unsigned int j = 0; j < P.dimension(); j++) {
                covariance_matrix(j, j) = 1.0 / (2.0 * new_a[j]);
            }
            NT mahalanobis_dist = std::sqrt(dist_vector.transpose() * covariance_matrix.inverse() * dist_vector);
            NT last_a_prod = 1.0;
            for (unsigned int j = 0; j < last_a.size(); ++j) {
                last_a_prod *= last_a[j];
            }
            *fnit = std::exp(-0.5 * std::pow(mahalanobis_dist, 2.0))
                    / (std::pow(2.0 * M_PI, P.dimension() / 2.0) * std::sqrt(covariance_matrix.determinant()))
                    / eval_exp_non(*pit, last_a_prod);
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

    std::vector<NT> result(last_a.size());
    for (unsigned int j = 0; j < last_a.size(); j++) {
        result[j] = last_a[j] * std::pow(ratio, k);
    }

    return result;
}


// Compute the sequence of non spherical gaussians
template<
    typename WalkType,
    typename walk_params,
    typename RandomPointGenerator,
    int simdLen,
    typename Grad,
    typename Func,
    typename Hess,
    typename func_params,
    typename Input,
    typename Polytope,
    typename NT,
    typename RandomNumberGenerator
>
void compute_annealing_schedule(
    Polytope& P,
    NT const& ratio,
    NT const& C,
    NT const& frac,
    unsigned int const& N,
    unsigned int const& walk_length,
    NT const& chebychev_radius,
    NT const& error,
    std::vector<std::vector<NT>>& a_vals,
    RandomNumberGenerator& rng
) {
    typedef typename Polytope::PointType Point;
    typedef typename Polytope::VT VT;

    // Compute the first gaussian
    // This uses the function from the standard volume_cooling_gaussians.hpp
    get_first_gaussian(P, frac, chebychev_radius, error, a_vals);
    std::vector<NT> a_stop(P.dimension(), 0.0);
    const NT tol = 0.001;
    unsigned int it = 0;
    unsigned int n = P.dimension();
    const unsigned int totalSteps = ((int)150/((1.0 - frac) * error)) + 1;

    for (unsigned int j = 0; j < P.dimension(); j++) {
        if (a_vals[0][j] < a_stop[j]) a_vals[0][j] = a_stop[j];
    }

#ifdef VOLESTI_DEBUG
std::cout << "first gaussian computed " << a_vals[0][0] << std::endl;
std::cout << "Computing the sequence of gaussians..\n" << std::endl;
#endif

    while (true) {
        NT curr_fn = 0;
        NT curr_its = 0;
        auto steps = totalSteps;

        // Create the CRHMC problem for this variance
        int dimension = P.dimension();
        func_params f_params = func_params(Point(dimension), a_vals[it], 1);
        Func f(f_params);
        Grad g(f_params);
        Hess h(f_params);
        Input input = convert2crhmc_input<Input, Polytope, Func, Grad, Hess>(P, f, g, h);
        typedef crhmc_problem<Point, Input> CrhmcProblem;
        CrhmcProblem problem = CrhmcProblem(input);
        Point p = Point(problem.center);
        if(problem.terminate) { return; }
        problem.options.simdLen = simdLen;
        walk_params params(input.df, p.dimension(), problem.options);
        if (input.df.params.eta > 0) {
            params.eta = input.df.params.eta;
        }

        int dim = p.dimension();

        // Create the walk object for this problem
        WalkType walk = WalkType(problem, p, input.df, input.f, params);

#ifdef VOLESTI_DEBUG
std::cout << "Get next: " << std::endl;
#endif   

        // Compute the next gaussian
        std::vector<NT> next_a = get_next_gaussian<WalkType, walk_params, RandomPointGenerator, simdLen, Grad, Func, CrhmcProblem>(
            P, p, a_vals[it], N, ratio, C, walk_length, rng, g, f, params, problem, walk);

#ifdef VOLESTI_DEBUG
std::cout << "Next Gaussian " << next_a[0] << std::endl;
#endif

        // Compute some ratios to decide if this is the last gaussian
        for (unsigned int j = 0; j < steps; j++) {
            walk.template apply(rng, walk_length);
            p = walk.getPoint();



            curr_its += 1.0;
            Eigen::Matrix<NT, Eigen::Dynamic, 1> dist_vector = p.getCoefficients();
            Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix_next = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>::Zero(P.dimension(), P.dimension());
            Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix_curr = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>::Zero(P.dimension(), P.dimension());
            for (unsigned int k = 0; k < P.dimension(); k++) {
                covariance_matrix_next(k, k) = 1.0 / (2.0 * next_a[k]);
                covariance_matrix_curr(k, k) = 1.0 / (2.0 * a_vals[it][k]);
            }

            NT mahalanobis_dist_next = std::sqrt(dist_vector.transpose() * covariance_matrix_next.inverse() * dist_vector);
            NT mahalanobis_dist_curr = std::sqrt(dist_vector.transpose() * covariance_matrix_curr.inverse() * dist_vector);
            curr_fn += std::exp(-0.5 * std::pow(mahalanobis_dist_next, 2.0))
                        / (std::pow(2.0 * M_PI, P.dimension() / 2.0) * std::sqrt(covariance_matrix_next.determinant()))
                        / (std::exp(-0.5 * std::pow(mahalanobis_dist_curr, 2.0))
                        / (std::pow(2.0 * M_PI, P.dimension() / 2.0) * std::sqrt(covariance_matrix_curr.determinant())));
        }

#ifdef VOLESTI_DEBUG
std::cout << "Condition function ratio " << curr_fn / curr_its << std::endl;

std::cout << "Print next_a "<< std::endl;
for (unsigned int j = 0; j < P.dimension(); j++) {
    std::cout << next_a[j] << std::endl;        
}
#endif

        // Remove the last gaussian.
        // Set the last a_i equal to zero
        bool all_positive = true;
        for (unsigned int j = 0; j < P.dimension(); j++) {
            if (next_a[j] <= 0) {
                all_positive = false;
                break;
            }
        }

        if (all_positive && curr_fn / curr_its > (1.0 + tol)) {

            a_vals.push_back(next_a);
            it++;

        } else if (!all_positive) {
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
        :   frac(0.1)
        ,   ratio(NT(1)-NT(1)/(NT(d)))
        ,   C(NT(2))
        ,   N(500 * ((int) C) + ((int) (d * d / 2)))
        ,   W(6*d*d+800)
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
    typedef typename GaussianFunctor::FunctionFunctor<Point>    Func;
    typedef typename GaussianFunctor::GradientFunctor<Point>    Grad;
    typedef typename GaussianFunctor::HessianFunctor<Point>     Hess;
    typedef typename GaussianFunctor::parameters<NT, Point>     func_params;

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
            > WalkType;

    typedef typename WalkTypePolicy::template parameters
          <
                  NT,
                  Grad
          > walk_params;

    typedef CrhmcRandomPointGenerator<WalkType> RandomPointGenerator;

    auto P(Pin); //copy and work with P because we are going to shift
    unsigned int n = P.dimension();
    unsigned int m = P.num_of_hyperplanes();
    non_gaussian_annealing_parameters<NT> parameters(P.dimension());

    // Consider Chebychev center as an internal point
    auto InnerBall = P.ComputeInnerBall();
    if (InnerBall.second < 0.0) return -1.0;

    Point c = InnerBall.first;
    NT radius = InnerBall.second;

    // Move the chebychev center to the origin and apply the same shifting to the polytope
    P.shift(c.getCoefficients());

    // Computing the sequence of gaussians
#ifdef VOLESTI_DEBUG
    std::cout<<"\n\nComputing annealing...\n"<<std::endl;
    double tstart2 = (double)clock()/(double)CLOCKS_PER_SEC;
#endif

    // Initialization for the schedule annealing
    std::vector<std::vector<NT>> a_vals;
    NT ratio = parameters.ratio;
    NT C = parameters.C;
    unsigned int N = parameters.N;

    compute_annealing_schedule
    <
        WalkType,
        walk_params,
        RandomPointGenerator,
        simdLen,
        Grad,
        Func,
        Hess,
        func_params,
        Input
    >(P, ratio, C, parameters.frac, N, walk_length, radius, error, a_vals, rng);


#ifdef VOLESTI_DEBUG
    std::cout<<"All the variances of schedule_annealing computed in = "
            << (double)clock()/(double)CLOCKS_PER_SEC-tstart2<<" sec"<<std::endl;
    auto j=0;
    for (auto avalIt = a_vals.begin(); avalIt!=a_vals.end(); avalIt++, j++){
        std::cout<<"a_"<<j<<" = ";
        for (auto aIt = avalIt->begin(); aIt != avalIt->end(); aIt++) {
            std::cout << *aIt << " ";
        }
        std::cout << std::endl;
    }
    std::cout<<std::endl;
#endif

    // Initialization for the approximation of the ratios
    unsigned int W = parameters.W;
    unsigned int mm = a_vals.size()-1;
    std::vector<NT> last_W2(W,0);
    std::vector<NT> fn(mm,0);
    std::vector<NT> its(mm,0);
    VT lamdas;
    lamdas.setZero(m);
    
    NT vol = 1.0;
    for (unsigned int j = 0; j < n; j++) {
        vol *= std::sqrt(M_PI / a_vals[0][j]);
    }

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
        func_params f_params = func_params(Point(dimension), *avalsIt, 1);
        
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
        walk_params params(input.df, p.dimension(), problem.options);

        if (input.df.params.eta > 0) {
            params.eta = input.df.params.eta;
        }

        WalkType walk = WalkType(problem, p, input.df, input.f, params);

        // Precalculate the covariance matrices outside the while loop
        Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix_next = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>::Zero(P.dimension(), P.dimension());
        Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix_curr = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>::Zero(P.dimension(), P.dimension());

        const NT epsilon = std::numeric_limits<NT>::epsilon() * 1000; // Adjust as needed

        for (unsigned int k = 0; k < P.dimension(); k++) {
            NT next_val = std::max((*(avalsIt+1))[k], epsilon);
            NT curr_val = std::max((*avalsIt)[k], epsilon);
            covariance_matrix_next(k, k) = 1.0 / (2.0 * next_val);
            covariance_matrix_curr(k, k) = 1.0 / (2.0 * curr_val);
        }

        // Precalculate the determinants and inverse matrices
        NT det_next = covariance_matrix_next.determinant();
        NT det_curr = covariance_matrix_curr.determinant();
        Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> inv_next = covariance_matrix_next.inverse();
        Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> inv_curr = covariance_matrix_curr.inverse();


        while (!done || (*itsIt)<min_steps)
        {
            walk.template apply(rng, walk_length);
            p = walk.getPoint();
            *itsIt = *itsIt + 1.0;
            
            Eigen::Matrix<NT, Eigen::Dynamic, 1> dist_vector = p.getCoefficients();
            
            NT mahalanobis_dist_next = std::sqrt(dist_vector.transpose() * inv_next * dist_vector);
            NT mahalanobis_dist_curr = std::sqrt(dist_vector.transpose() * inv_curr * dist_vector);
            
            NT density_next = std::exp(-0.5 * std::pow(mahalanobis_dist_next, 2.0))
                            / (std::pow(2.0 * M_PI, P.dimension() / 2.0) * std::sqrt(det_next));
            
            NT density_curr = std::exp(-0.5 * std::pow(mahalanobis_dist_curr, 2.0))
                            / (std::pow(2.0 * M_PI, P.dimension() / 2.0) * std::sqrt(det_curr));
            
            *fnIt += density_next / density_curr;

            NT val = (*fnIt) / (*itsIt);

#ifdef VOLESTI_DEBUG
        std::cout << "density curr: "<< density_curr << std::endl;
#endif

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

            if ( (max_val-min_val)/max_val <= curr_eps/2.0 )
            {
                done=true;
            }

            index = index%W + 1;
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