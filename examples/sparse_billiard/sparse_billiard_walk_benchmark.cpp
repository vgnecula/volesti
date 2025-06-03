#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <boost/random.hpp>
#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "sampling/random_point_generators.hpp"
#include "random_walks/random_walks.hpp"
#include "convex_bodies/hpolytope.h"
#include "generators/h_polytopes_generator.h"
#include "generators/known_polytope_generators.h"
#include "diagnostics/effective_sample_size.hpp"
#include "diagnostics/multivariate_psrf.hpp"
#include "diagnostics/univariate_psrf.hpp"
#include "generators/order_polytope_generator.h"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;
typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;
typedef BoostRandomNumberGenerator<boost::mt19937, NT> RNGType;

// Define both dense and sparse polytope types
typedef HPolytope<Point> DenseHPOLYTOPE;
typedef HPolytope<Point, Eigen::SparseMatrix<NT, Eigen::RowMajor>> SparseHPOLYTOPE;
typedef typename DenseHPOLYTOPE::LLTType DenseLLTType;
typedef typename SparseHPOLYTOPE::LLTType SparseLLTType;  // Use the polytope's LLT type

typedef BilliardWalk::template Walk<DenseHPOLYTOPE, RNGType> DenseBilliardWalkType;
typedef BilliardWalk::template Walk<SparseHPOLYTOPE, RNGType> SparseBilliardWalkType;

PushBackWalkPolicy push_back_policy;

const unsigned int FIXED_SEED = 42;  // Fixed seed for reproducibility

template<class HPolytope>
typename HPolytope::PointType
analytic_center(const HPolytope& P,
                int    max_it = 10,
                typename HPolytope::NT tol = 1e-8)
{
    using NT = typename HPolytope::NT;
    using VT = typename HPolytope::VT;
    using MT = typename HPolytope::MT;
    using Point = typename HPolytope::PointType;
    using LLTType = typename HPolytope::LLTType;

    MT const&  A = P.get_mat();
    VT const&  b = P.get_vec();

    Point x = P.InnerBall().first;

    for (int it = 0; it < max_it; ++it) {
        // For sparse matrices, we need to convert to dense for some operations
        VT x_coeff = x.getCoefficients();
        VT slack = b - A * x_coeff;    // m×1  (sparse·dense)
        if (slack.minCoeff() <= NT(0)) break;      // numerical guard

        /* gradient  g = Aᵀ (1/slack) */
        VT slack_inv = slack.cwiseInverse();
        VT g = A.transpose() * slack_inv;

        if (g.norm() < tol) break;                 // converged

        /* Hessian  H = Aᵀ diag(1/slack²) A   (sparse) */
        VT d = slack.array().pow(-2);
        
        // For sparse matrices, we need to handle the diagonal multiplication carefully
        MT H;
        if constexpr (std::is_same<MT, Eigen::SparseMatrix<NT>>::value) {
            // For sparse matrices, use sparse operations
            Eigen::SparseMatrix<NT> D = d.asDiagonal();
            H = A.transpose() * D * A;
        } else {
            // For dense matrices, use dense operations
            H = A.transpose() * d.asDiagonal() * A;
        }

        /* Newton step: solve  H s = g  with LLT */
        LLTType llt;
        llt.compute(H);
        VT step = llt.solve(g);

        /* back-tracking line search (keeps x inside P) */
        NT α = 1.0;
        while (α > NT(1e-4)) {
            Point x_new = x - α * Point(step);
            if (P.is_in(x_new)) { x = x_new; break; }
            α *= NT(0.5);
        }
        if (α <= NT(1e-4)) break;
    }
    return x;
}

struct BenchmarkResults {
    NT ess_min;
    NT ess_avg;
    NT psrf_max;
    double time_walk;       // only track walk time
    std::string walk_type;
    int dimension;
    int num_samples;
};

// Function to run benchmark with regular billiard walk
BenchmarkResults benchmark_regular_billiard_walk(DenseHPOLYTOPE& P, unsigned int num_samples, unsigned int walk_length) {
    std::cout << "Benchmarking Regular Billiard Walk..." << std::endl;
    
    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;
    
    RNGType rng(P.dimension());
    rng.set_seed(FIXED_SEED);  // Set fixed seed
    
    // Use lpsolve directly for inner ball computation to ensure consistency
    Point starting_point = P.ComputeInnerBall().first;
    
    auto t1 = clock::now();
    
    // Generate samples using regular billiard walk
    std::vector<Point> randPoints;
    typedef RandomPointGenerator<DenseBilliardWalkType> Generator;
    Generator::apply(P, starting_point, num_samples, walk_length,
                     randPoints, push_back_policy, rng);
    
    auto t2 = clock::now();
    
    // Convert to matrix for multichain PSRF computation
    MT samples(P.dimension(), num_samples);
    for (size_t i = 0; i < randPoints.size(); ++i) {
        samples.col(i) = randPoints[i].getCoefficients();
    }
    
    // Compute PSRF using standard multivariate function
    NT psrf = multivariate_psrf<NT, VT, MT>(samples);
    
    // Compute diagnostics
    unsigned int min_ess;
    VT ess_vector = effective_sample_size<NT, VT, MT>(samples, min_ess);
    
    BenchmarkResults results;
    results.ess_min = ess_vector.minCoeff();
    results.ess_avg = ess_vector.mean();
    results.psrf_max = psrf;
    results.time_walk = seconds(t2 - t1).count();
    results.walk_type = "Regular Billiard";
    results.dimension = P.dimension();
    results.num_samples = num_samples;
    
    return results;
}

// Function to run benchmark with sparse billiard walk
BenchmarkResults benchmark_sparse_billiard_walk(SparseHPOLYTOPE& P, unsigned int num_samples, unsigned int walk_length) {
    std::cout << "Benchmarking Sparse Billiard Walk..." << std::endl;
    
    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;
    
    RNGType rng(P.dimension());
    rng.set_seed(FIXED_SEED);  // Set fixed seed
    
    // Use lpsolve directly for inner ball computation to ensure consistency
    Point starting_point = P.ComputeInnerBall().first;
    
    // Compute analytic center with fixed seed
    Point x_ac = analytic_center(P);
    
    // Shift the polytope to use the analytic center as the origin
    auto A = P.get_mat();
    VT b = P.get_vec();
    VT slack = b - A * x_ac.getCoefficients();
    SparseHPOLYTOPE P_round = P;
    P_round.shift(x_ac.getCoefficients());
    P_round.normalize();
    
    // Create sparse LLT for on-the-fly rounding
    SparseLLTType llt;
    try {
        // First try to compute the LLT decomposition
        llt.compute(P_round.get_mat().transpose() * P_round.get_mat());
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("LLT decomposition failed");
        }
        
        // Now compute the rounded diameter with fixed seed
        RNGType diameter_rng(P_round.dimension());
        diameter_rng.set_seed(FIXED_SEED);
        double L_round = rounded_diameter(P_round, llt, diameter_rng);
        if (L_round <= 0 || std::isnan(L_round)) {
            throw std::runtime_error("Invalid rounded diameter");
        }
        
        BilliardWalk::parameters parms(L_round, true);
        Point origin(P.dimension());
        origin.set_to_origin();
        
        // Create walk with LLT factor and use sparse reflection
        SparseBilliardWalkType walk(P_round, origin, rng, parms, llt);
        std::vector<Point> randPoints;
        randPoints.reserve(num_samples);

        auto t1 = clock::now();
        
        Point current_point = origin;
        unsigned int successful_samples = 0;
        for (unsigned int i = 0; i < num_samples; ++i) {
            try {
                walk.apply(P_round, current_point, walk_length, rng);  // u is managed internally
                randPoints.push_back(Point(current_point.getCoefficients() + x_ac.getCoefficients()));
                successful_samples++;
            } catch (const std::exception& e) {
                std::cerr << "Error during walk step " << i << ": " << e.what() << std::endl;
                // Continue with next sample
                continue;
            }
        }
        
        if (successful_samples == 0) {
            throw std::runtime_error("No successful samples were generated");
        }
        
        auto t2 = clock::now();
        
        // Convert to matrix for multichain PSRF computation
        MT samples(P.dimension(), successful_samples);
        for (size_t i = 0; i < randPoints.size(); ++i) {
            samples.col(i) = randPoints[i].getCoefficients();
        }
        
        NT psrf = multivariate_psrf<NT, VT, MT>(samples);
        
        unsigned int min_ess;
        VT ess_vector = effective_sample_size<NT, VT, MT>(samples, min_ess);
        
        BenchmarkResults results;
        results.ess_min = ess_vector.minCoeff();
        results.ess_avg = ess_vector.mean();
        results.psrf_max = psrf;
        results.time_walk = seconds(t2 - t1).count();
        results.walk_type = "Sparse Billiard";
        results.dimension = P.dimension();
        results.num_samples = successful_samples;
        
        return results;
    } catch (const std::exception& e) {
        std::cerr << "Error in sparse billiard walk: " << e.what() << std::endl;
        throw;
    }
}

void print_results(const std::vector<BenchmarkResults>& results) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::left << std::setw(15) << "Walk Type"
              << std::setw(8) << "Dim " 
              << std::setw(10) << "Samples "
              << std::setw(12) << "Time Walk(s) "
              << std::setw(12) << "Min ESS "
              << std::setw(12) << "Avg ESS "
              << std::setw(12) << "Max PSRF " << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(15) << result.walk_type
                  << std::setw(8) << result.dimension
                  << std::setw(10) << result.num_samples
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.time_walk
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.ess_min
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.ess_avg
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.psrf_max << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl; 
}

// Helper function to run a single benchmark case
void run_benchmark_case(std::vector<BenchmarkResults>& all_results, 
                       unsigned int dim, 
                       unsigned int num_relations,
                       unsigned int num_samples,
                       const std::string& test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    
    // Create dense polytope first with fixed seed
    DenseHPOLYTOPE P_dense = random_orderpoly<DenseHPOLYTOPE, NT>(dim, num_relations, FIXED_SEED);
    P_dense.ComputeInnerBall();
    
    // Convert to sparse polytope
    SparseHPOLYTOPE P_sparse(P_dense.dimension(), P_dense.get_mat().sparseView(), P_dense.get_vec());
    P_sparse.ComputeInnerBall();
    
    unsigned int walk_length = 3 * dim;
    
    auto regular_result = benchmark_regular_billiard_walk(P_dense, num_samples, walk_length);
    auto sparse_result = benchmark_sparse_billiard_walk(P_sparse, num_samples, walk_length);
    
    all_results.push_back(regular_result);
    all_results.push_back(sparse_result);
}

void run_comprehensive_benchmark() {
    std::vector<BenchmarkResults> all_results;
    unsigned int num_samples = 5000;
    
    // Test 1: Small order polytope (10D) with sparse relations
    run_benchmark_case(all_results, 10, 25, num_samples, "Test 1: 10D Order Polytope (Sparse)");
    
    // Test 2: Medium order polytope (15D) with medium density
    run_benchmark_case(all_results, 15, 45, num_samples, "Test 2: 15D Order Polytope (Medium)");
    
    // Test 3: Large order polytope (20D) with dense relations
    run_benchmark_case(all_results, 20, 80, num_samples, "Test 3: 20D Order Polytope (Dense)");
    
    // Test 4: High-dimensional polytope (30D) with sparse relations
    run_benchmark_case(all_results, 30, 100, num_samples, "Test 4: 30D Order Polytope (Sparse)");
    
    // Test 5: Very high-dimensional polytope (40D) with medium density
    run_benchmark_case(all_results, 40, 150, num_samples, "Test 5: 40D Order Polytope (Medium)");
    
    // Test 6: Ultra high-dimensional polytope (50D) with sparse relations
    run_benchmark_case(all_results, 50, 200, num_samples, "Test 6: 50D Order Polytope (Sparse)");
    
    print_results(all_results);
}

int main() {
    std::cout << "Sparse Billiard Walk Benchmark" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << "This benchmark compares regular vs sparse billiard walk performance" << std::endl;
    std::cout << "using Effective Sample Size (ESS) and PSRF metrics." << std::endl;
    try {
        run_comprehensive_benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const char* msg) {
        std::cerr << "Error: " << msg << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
    return 0;
} 