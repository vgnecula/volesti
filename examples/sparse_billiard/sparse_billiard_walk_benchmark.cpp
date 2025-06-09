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
#include "preprocess/barrier_center_ellipsoid.hpp"
#include "preprocess/svd_rounding.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;
typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;
typedef BoostRandomNumberGenerator<boost::mt19937, NT> RNGType;

// Define both dense and sparse polytope types
typedef HPolytope<Point> DenseHPOLYTOPE;
typedef HPolytope<Point, Eigen::SparseMatrix<NT, Eigen::RowMajor>> SparseHPOLYTOPE;
typedef BilliardWalk::template Walk<DenseHPOLYTOPE, RNGType> DenseBilliardWalkType;
typedef BilliardWalk::template Walk<SparseHPOLYTOPE, RNGType> SparseBilliardWalkType;

PushBackWalkPolicy push_back_policy;

const unsigned int FIXED_SEED = 42;  // Fixed seed for reproducibility

struct BenchmarkResults {
    NT ess_min;
    NT ess_avg;
    NT psrf_max;
    double time_walk;       // only track walk time
    std::string walk_type;
    int dimension;
    int num_samples;
};

// =============================================================================
// STEP 1: BENCHMARK STRUCTURE FOR DENSE ROUNDED BILLIARD WALK
// =============================================================================
BenchmarkResults benchmark_dense_rounded_billiard_walk(DenseHPOLYTOPE& P, unsigned int num_samples, unsigned int walk_length) {
    std::cout << "Benchmarking Dense Rounded Billiard Walk..." << std::endl;

    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;

    RNGType rng(P.dimension());
    rng.set_seed(FIXED_SEED);

    // Step 1: Compute analytic center and Hessian using the established barrier center function
    auto [H, x_ac_vec, converged] = barrier_center_ellipsoid_linear_ineq<MT, EllipsoidType::LOG_BARRIER, NT>(P.get_mat(), P.get_vec());
    if (!converged) throw std::runtime_error("Failed to compute analytic center");
    Point x_ac(x_ac_vec);

    // Step 2: Create copy of polytope and shift to analytic center
    DenseHPOLYTOPE P_shifted = P;
    P_shifted.shift(-x_ac.getCoefficients());

    // Step 3: Compute the rounding transformation using Cholesky decomposition of the Hessian
    // This follows the same pattern as other rounding methods in volesti
    Eigen::LLT<MT> llt(H);
    if (llt.info() != Eigen::Success) throw std::runtime_error("LLT decomposition failed");
    
    // Get L such that H = L * L^T, then use L^(-1) for the transformation
    MT L = llt.matrixL();
    MT L_inv = L.triangularView<Eigen::Lower>().solve(MT::Identity(P.dimension(), P.dimension()));
    
    // Step 4: Apply the linear transformation to round the polytope: A_rounded = A * L^(-1)
    MT A_rounded = P_shifted.get_mat() * L_inv;
    DenseHPOLYTOPE P_rounded(P.dimension(), A_rounded, P_shifted.get_vec());

    // Step 5: Find a feasible starting point in the rounded polytope
    Point origin(P.dimension());
    origin.set_to_origin();
    if (!P_rounded.is_in(origin)) {
        origin = P_rounded.ComputeInnerBall().first;
    }

    auto t1 = clock::now();

    // Step 6: Perform walk on the rounded polytope
    std::vector<Point> randPoints;
    typedef RandomPointGenerator<DenseBilliardWalkType> Generator;
    Generator::apply(P_rounded, origin, num_samples, walk_length, randPoints, push_back_policy, rng);

    auto t2 = clock::now();

    // Step 7: Back-transform samples to original space: x_original = L_inv * x_rounded + x_ac
    for (auto& p : randPoints) {
        VT y = p.getCoefficients();          // sample in rounded space
        VT x_original = L_inv * y + x_ac.getCoefficients();   // <-- use L_inv!
        p = Point(x_original);
    } 

    // Step 8: Compute diagnostics
    MT samples(P.dimension(), num_samples);
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
    results.walk_type = "Dense Rounded Billiard";
    results.dimension = P.dimension();
    results.num_samples = num_samples;

    return results;
}


// =============================================================================
// STEP 2: BENCHMARK STRUCTURE FOR SPARSE BILLIARD WALK  
// =============================================================================
BenchmarkResults benchmark_sparse_billiard_walk(SparseHPOLYTOPE& P, unsigned int num_samples, unsigned int walk_length) {
    std::cout << "Benchmarking Sparse Billiard Walk..." << std::endl;
    
    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;
    
    RNGType rng(P.dimension());
    rng.set_seed(FIXED_SEED);
    
    // =============================================================================
    // DEBUG: BARRIER CENTER COMPUTATION
    // =============================================================================
    std::cout << "\n=== BARRIER CENTER DEBUG ===" << std::endl;
    
    // Check input polytope properties
    std::cout << "Input polytope properties:" << std::endl;
    std::cout << "  Dimension: " << P.dimension() << std::endl;
    std::cout << "  Num hyperplanes: " << P.num_of_hyperplanes() << std::endl;
    
    // Get the constraint matrix and vector
    auto A_matrix = P.get_mat();
    auto b_vector = P.get_vec();
    
    std::cout << "  A matrix type: " << typeid(A_matrix).name() << std::endl;
    std::cout << "  A matrix size: " << A_matrix.rows() << "x" << A_matrix.cols() << std::endl;
    std::cout << "  A matrix nnz: " << A_matrix.nonZeros() << std::endl;
    std::cout << "  b vector size: " << b_vector.size() << std::endl;
    std::cout << "  b vector range: [" << b_vector.minCoeff() << ", " << b_vector.maxCoeff() << "]" << std::endl;
    
    // Check if polytope is bounded and has interior
    std::cout << "\nPolytope validation:" << std::endl;
    Point test_origin(P.dimension());
    test_origin.set_to_origin();
    std::cout << "  Origin feasible: " << (P.is_in(test_origin) ? "YES" : "NO") << std::endl;
    
    // Check constraint satisfaction at origin
    VT Ax = A_matrix * test_origin.getCoefficients();
    VT slack = b_vector - Ax;
    std::cout << "  Constraint slack at origin: [" << slack.minCoeff() << ", " << slack.maxCoeff() << "]" << std::endl;
    int violated_at_origin = (slack.array() <= 0).count();
    std::cout << "  Violated constraints at origin: " << violated_at_origin << " / " << slack.size() << std::endl;
    
    // =============================================================================
    // CALL BARRIER CENTER FUNCTION WITH DEBUGGING
    // =============================================================================
    std::cout << "\nCalling barrier_center_ellipsoid_linear_ineq..." << std::endl;
    
    MT Hessian;
    VT x_ac_vec;
    bool converged;
    
    try {
        
        // Step 1: Convert sparse matrix to dense for barrier center computation
        MT A_dense = MT(A_matrix);  // Convert sparse to dense
        VT b_dense = b_vector;      // b is already dense

        // Step 2: Call barrier center function with dense matrix
        auto result = barrier_center_ellipsoid_linear_ineq<MT, EllipsoidType::LOG_BARRIER, NT>(A_dense, b_dense);

        Hessian = std::get<0>(result);
        x_ac_vec = std::get<1>(result);
        converged = std::get<2>(result);
        
        std::cout << "Barrier center computation result:" << std::endl;
        std::cout << "  Converged: " << (converged ? "YES" : "NO") << std::endl;
        std::cout << "  Hessian type: " << typeid(Hessian).name() << std::endl;
        std::cout << "  Hessian size: " << Hessian.rows() << "x" << Hessian.cols() << std::endl;
        std::cout << "  Hessian is dense matrix" << std::endl;
        std::cout << "  Analytic center: " << x_ac_vec.transpose() << std::endl;
        
        // Check if analytic center is actually feasible
        Point x_ac(x_ac_vec);
        std::cout << "  Analytic center feasible: " << (P.is_in(x_ac) ? "YES" : "NO") << std::endl;
        
        // Check constraint satisfaction at analytic center
        VT Ax_ac = A_matrix * x_ac_vec;
        VT slack_ac = b_vector - Ax_ac;
        std::cout << "  Constraint slack at analytic center: [" << slack_ac.minCoeff() << ", " << slack_ac.maxCoeff() << "]" << std::endl;
        int violated_at_ac = (slack_ac.array() <= 1e-10).count();
        std::cout << "  Violated constraints at analytic center: " << violated_at_ac << " / " << slack_ac.size() << std::endl;
        
        // Check Hessian properties
        Eigen::VectorXd eigenvals = Eigen::SelfAdjointEigenSolver<MT>(Hessian).eigenvalues();
        std::cout << "  Hessian eigenvalues: [" << eigenvals.minCoeff() << ", " << eigenvals.maxCoeff() << "]" << std::endl;
        std::cout << "  Hessian condition number: " << (eigenvals.maxCoeff() / eigenvals.minCoeff()) << std::endl;
        std::cout << "  Hessian positive definite: " << (eigenvals.minCoeff() > 1e-12 ? "YES" : "NO") << std::endl;
        
        if (!converged) {
            std::cout << "ERROR: Barrier center computation did not converge!" << std::endl;
            throw std::runtime_error("Failed to compute analytic center");
        }
        
        if (eigenvals.minCoeff() <= 1e-12) {
            std::cout << "ERROR: Hessian is not positive definite!" << std::endl;
            std::cout << "This suggests the barrier method failed or the polytope has issues." << std::endl;
            throw std::runtime_error("Hessian is not positive definite");
        }
        
        std::cout << "=== END BARRIER CENTER DEBUG ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION in barrier_center_ellipsoid_linear_ineq: " << e.what() << std::endl;
        throw;
    }
    
    Point x_ac(x_ac_vec);
    
    // Step 2: Shift polytope to analytic center (keep sparse)
    SparseHPOLYTOPE P_shifted = P;
    P_shifted.shift(-x_ac.getCoefficients());
    
    // Step 3: Convert Hessian to sparse format
    Eigen::SparseMatrix<NT, Eigen::ColMajor> H_sparse;
    if constexpr (std::is_same_v<decltype(Hessian), Eigen::SparseMatrix<NT, Eigen::ColMajor>>) {
        H_sparse = Hessian;
    } else {
        H_sparse = Hessian.sparseView();
    }
    
    // Step 4: Create NEW sparse billiard walk with Hessian
    Point origin(P.dimension());
    origin.set_to_origin();

    if ( !P_shifted.is_in(origin) ) {
        origin = P_shifted.ComputeInnerBall().first;
        std::cout << "Computed inner ball" << std::endl;
    } else {
        std::cout << "WARNING: Origin is not in the polytope" << std::endl;
        return BenchmarkResults();
    }
    
    // Fix: Create parameters object with proper walk length
    typedef SparseBilliardWalk::template Walk<SparseHPOLYTOPE, RNGType> SparseBilliardWalkType;
    SparseBilliardWalkType::parameters parms(walk_length, true);
    SparseBilliardWalkType walk(P_shifted, origin, rng, parms, H_sparse);

    auto t1 = clock::now();

    std::vector<Point> randPoints;
    randPoints.reserve(num_samples);

    Point current_point = origin;
    unsigned int successful_samples = 0;

    for (unsigned int i = 0; i < num_samples; ++i) {
        try {
            walk.apply(P_shifted, current_point, walk_length, rng);
            // Transform back to original space
            randPoints.push_back(Point(current_point.getCoefficients() + x_ac.getCoefficients()));
            successful_samples++;
        } catch (const std::exception& e) {
            std::cerr << "Error during walk step " << i << ": " << e.what() << std::endl;
            continue;
        }
    } 
    
    auto t2 = clock::now();
    
    if (successful_samples == 0) {
        throw std::runtime_error("No successful samples were generated");
    }
    
    // Compute diagnostics
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

// =============================================================================
// STEP 3: UPDATED BENCHMARK RUNNER
// =============================================================================

void run_benchmark_case(std::vector<BenchmarkResults>& all_results, 
                       unsigned int dim, 
                       unsigned int num_relations,
                       unsigned int num_samples,
                       const std::string& test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    
    // Create dense polytope first with fixed seed
    DenseHPOLYTOPE P_dense = random_orderpoly<DenseHPOLYTOPE, NT>(dim, num_relations, FIXED_SEED);
    P_dense.ComputeInnerBall();
    
    // Convert to sparse polytope (same mathematical polytope)
    SparseHPOLYTOPE P_sparse(P_dense.dimension(), P_dense.get_mat().sparseView(), P_dense.get_vec());
    P_sparse.ComputeInnerBall();
    
    unsigned int walk_length = 3 * dim;
    
    // Compare dense rounded vs sparse (with lazy rounding)
    auto dense_rounded_result = benchmark_dense_rounded_billiard_walk(P_dense, num_samples, walk_length);
    auto sparse_result = benchmark_sparse_billiard_walk(P_sparse, num_samples, walk_length);
    
    all_results.push_back(dense_rounded_result);
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
    
    /*
    // Test 4: High-dimensional polytope (30D) with sparse relations
    run_benchmark_case(all_results, 30, 100, num_samples, "Test 4: 30D Order Polytope (Sparse)");
    
    // Test 5: Very high-dimensional polytope (40D) with medium density
    run_benchmark_case(all_results, 40, 150, num_samples, "Test 5: 40D Order Polytope (Medium)");
    
    // Test 6: Ultra high-dimensional polytope (50D) with sparse relations
    run_benchmark_case(all_results, 50, 200, num_samples, "Test 6: 50D Order Polytope (Sparse)");
    */
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