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

struct BenchmarkResults {
    NT ess_min;
    NT ess_avg;
    NT psrf_max;
    double time_walk;       // only track walk time
    std::string walk_type;
    int dimension;
    int num_samples;
    std::string polytope_description;
};

// Function to run benchmark with regular billiard walk
BenchmarkResults benchmark_regular_billiard_walk(DenseHPOLYTOPE& P, unsigned int num_samples, unsigned int walk_length, const std::string& polytope_description) {
    
    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;
    
    RNGType rng(P.dimension());
    rng.set_seed(FIXED_SEED);  // Set fixed seed
    
    // Compute analytic center
    auto [_, x_ac_vec, converged] = barrier_center_ellipsoid_linear_ineq<MT, EllipsoidType::LOG_BARRIER, NT>(P.get_mat(), P.get_vec());
    if (!converged) {
        throw std::runtime_error("Failed to compute analytic center");
    }
    Point starting_point(x_ac_vec);
 
    
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
    results.walk_type = " Regular";
    results.dimension = P.dimension();
    results.num_samples = num_samples;
    results.polytope_description = polytope_description;
    
    return results;
}

// Function to run benchmark with sparse billiard walk
BenchmarkResults benchmark_sparse_billiard_walk(SparseHPOLYTOPE& P, unsigned int num_samples, unsigned int walk_length, const std::string& polytope_description) {
    
    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;
    
    RNGType rng(P.dimension());
    rng.set_seed(FIXED_SEED);  // Set fixed seed
    
    // Use lpsolve directly for inner ball computation to ensure consistency
    Point starting_point = P.ComputeInnerBall().first;
    
    // Compute analytic center
    auto [Hessian, x_ac_vec, converged] = barrier_center_ellipsoid_linear_ineq<MT, EllipsoidType::LOG_BARRIER, NT>(P.get_mat(), P.get_vec());
    if (!converged) {
        throw std::runtime_error("Failed to compute analytic center");
    }
    Point x_ac(x_ac_vec);
    
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
        results.walk_type = " Sparse";
        results.dimension = P.dimension();
        results.num_samples = successful_samples;
        results.polytope_description = polytope_description;
        
        return results;
    } catch (const std::exception& e) {
        std::cerr << "Error in sparse billiard walk: " << e.what() << std::endl;
        throw;
    }
}

void print_results(const std::vector<BenchmarkResults>& results) {
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    std::cout << std::left << std::setw(25) << "Polytope"
              << std::setw(15) << "Walk Type"
              << std::setw(8) << "Dim " 
              << std::setw(10) << "Samples "
              << std::setw(12) << "Time Walk(s) "
              << std::setw(12) << "Min ESS "
              << std::setw(12) << "Avg ESS "
              << std::setw(12) << "Max PSRF " << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(25) << result.polytope_description
                  << std::setw(15) << result.walk_type
                  << std::setw(8) << result.dimension
                  << std::setw(10) << result.num_samples
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.time_walk
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.ess_min
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.ess_avg
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.psrf_max << std::endl;
    }
    std::cout << std::string(100, '=') << std::endl; 
}

// Generic function to evaluate any polytope pair (dense and sparse versions)
std::vector<BenchmarkResults> evaluate_polytope_pair(const DenseHPOLYTOPE& P_dense_template, 
                          const SparseHPOLYTOPE& P_sparse_template,
                          const std::string& description,
                          unsigned int num_samples) {
    
    // Make copies so we don't modify the originals
    DenseHPOLYTOPE P_dense = P_dense_template;
    SparseHPOLYTOPE P_sparse = P_sparse_template;
    
    // Compute inner ball for both copies (this is required for proper functionality)
    P_dense.ComputeInnerBall();
    P_sparse.ComputeInnerBall();
    
    unsigned int walk_length = 3 * P_dense.dimension();
    
    std::vector<BenchmarkResults> current_results;
    
    try {
        auto regular_result = benchmark_regular_billiard_walk(P_dense, num_samples, walk_length, description);
        auto sparse_result = benchmark_sparse_billiard_walk(P_sparse, num_samples, walk_length, description);
        
        current_results.push_back(regular_result);
        current_results.push_back(sparse_result);
        
    } catch (const std::exception& e) {
        std::cerr << "Error evaluating " << description << ": " << e.what() << std::endl;
    }
    
    return current_results;
}

// Factory functions for different polytope types
std::pair<DenseHPOLYTOPE, SparseHPOLYTOPE> create_order_polytope(unsigned int dim, unsigned int num_relations) {
    // Create dense polytope first with fixed seed
    DenseHPOLYTOPE P_dense = random_orderpoly<DenseHPOLYTOPE, NT>(dim, num_relations, FIXED_SEED);
    
    // Convert to sparse polytope
    SparseHPOLYTOPE P_sparse(P_dense.dimension(), P_dense.get_mat().sparseView(), P_dense.get_vec());
    
    return std::make_pair(P_dense, P_sparse);
}

// -----------------------------------------------------------------------------
//  Extra helper: load a .txt adjacency-matrix file from the le-counting repo
// -----------------------------------------------------------------------------
std::pair<DenseHPOLYTOPE,SparseHPOLYTOPE>
create_order_polytope_from_file(const std::string& path_to_instance)
{
    // 1. Slurp the file into a stringstream -------------------------------
    std::ifstream in(path_to_instance);
    if (!in.good())
        throw std::runtime_error("Cannot open instance file: " + path_to_instance);

    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();

    // 2. Parse the adjacency matrix  -> Poset ----------------------------
    Poset poset = read_poset_from_file_adj_matrix(buffer).second;

    // 3. Build the dense H-polytope  -------------------------------------
    DenseHPOLYTOPE P_dense = get_orderpoly<DenseHPOLYTOPE>(poset);

    // 4. Sparse mirror  ---------------------------------------------------
    SparseHPOLYTOPE P_sparse(P_dense.dimension(),
                             P_dense.get_mat().sparseView(),
                             P_dense.get_vec());

    return {P_dense, P_sparse};
}

// Corrected function - replace your existing create_mittlemann_polytope function
std::pair<DenseHPOLYTOPE, SparseHPOLYTOPE> 
create_mittlemann_polytope(const std::string& mtx_file, 
                          const std::string& row_lower_file,
                          const std::string& row_upper_file) {
    
    // Load matrix from Matrix Market file
    Eigen::SparseMatrix<NT> A_sparse;
    if (!Eigen::loadMarket(A_sparse, mtx_file)) {
        throw std::runtime_error("Failed to load matrix from " + mtx_file);
    }
    
    // Load row bounds
    std::vector<NT> row_lower_vec, row_upper_vec;
    std::ifstream file_lower(row_lower_file, std::ios::binary);
    std::ifstream file_upper(row_upper_file, std::ios::binary);
    
    if (!file_lower || !file_upper) {
        throw std::runtime_error("Failed to load bound files");
    }
    
    // Read numpy files (simple binary format)
    // Skip numpy header and read data
    file_lower.seekg(80); // Skip numpy header
    file_upper.seekg(80);
    
    row_lower_vec.resize(A_sparse.rows());
    row_upper_vec.resize(A_sparse.rows());
    
    file_lower.read(reinterpret_cast<char*>(row_lower_vec.data()), 
                   A_sparse.rows() * sizeof(NT));
    file_upper.read(reinterpret_cast<char*>(row_upper_vec.data()), 
                   A_sparse.rows() * sizeof(NT));
    
    // Convert to Eigen vectors
    VT row_lower = Eigen::Map<VT>(row_lower_vec.data(), A_sparse.rows());
    VT row_upper = Eigen::Map<VT>(row_upper_vec.data(), A_sparse.rows());
    
    // Create constraint matrix and RHS for Ax <= b format
    std::vector<Eigen::Triplet<NT>> triplets;
    std::vector<NT> rhs_vec;
    
    int constraint_count = 0;
    
    // Add upper bound constraints: Ax <= row_upper
    for (int i = 0; i < A_sparse.rows(); ++i) {
        if (std::isfinite(row_upper[i])) {
            // Correct way to iterate over row i in CSC matrix
            for (int k = 0; k < A_sparse.outerSize(); ++k) {
                for (Eigen::SparseMatrix<NT>::InnerIterator it(A_sparse, k); it; ++it) {
                    if (it.row() == i) {
                        triplets.push_back(Eigen::Triplet<NT>(constraint_count, it.col(), it.value()));
                    }
                }
            }
            rhs_vec.push_back(row_upper[i]);
            constraint_count++;
        }
    }
    
    // Add lower bound constraints: -Ax <= -row_lower
    for (int i = 0; i < A_sparse.rows(); ++i) {
        if (std::isfinite(row_lower[i])) {
            // Correct way to iterate over row i in CSC matrix
            for (int k = 0; k < A_sparse.outerSize(); ++k) {
                for (Eigen::SparseMatrix<NT>::InnerIterator it(A_sparse, k); it; ++it) {
                    if (it.row() == i) {
                        triplets.push_back(Eigen::Triplet<NT>(constraint_count, it.col(), -it.value()));
                    }
                }
            }
            rhs_vec.push_back(-row_lower[i]);
            constraint_count++;
        }
    }
    
    // Build final constraint matrix
    Eigen::SparseMatrix<NT> A_final(constraint_count, A_sparse.cols());
    A_final.setFromTriplets(triplets.begin(), triplets.end());
    
    VT b_final = Eigen::Map<VT>(rhs_vec.data(), constraint_count);
    
    // Create dense version
    MT A_dense = A_final;
    DenseHPOLYTOPE P_dense(A_sparse.cols(), A_dense, b_final);
    
    // Create sparse version
    SparseHPOLYTOPE P_sparse(A_sparse.cols(), A_final, b_final);
    
    return std::make_pair(P_dense, P_sparse);
}

void run_comprehensive_benchmark() {
    unsigned int num_samples = 5000;
    std::vector<BenchmarkResults> all_results;

    std::cout << "Running benchmarks..." << std::endl;

    //-------------------------
    // Order Polytopes 
    //-------------------------
    
    // Test 1: Small order polytope (10D) with sparse relations
    auto [P1_dense, P1_sparse] = create_order_polytope(10, 25);
    auto results1 = evaluate_polytope_pair(P1_dense, P1_sparse, "10D Order Polytope", num_samples);
    all_results.insert(all_results.end(), results1.begin(), results1.end());
    
    // Test 2: Medium order polytope (15D) with medium density
    auto [P2_dense, P2_sparse] = create_order_polytope(15, 45);
    auto results2 = evaluate_polytope_pair(P2_dense, P2_sparse, "15D Order Polytope", num_samples);
    all_results.insert(all_results.end(), results2.begin(), results2.end());
    
    // Test 3: Large order polytope (20D) with dense relations
    auto [P3_dense, P3_sparse] = create_order_polytope(20, 80);
    auto results3 = evaluate_polytope_pair(P3_dense, P3_sparse, "20D Order Polytope", num_samples);
    all_results.insert(all_results.end(), results3.begin(), results3.end());
    
    // Test 4: High-dimensional polytope (30D) with sparse relations
    auto [P4_dense, P4_sparse] = create_order_polytope(30, 100);
    auto results4 = evaluate_polytope_pair(P4_dense, P4_sparse, "30D Order Polytope", num_samples);
    all_results.insert(all_results.end(), results4.begin(), results4.end());
    
    // Test 5: Very high-dimensional polytope (40D) with medium density
    auto [P5_dense, P5_sparse] = create_order_polytope(40, 150);
    auto results5 = evaluate_polytope_pair(P5_dense, P5_sparse, "40D Order Polytope", num_samples);
    all_results.insert(all_results.end(), results5.begin(), results5.end());
    
    // Test 6: Ultra high-dimensional polytope (50D) with sparse relations
    auto [P6_dense, P6_sparse] = create_order_polytope(50, 200);
    auto results6 = evaluate_polytope_pair(P6_dense, P6_sparse, "50D Order Polytope", num_samples);
    all_results.insert(all_results.end(), results6.begin(), results6.end());

    //-------------------------
    //  Mittlemann Polytopes
    //---------------------------
    try {
        auto [P_rococo_dense, P_rococo_sparse] = create_mittlemann_polytope(
            "../mittlemann/rococoC10-001000/rococoC10-001000.mtx",
            "../mittlemann/rococoC10-001000/row_lower.npy", 
            "../mittlemann/rococoC10-001000/row_upper.npy"
        );
        std::cout << "Created mittlemann rococoC10, continuing to evaluation..." << std::endl;
        auto results_rococo = evaluate_polytope_pair(P_rococo_dense, P_rococo_sparse, 
                                                "Mittlemann rococoC10", num_samples);
        all_results.insert(all_results.end(), results_rococo.begin(), results_rococo.end());
    } catch (const std::exception& e) {
        std::cerr << "Skipping Mittlemann polytope: " << e.what() << std::endl;
    }

    // -----------------------------------------------------------------------------
    //  Real posets from the le-counting-practice data set
    // -----------------------------------------------------------------------------
    std::vector<std::string> le_instances = {
        "../le-counting-practice/instances/bipartite_0.5_032_0.txt",
        //"../le-counting-practice/bayesiannetwork_andes_128_0.txt"
    };

    for (const std::string& fname : le_instances)
    {
        try {
            auto [Pi_dense, Pi_sparse] = create_order_polytope_from_file(fname);
            std::cout << "Created polytope from file " << fname << ", continuing to evaluation..." << std::endl;
            auto Ri   = evaluate_polytope_pair(Pi_dense, Pi_sparse, "LE-poset: "+fname,
                                            num_samples);
            all_results.insert(all_results.end(), Ri.begin(), Ri.end());
        }
        catch (const std::exception& e) {
            std::cerr << "Skipping " << fname << "  (" << e.what() << ")\n";
        }
    }


    // Print final combined results table
    print_results(all_results);

    //-------------------------
    // Future: Birkhoff Polytopes (example of how easy it will be to add)
    //-------------------------
    // auto [B1_dense, B1_sparse] = create_birkhoff_polytope(4);
    // auto resultsB1 = evaluate_polytope_pair(B1_dense, B1_sparse, "4x4 Birkhoff Polytope", num_samples);
    // all_results.insert(all_results.end(), resultsB1.begin(), resultsB1.end());
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