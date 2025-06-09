// VolEsti (volume computation and sampling library)
// Sparse Billiard Walk for uniform distribution with lazy rounding

#ifndef RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP
#define RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "convex_bodies/hpolytope.h"
#include "sampling/sphere.hpp"
#include "generators/boost_random_number_generator.hpp"

// Sparse Billiard walk for uniform distribution with lazy rounding
struct SparseBilliardWalk
{

template
<
    typename Polytope,
    typename RandomNumberGenerator
>
struct Walk
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef Eigen::SparseMatrix<NT, Eigen::ColMajor> SparseMT;
    typedef Eigen::SparseMatrix<NT, Eigen::RowMajor> SparseRowMT;
    typedef typename Point::Coeff VT;
    typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;

    struct parameters
    {
        parameters(double L = 0, bool set = false)
            : m_L(L), set_L(set),
            inner_vi_ak(0), facet_prev(-1) {}

        double m_L;   // requested billiard length
        bool   set_L; // whether the user fixed it

        // --- per-step scratch fields -------------
        NT  inner_vi_ak; // ⟨v,a_k⟩ / ‖a_k L‖  of the facet we just hit
        int  facet_prev;  // index of that facet
    }; 

    // Constructor with Hessian matrix (core requirement from TolisChal)
    // From: "I would implement a new billiard walk struct that would take as input 
    //        a hessian matrix H and an unrounded polytope with a sparse matrix _A."
    template <typename GenericPolytope>
    Walk(GenericPolytope &P, 
         Point const& p, 
         RandomNumberGenerator &rng,
         parameters const& user_params,
         SparseMT const& Hessian) : _param(user_params)
    {

        _Len = _param.set_L ? _param.m_L : 2.0 * std::sqrt(static_cast<double>(P.dimension()));
 
        // From: "We don't apply the transformation on the polytope to preserve the sparsity of A"
        // Store original sparse A and b (never transform them)
        _A = P.get_mat();
        _b = P.get_vec();
        
        // Compute L_inv from Hessian as TolisChal specified
        compute_cholesky_factor(Hessian);
        
        initialize(P, p, rng);
    }

private:
    
    void compute_cholesky_factor(const SparseMT &H)
    {
        std::cout << "=== CORRECTED CHOLESKY SETUP ===" << std::endl;
        std::cout << "Hessian size: " << H.rows() << "x" << H.cols() << std::endl;
        std::cout << "Hessian nnz: " << H.nonZeros() << std::endl;
        
        // CRITICAL FIX 1: Follow the comment exactly
        // "The Hessian can be used to round the polytope by computing the cholesky L of H^{-1}"
        // "Then, the cholesky L_inv of H will also be sparse and triangular"
        
        Eigen::MatrixXd H_dense = H.toDense();
        std::cout << "Computing H^(-1)..." << std::endl;
        Eigen::MatrixXd H_inv = H_dense.inverse();
        
        // Cholesky of H^(-1): H^(-1) = L * L^T
        std::cout << "Computing Cholesky of H^(-1)..." << std::endl;
        Eigen::LLT<Eigen::MatrixXd> llt_Hinv(H_inv);
        if (llt_Hinv.info() != Eigen::Success)
            throw std::runtime_error("Cholesky decomposition of H^(-1) failed");
        
        Eigen::MatrixXd L = llt_Hinv.matrixL();  // L where H^(-1) = L * L^T
        _L = L.sparseView();  // Store L (lower triangular)
        
        std::cout << "L (from H^(-1) = L*L^T) nnz: " << _L.nonZeros() << std::endl;
        
        // CRITICAL FIX 2: Verify the transformation works
        // Check: L^T * H * L should equal I (identity)
        Eigen::MatrixXd verification = L.transpose() * H_dense * L;
        Eigen::MatrixXd error = verification - Eigen::MatrixXd::Identity(H.rows(), H.cols());
        std::cout << "Verification ||L^T * H * L - I||: " << error.norm() << std::endl;
        
        if (error.norm() > 1e-10) {
            std::cout << "ERROR: Transformation verification failed!" << std::endl;
            std::cout << "The rounding transformation is incorrect." << std::endl;
            throw std::runtime_error("Invalid rounding transformation");
        }
        
        // Store b without modification for now
        _b_scaled = _b;
        
        std::cout << "=== END CORRECTED CHOLESKY SETUP ===" << std::endl;
    }

    // CRITICAL FIX 3: Implement TRUE lazy boundary oracle as described in comment
    // "instead of doing Ax = A_rounded*x we do Ax = A * L_inv.template triangularView<Eigen::Upper>().solve(x)"
    std::pair<NT, int>
    line_positive_intersect(Point const& r, Point const& v,
                            VT& Ar_out, VT& Av_out)
    {
        static int debug_call_count = 0;
        debug_call_count++;
        bool debug_print = (debug_call_count <= 3);
        
        if (debug_print) {
            std::cout << "\n=== CORRECTED INTERSECTION (call " << debug_call_count << ") ===" << std::endl;
        }

        // Step 1: Apply coordinate transformation
        // CRITICAL FIX: We need x_rounded = L * x_original (not L^(-1) * x_original)
        // This is because in the rounded space, the constraint is (A * L^(-1)) * x_rounded <= b
        // So if x_original satisfies A * x_original <= b, then x_rounded = L * x_original
        // satisfies (A * L^(-1)) * x_rounded = A * L^(-1) * L * x_original = A * x_original <= b
        
        VT r_original = r.getCoefficients();
        VT v_original = v.getCoefficients();
        
        // Apply L transformation (not L^(-1))
        VT r_rounded = _L * r_original;
        VT v_rounded = _L * v_original;
        
        if (debug_print) {
            std::cout << "Lazy coordinate transformation:" << std::endl;
            std::cout << "  ||r_original||: " << r_original.norm() << std::endl;
            std::cout << "  ||r_rounded||:  " << r_rounded.norm() << std::endl;
            std::cout << "  Transformation ratio: " << r_rounded.norm() / r_original.norm() << std::endl;
        }

        // Step 2: Apply constraints in original space to original coordinates
        // This is the key insight: we evaluate A * x_original, not A * x_rounded
        Ar_out = _A * r_original;
        Av_out = _A * v_original;

        if (debug_print) {
            // Verify feasibility
            VT slack = _b - Ar_out;
            int violated = (slack.array() < -1e-10).count();
            std::cout << "Feasibility check: violated=" << violated << "/" << slack.size() << std::endl;
            std::cout << "Slack range: [" << slack.minCoeff() << ", " << slack.maxCoeff() << "]" << std::endl;
        }

        // Step 3: Find intersection λ such that A * (r + λ*v) = b
        NT lambda_min = std::numeric_limits<NT>::max();
        int facet = -1;
        int positive_lambdas = 0;
        
        for (int i = 0; i < Av_out.size(); ++i)
        {
            NT av = Av_out(i);
            if (std::abs(av) < NT(1e-12)) continue;

            NT lambda = (_b(i) - Ar_out(i)) / av;
            
            if (debug_print && i < 5) {
                std::cout << "Constraint " << i << ": lambda=" << lambda << " (av=" << av << ")" << std::endl;
            }

            if (lambda > NT(1e-12)) {
                positive_lambdas++;
                if (lambda < lambda_min) {
                    lambda_min = lambda;
                    facet = i;
                    // Store the inner product in the rounded space for reflection
                    // We need <v_rounded, a_rounded> where a_rounded = L^T * a_i
                    VT a_i(_A.cols());
                    for (int j = 0; j < _A.cols(); ++j) {
                        a_i(j) = _A.coeff(i, j);
                    }
                    VT a_rounded = _L.transpose() * a_i;
                    _param.inner_vi_ak = v_rounded.dot(a_rounded) / a_rounded.squaredNorm();
                    _param.facet_prev = i;
                }
            }
        }

        if (debug_print) {
            std::cout << "Found " << positive_lambdas << " positive intersections" << std::endl;
            std::cout << "Best: lambda=" << lambda_min << ", facet=" << facet << std::endl;
            std::cout << "=== END CORRECTED INTERSECTION ===" << std::endl;
        }

        if (facet == -1) {
            lambda_min = std::numeric_limits<NT>::max();
        }

        if (debug_print && facet != -1) {
            std::cout << "\nMetric analysis for chosen facet:" << std::endl;
            
            // Get the normal vector
            VT a_i(_A.cols());
            for (int j = 0; j < _A.cols(); ++j) {
                a_i(j) = _A.coeff(facet, j);
            }
            
            // Compute various norms
            VT a_rounded = _L.transpose() * a_i;
            VT v_vec = v.getCoefficients();
            VT v_rounded = _L * v_vec;
            
            std::cout << "  ||a_original||: " << a_i.norm() << std::endl;
            std::cout << "  ||a_rounded||: " << a_rounded.norm() << std::endl;
            std::cout << "  ||v_original||: " << v_vec.norm() << std::endl;
            std::cout << "  ||v_rounded||: " << v_rounded.norm() << std::endl;
            
            // Step size in different metrics
            VT step_original = lambda_min * v_vec;
            VT step_rounded = lambda_min * v_rounded;
            std::cout << "  Step size (original): " << step_original.norm() << std::endl;
            std::cout << "  Step size (rounded): " << step_rounded.norm() << std::endl;
            
            // Distance to boundary in Hessian metric
            // d_H = sqrt(v^T H v) * lambda
            // Since H^(-1) = L L^T, we have ||v||_H = ||L^(-1) v||
            VT Linv_v = _L.template triangularView<Eigen::Lower>().solve(v_vec); 
            NT v_H_norm = Linv_v.norm();
            std::cout << "  Distance in H-metric: " << lambda_min * v_H_norm << std::endl;
        }

        return {lambda_min, facet};
    } 

    // Replace the existing compute_reflection with this debug version:
    void compute_reflection(Point& v, Point& u)
    {
        static int reflection_count = 0;
        reflection_count++;
        bool debug = (reflection_count <= 5);
        
        int facet = _param.facet_prev;
        if (facet < 0 || facet >= _A.rows()) {
            std::cout << "WARNING: Invalid facet " << facet << std::endl;
            return;
        }
        
        if (debug) {
            std::cout << "\n=== REFLECTION DEBUG " << reflection_count << " ===" << std::endl;
            std::cout << "Facet: " << facet << std::endl;
        }
        
        // Get the constraint normal from original A
        VT a_original(_A.cols());
        for (int j = 0; j < _A.cols(); ++j) {
            a_original(j) = _A.coeff(facet, j);
        }
        
        // Current velocity
        VT v_original = v.getCoefficients();
        VT u_original = u.getCoefficients();
        
        // METHOD 1: Reflection in original space
        NT v_dot_a_orig = v_original.dot(a_original) / a_original.squaredNorm();
        VT v_reflected_orig = v_original - 2.0 * v_dot_a_orig * a_original;
        
        // METHOD 2: Reflection in rounded space
        VT v_rounded = _L * v_original;
        VT a_rounded = _L.transpose() * a_original;
        NT v_dot_a_round = v_rounded.dot(a_rounded) / a_rounded.squaredNorm();
        VT v_reflected_round = v_rounded - 2.0 * v_dot_a_round * a_rounded;
        
        // Transform back to original space
        VT v_reflected_back = _L.template triangularView<Eigen::Lower>().solve(v_reflected_round); 
        
        if (debug) {
            std::cout << "Original space:" << std::endl;
            std::cout << "  ||a_original||: " << a_original.norm() << std::endl;
            std::cout << "  <v,a>_orig: " << v_original.dot(a_original) << std::endl;
            std::cout << "  ||v_before||: " << v_original.norm() << std::endl;
            std::cout << "  ||v_after||: " << v_reflected_orig.norm() << std::endl;
            
            std::cout << "Rounded space:" << std::endl;
            std::cout << "  ||a_rounded||: " << a_rounded.norm() << std::endl;
            std::cout << "  <v,a>_round: " << v_rounded.dot(a_rounded) << std::endl;
            std::cout << "  ||v_rounded_before||: " << v_rounded.norm() << std::endl;
            std::cout << "  ||v_rounded_after||: " << v_reflected_round.norm() << std::endl;
            
            std::cout << "Comparison:" << std::endl;
            std::cout << "  ||v_reflected_orig||: " << v_reflected_orig.norm() << std::endl;
            std::cout << "  ||v_reflected_back||: " << v_reflected_back.norm() << std::endl;
            std::cout << "  Difference: " << (v_reflected_orig - v_reflected_back).norm() << std::endl;
            
            // Check if reflection preserves feasibility
            Point p_test = _p + 0.01 * Point(v_reflected_back);
            VT Ap_test = _A * p_test.getCoefficients();
            VT slack_test = _b - Ap_test;
            std::cout << "  Feasibility after small step: " 
                    << (slack_test.minCoeff() > -1e-10 ? "OK" : "VIOLATED") << std::endl;
        }
        
        // Use the rounded space reflection (METHOD 2)
        v = Point(v_reflected_back);
        u = Point(u_original - 2.0 * (u_original.dot(a_original) / a_original.squaredNorm()) * a_original);
        
        if (debug) {
            std::cout << "=== END REFLECTION DEBUG ===" << std::endl;
        }
    }

public:

    // CRITICAL FIX 5: Enhanced walk with better error handling
    // Modified apply function with more debugging:
    template<typename GenericPolytope>
    inline void apply(GenericPolytope &P,
                    Point& p,
                    unsigned int const& walk_length,
                    RandomNumberGenerator &rng)
    {
        static int walk_call_count = 0;
        walk_call_count++;
        bool debug_walk = (walk_call_count <= 2);  // Debug first two walks
        
        if (debug_walk) {
            std::cout << "\n=== WALK " << walk_call_count << " START ===" << std::endl;
            std::cout << "Walk length: " << walk_length << std::endl;
            debug_walk_state("Initial state");
        }

        unsigned int n = P.dimension();
        const NT dl = 0.995;
        
        for (auto j = 0u; j < walk_length; ++j)
        {
            if (debug_walk && j < 3) {
                debug_walk_state("Step start", j);
            }
            
            NT T = rng.sample_urdist() * _Len;
            _v = GetDirection<Point>::apply(n, rng);

            Point p0 = _p;
            int it = 0;
            int reflections_this_step = 0;
            
            while (it < 50 * n)
            {
                auto pbpair = line_positive_intersect(_p, _v, _lambdas, _Av);
                NT lambda = pbpair.first;
                int facet = pbpair.second;

                if (facet < 0 || lambda <= 0 || lambda == std::numeric_limits<NT>::max()) {
                    _p = p0;
                    _v = GetDirection<Point>::apply(n, rng);
                    break;
                }

                if (T <= lambda) {
                    _p += (T * _v);
                    _lambda_prev = T;
                    
                    if (debug_walk && j < 3) {
                        std::cout << "  Free flight: T=" << T << ", lambda=" << lambda 
                                << ", reflections=" << reflections_this_step << std::endl;
                    }
                    break;
                }

                _lambda_prev = dl * lambda;
                _p += (_lambda_prev * _v);
                T -= _lambda_prev;

                compute_reflection(_v, _p);
                reflections_this_step++;
                it++;
            }
            
            if (debug_walk && j < 3) {
                debug_walk_state("Step end", j);
            }
        }
        
        if (debug_walk) {
            std::cout << "=== WALK " << walk_call_count << " END ===" << std::endl;
            
            // Check how far we've moved
            VT p_final = _p.getCoefficients();
            VT p_initial = p.getCoefficients();
            std::cout << "Total displacement: " << (p_final - p_initial).norm() << std::endl;
        }
        
        p = Point(_p.getCoefficients());
    }

private:

    template<typename GenericPolytope>
    inline void initialize(GenericPolytope &P,
                        Point const& p,
                        RandomNumberGenerator &rng)
    {
        // Store originals for debugging
        _original_A_dense = MT(_A);
        _original_b = _b;
        std::cout << "\n=== INITIALIZATION DEBUG ===" << std::endl;
        std::cout << "Polytope dimension: " << P.dimension() << std::endl;
        std::cout << "Polytope hyperplanes: " << P.num_of_hyperplanes() << std::endl;
        std::cout << "Starting point: " << p.getCoefficients().transpose() << std::endl;
        
        // Check if starting point satisfies constraints
        VT Ap = _A * p.getCoefficients();
        VT slack = _b - Ap;
        std::cout << "Constraint slack range: [" << slack.minCoeff() << ", " << slack.maxCoeff() << "]" << std::endl;
        int violated = (slack.array() < 0).count();
        std::cout << "Violated constraints: " << violated << " out of " << slack.size() << std::endl;
        
        unsigned int n = P.dimension();
        const NT dl = 0.995;
        _lambdas.setZero(P.num_of_hyperplanes());
        _Av.setZero(P.num_of_hyperplanes());
        _p = p;
        _v = GetDirection<Point>::apply(n, rng);
        
        std::cout << "Initial direction: " << _v.getCoefficients().transpose() << std::endl;

        NT T = rng.sample_urdist() * _Len;
        std::cout << "Initial travel distance T: " << T << std::endl;
        
        // Rest of initialization...
        Point p0 = _p;
        auto pbpair = line_positive_intersect(_p, _v, _lambdas, _Av);
        
        std::cout << "First intersection result: lambda=" << pbpair.first << ", facet=" << pbpair.second << std::endl;
        std::cout << "=== END INITIALIZATION DEBUG ===" << std::endl;
            


        NT lambda = pbpair.first;
        int facet = pbpair.second;

        /* ========== NEW SAFETY GUARD ========== */
        if (facet < 0) {           // no positive intersection found
            _p += T * _v;          // free flight the remaining distance
            _lambda_prev = T;
            std::cout << "WARNING: No positive intersection found" << std::endl;
        }

        if (T <= pbpair.first) {
            _p += (T * _v);
            _lambda_prev = T;
            return;
        }
        
        _lambda_prev = dl * pbpair.first;
        _p += (_lambda_prev * _v);
        T -= _lambda_prev;
        
        compute_reflection(_v, _p);
        
        int it = 0;
        while (it <= 50*n)
        {
            Point r_new = _p + _lambda_prev * _v;
            auto pbpair = line_positive_intersect(r_new, _v, _lambdas, _Av);
            
            if (T <= pbpair.first) {
                _p += (T * _v);
                _lambda_prev = T;
                break;
            } else if (it == 50*n) {
                _lambda_prev = rng.sample_urdist() * pbpair.first;
                _p += (_lambda_prev * _v);
                break;
            }
            
            _lambda_prev = dl * pbpair.first;
            _p += (_lambda_prev * _v);
            T -= _lambda_prev;
            
            compute_reflection(_v, _p);
            it++;
        }
    }

    // Inside the Walk class, add this debug function:
    void debug_walk_state(const std::string& label, int step = -1) {
        static int debug_count = 0;
        if (debug_count++ > 10) return;  // Limit debug output
        
        std::cout << "\n=== WALK STATE: " << label;
        if (step >= 0) std::cout << " (step " << step << ")";
        std::cout << " ===" << std::endl;
        
        // Current position
        VT p_vec = _p.getCoefficients();
        std::cout << "Position ||p||: " << p_vec.norm() << std::endl;
        
        // Check feasibility
        VT Ap = _A * p_vec;
        VT slack = _b - Ap;
        int violated = (slack.array() < -1e-10).count();
        std::cout << "Feasibility: " << (violated == 0 ? "OK" : "VIOLATED") 
                << " (slack range: [" << slack.minCoeff() << ", " << slack.maxCoeff() << "])" << std::endl;
        
        // Position in rounded space
        VT p_rounded = _L * p_vec;
        std::cout << "||p_rounded||: " << p_rounded.norm() 
                << " (ratio: " << p_rounded.norm() / p_vec.norm() << ")" << std::endl;
        
        // Velocity info
        if (_v.dimension() > 0) {
            VT v_vec = _v.getCoefficients();
            VT v_rounded = _L * v_vec;
            std::cout << "||v||: " << v_vec.norm() << ", ||v_rounded||: " << v_rounded.norm() << std::endl;
            
            // Compute velocity norm in Hessian metric
            VT Lv = _L.template triangularView<Eigen::Lower>().solve(v_vec);
            std::cout << "||v||_H (Hessian norm): " << Lv.norm() << std::endl;
        }
    }

     // Updated member variables for corrected approach
    SparseRowMT _A;              // Original sparse A
    VT _b;                       // Original b vector  
    SparseMT _L;                 // L where H^(-1) = L * L^T (lower triangular)
    VT _b_scaled;                // b vector (unchanged for now)
    
    // Keep the rest for compatibility
    NT _Len;
    Point _p;
    Point _v;
    NT _lambda_prev;
    VT _lambdas;
    VT _Av;
    parameters _param;
    MT _original_A_dense;  // Store for comparison
    VT _original_b;        // Store for comparison

};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP