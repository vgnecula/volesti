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

        // Step 1: Apply coordinate transformation using triangular solve
        // From comment: "Ax = A * L_inv.template triangularView<Eigen::Upper>().solve(x)"
        // But L is lower triangular, so we need to solve L * y = x for y
        // This gives us y = L^(-1) * x, which is the rounded coordinate
        
        VT r_original = r.getCoefficients();
        VT v_original = v.getCoefficients();
        
        // Solve L * r_rounded = r_original for r_rounded
        VT r_rounded = r_original;
        VT v_rounded = v_original;
        
        // CRITICAL: Use lower triangular solve since L is lower triangular
        _L.template triangularView<Eigen::Lower>().solveInPlace(r_rounded);
        _L.template triangularView<Eigen::Lower>().solveInPlace(v_rounded);
        
        if (debug_print) {
            std::cout << "Lazy coordinate transformation:" << std::endl;
            std::cout << "  ||r_original||: " << r_original.norm() << std::endl;
            std::cout << "  ||r_rounded||:  " << r_rounded.norm() << std::endl;
            std::cout << "  Transformation ratio: " << r_rounded.norm() / r_original.norm() << std::endl;
        }

        // Step 2: Apply constraints in original space to transformed coordinates
        // This preserves sparsity of A!
        Ar_out = _A * r_rounded;
        Av_out = _A * v_rounded;

        if (debug_print) {
            // Verify feasibility after transformation
            VT slack_transformed = _b_scaled - Ar_out;
            int violated_transformed = (slack_transformed.array() < -1e-10).count();
            std::cout << "Transformed feasibility: violated=" << violated_transformed << "/" << slack_transformed.size() << std::endl;
            std::cout << "Slack range: [" << slack_transformed.minCoeff() << ", " << slack_transformed.maxCoeff() << "]" << std::endl;
            
            // Double-check: verify point is feasible in original space
            VT Ar_original_check = _A * r_original;
            VT slack_original_check = _b - Ar_original_check;
            std::cout << "Original space check: violated=" << (slack_original_check.array() < -1e-10).count() << "/" << slack_original_check.size() << std::endl;
        }

        // Step 3: Find intersection λ such that A * (r + λ*v) = b
        NT lambda_min = std::numeric_limits<NT>::max();
        int facet = -1;
        int positive_lambdas = 0;
        
        for (int i = 0; i < Av_out.size(); ++i)
        {
            NT av = Av_out(i);
            if (std::abs(av) < NT(1e-12)) continue;

            NT lambda = (_b_scaled(i) - Ar_out(i)) / av;
            
            if (debug_print && i < 5) {
                std::cout << "Constraint " << i << ": lambda=" << lambda << " (av=" << av << ")" << std::endl;
            }

            if (lambda > NT(1e-12)) {
                positive_lambdas++;
                if (lambda < lambda_min) {
                    lambda_min = lambda;
                    facet = i;
                    _param.inner_vi_ak = av;
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

        return {lambda_min, facet};
    }

    // CRITICAL FIX 4: Corrected reflection oracle
    void compute_reflection(Point& v, Point& u)
    {
        NT coef = -2.0 * _param.inner_vi_ak;
        int facet = _param.facet_prev;
        
        if (facet < 0 || facet >= _A.rows()) {
            std::cout << "WARNING: Invalid facet " << facet << std::endl;
            return;
        }
        
        // Get the constraint normal from original A (preserves sparsity)
        VT a_facet(_A.cols());
        for (int j = 0; j < _A.cols(); ++j) {
            a_facet(j) = _A.coeff(facet, j);
        }
        
        // Transform normal to rounded space: a_rounded = L^(-T) * a_facet
        // Since L is lower triangular, L^(-T) is upper triangular
        // Solve L^T * a_rounded = a_facet for a_rounded
        VT a_rounded = a_facet;
        _L.transpose().template triangularView<Eigen::Upper>().solveInPlace(a_rounded);
        
        // Apply reflection in rounded space
        Point reflection(coef * a_rounded);
        v += reflection;
        u += reflection;
        
        static int reflection_count = 0;
        reflection_count++;
        if (reflection_count <= 3) {
            std::cout << "Corrected reflection " << reflection_count << ": facet=" << facet 
                      << ", coef=" << coef << ", ||a_rounded||=" << a_rounded.norm() << std::endl;
        }
    }

public:

    // CRITICAL FIX 5: Enhanced walk with better error handling
    template<typename GenericPolytope>
    inline void apply(GenericPolytope &P,
                    Point& p,
                    unsigned int const& walk_length,
                    RandomNumberGenerator &rng)
    {
        static int walk_call_count = 0;
        walk_call_count++;
        bool debug_walk = (walk_call_count == 1);  // Only debug first walk
        
        if (debug_walk) {
            std::cout << "\n=== CORRECTED WALK DEBUG ===" << std::endl;
            std::cout << "Walk length: " << walk_length << std::endl;
        }

        unsigned int n = P.dimension();
        const NT dl = 0.995;
        
        int successful_steps = 0;
        int failed_steps = 0;
        int resets = 0;

        for (auto j = 0u; j < walk_length; ++j)
        {
            NT T = rng.sample_urdist() * _Len;
            _v = GetDirection<Point>::apply(n, rng);

            Point p0 = _p;
            int it = 0;
            int consecutive_failures = 0;
            
            while (it < 50 * n)
            {
                auto pbpair = line_positive_intersect(_p, _v, _lambdas, _Av);

                NT lambda = pbpair.first;
                int facet = pbpair.second;

                // Better error handling
                if (facet < 0 || lambda <= 0 || lambda == std::numeric_limits<NT>::max()) {
                    consecutive_failures++;
                    
                    if (consecutive_failures > 3) {  // Reduced threshold
                        // Reset to feasible state
                        _p = p0;
                        _v = GetDirection<Point>::apply(n, rng);
                        resets++;
                        if (debug_walk && j < 3) {
                            std::cout << "Reset at step " << j << " after " << consecutive_failures << " failures" << std::endl;
                        }
                        break;
                    }
                    
                    // Small perturbation
                    _p += 0.001 * GetDirection<Point>::apply(n, rng);
                    it++;
                    continue;
                }
                
                consecutive_failures = 0;

                if (T <= lambda) {
                    _p += (T * _v);
                    _lambda_prev = T;
                    successful_steps++;
                    break;
                }

                _lambda_prev = dl * lambda;
                _p += (_lambda_prev * _v);
                T -= _lambda_prev;

                compute_reflection(_v, _p);
                it++;
            }
            
            if (it >= 50 * n) {
                _p = p0;  // Reset on timeout
                resets++;
            }
        }
        
        if (debug_walk) {
            std::cout << "Walk completed: " << successful_steps << "/" << walk_length << " successful" << std::endl;
            std::cout << "Resets: " << resets << std::endl;
            std::cout << "=== END CORRECTED WALK DEBUG ===" << std::endl;
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