// VolEsti (volume computation and sampling library)
// CLEAN SPARSE BILLIARD WALK - Correct transformations, simple design

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
        NT  inner_vi_ak; // ⟨v,a_k⟩ / ‖a_k‖²  of the facet we just hit  
        int  facet_prev;  // index of that facet
    }; 

    // Constructor following Tolis's TRUE lazy specification
    template <typename GenericPolytope>
    Walk(GenericPolytope &P, 
         Point const& p, 
         RandomNumberGenerator &rng,
         parameters const& user_params,
         SparseMT const& Hessian) : _param(user_params)
    {
        // Conservative billiard length for rounded space stability
        _Len = _param.set_L ? _param.m_L : NT(1.0) * std::sqrt(static_cast<double>(P.dimension()));
 
        // Store original sparse A and b (NEVER transform them - this is the key!)
        _A = P.get_mat();
        _b = P.get_vec();
        
        // STEP 1: Compute L_inv from Hessian and prepare for efficient solves
        compute_cholesky_factor(Hessian);
        
        // STEP 2: Transform starting point to rounded space
        VT p_original = p.getCoefficients();
        VT p_rounded = _L_dense * p_original;  // CORRECTED: y = L * x (forward transform)
        
        std::cout << "=== CLEAN SPARSE INITIALIZATION ===" << std::endl;
        std::cout << "Billiard length: " << _Len << std::endl;
        std::cout << "Starting point (original): " << p_original.transpose() << std::endl;
        std::cout << "Starting point (rounded): " << p_rounded.transpose() << std::endl;
        
        // Verify lazy evaluation: A * L_inv * p_rounded should equal A * p_original
        VT p_transformed_back = _L_inv_dense * p_rounded;  // CORRECTED: x = L_inv * y (inverse transform)
        VT Ap_original = _A * p_original;  // ← KEEP SPARSE!
        VT Ap_lazy = _A * p_transformed_back;  // ← KEEP SPARSE!
        NT lazy_error = (Ap_original - Ap_lazy).norm();
        std::cout << "Lazy evaluation error: " << lazy_error << " (should be ~0)" << std::endl;
        std::cout << "=== END CLEAN INITIALIZATION ===" << std::endl;
        
        Point p_rounded_point(p_rounded);
        initialize(P, p_rounded_point, rng);
    }

private:
    
    void compute_cholesky_factor(const SparseMT &H)
    {
        std::cout << "=== CHOLESKY FACTOR COMPUTATION ===" << std::endl;
        std::cout << "Hessian size: " << H.rows() << "x" << H.cols() << std::endl;
        
        // Convert to dense for Cholesky (Hessian is typically small and dense anyway)
        MT H_dense = H.toDense();
        
        // Step 1: Cholesky decomposition H = L * L^T
        Eigen::LLT<MT> llt(H_dense);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("Cholesky decomposition of H failed");
        
        // Step 2: Get L and L^(-1) - keep both dense for efficient triangular solves
        _L_dense = llt.matrixL();
        _L_inv_dense = _L_dense.template triangularView<Eigen::Lower>().solve(
            MT::Identity(H.rows(), H.cols())
        );
        
        // For consistency, also keep sparse version (though we prefer dense for solves)
        _L_inv = _L_inv_dense.sparseView();
        
        std::cout << "L and L_inv computed (dense cached for efficiency)" << std::endl;
        std::cout << "=== END CHOLESKY COMPUTATION ===" << std::endl;
    }

    // OPTIMIZED: True lazy oracle with sparse operations
    std::pair<NT, int>
    line_positive_intersect(Point const& r, Point const& v,
                            VT& Ar_out, VT& Av_out)
    {
        static int debug_call_count = 0;
        debug_call_count++;
        bool debug_print = (debug_call_count <= 3);
        
        if (debug_print) {
            std::cout << "\n=== SPARSE ORACLE (call " << debug_call_count << ") ===" << std::endl;
        }

        VT r_rounded = r.getCoefficients();
        VT v_rounded = v.getCoefficients();
        
        // CORRECTED: Transform back to original space: x = L_inv * y
        VT r_original = _L_inv_dense * r_rounded;
        VT v_original = _L_inv_dense * v_rounded;
        
        // Apply original SPARSE A to transformed coordinates - THIS IS THE KEY!
        Ar_out = _A * r_original;  // ← SPARSE matrix-vector multiply!
        Av_out = _A * v_original;  // ← SPARSE matrix-vector multiply!

        if (debug_print) {
            VT slack = _b - Ar_out;
            int violated = (slack.array() < -1e-10).count();
            std::cout << "Feasibility check: violated=" << violated << "/" << slack.size() << std::endl;
            std::cout << "Slack range: [" << slack.minCoeff() << ", " << slack.maxCoeff() << "]" << std::endl;
        }

        // Find intersection λ such that A * (r_original + λ*v_original) = b
        NT lambda_min = std::numeric_limits<NT>::max();
        int facet = -1;
        
        for (int i = 0; i < Av_out.size(); ++i)
        {
            NT av = Av_out(i);
            if (std::abs(av) < NT(1e-12)) continue;

            NT lambda = (_b(i) - Ar_out(i)) / av;

            if (lambda > NT(1e-12)) {
                if (lambda < lambda_min) {
                    lambda_min = lambda;
                    facet = i;
                    
                    // For reflection, compute the inner product efficiently
                    // Get facet normal from sparse A (avoid dense conversion)
                    VT a_original = extract_sparse_row(_A, i);
                    
                    // Transform normal to rounded space: a_rounded = L^T * a_original
                    VT a_rounded = _L_dense.transpose() * a_original;
                    _param.inner_vi_ak = v_rounded.dot(a_rounded) / a_rounded.squaredNorm();
                    _param.facet_prev = i;
                }
            }
        }

        if (debug_print) {
            std::cout << "Best intersection: lambda=" << lambda_min << ", facet=" << facet << std::endl;
            std::cout << "=== END SPARSE ORACLE ===" << std::endl;
        }

        if (facet == -1) {
            lambda_min = std::numeric_limits<NT>::max();
        }

        return {lambda_min, facet};
    }
    
    // Helper: Extract row from sparse matrix efficiently
    VT extract_sparse_row(const SparseRowMT& A, int row_idx) {
        VT row_dense = VT::Zero(A.cols());
        for (typename SparseRowMT::InnerIterator it(A, row_idx); it; ++it) {
            row_dense(it.col()) = it.value();
        }
        return row_dense;
    }

    // OPTIMIZED: Lazy reflection with efficient operations
    void compute_reflection(Point& v, Point const&)
    {
        static int reflection_count = 0;
        reflection_count++;
        bool debug = (reflection_count <= 3);
        
        int facet = _param.facet_prev;
        if (facet < 0 || facet >= _A.rows()) {
            std::cout << "ERROR: Invalid facet " << facet << std::endl;
            return;
        }
        
        if (debug) {
            std::cout << "\n=== SPARSE REFLECTION " << reflection_count << " ===" << std::endl;
            std::cout << "Facet: " << facet << std::endl;
        }
        
        // Get the constraint normal from original sparse A efficiently
        VT a_original = extract_sparse_row(_A, facet);
        
        // Transform normal to rounded space: a_rounded = L^T * a_original  
        VT a_rounded = _L_dense.transpose() * a_original;
        VT v_rounded = v.getCoefficients();
        
        // Reflection in rounded space using transformed normal
        NT v_dot_a = v_rounded.dot(a_rounded);
        NT a_norm_sq = a_rounded.squaredNorm();
        
        if (a_norm_sq < NT(1e-14)) {
            std::cout << "ERROR: Zero normal vector in rounded space" << std::endl;
            return;
        }
        
        VT v_reflected = v_rounded - 2.0 * (v_dot_a / a_norm_sq) * a_rounded;
        
        if (debug) {
            std::cout << "inner_vi_ak: " << _param.inner_vi_ak << std::endl;
            std::cout << "v_dot_a: " << v_dot_a << std::endl;
            std::cout << "||a_original||: " << a_original.norm() << std::endl;
            std::cout << "||a_rounded||: " << a_rounded.norm() << std::endl;
            std::cout << "||v_before||: " << v_rounded.norm() << std::endl;
            std::cout << "||v_after||: " << v_reflected.norm() << std::endl;
        }
        
        v = Point(v_reflected);
        
        if (debug) {
            std::cout << "=== END SPARSE REFLECTION ===" << std::endl;
        }
    }

public:

    // Walk operates entirely in rounded space with optimized lazy evaluation
    template<typename GenericPolytope>
    inline void apply(GenericPolytope &P,
                    Point& p,
                    unsigned int const& walk_length,
                    RandomNumberGenerator &rng)
    {
        static int walk_call_count = 0;
        walk_call_count++;
        bool debug_walk = (walk_call_count <= 2);
        
        if (debug_walk) {
            std::cout << "\n=== SPARSE WALK " << walk_call_count << " START ===" << std::endl;
            std::cout << "Walk length: " << walk_length << std::endl;
            std::cout << "Billiard length: " << _Len << std::endl;
        }

        unsigned int n = P.dimension();
        const NT dl = 0.995;
        
        for (auto j = 0u; j < walk_length; ++j)
        {
            NT T = rng.sample_urdist() * _Len;
            _v = GetDirection<Point>::apply(n, rng);

            Point p0 = _p;
            int it = 0;
            
            while (it < 50 * n)
            {
                auto pbpair = line_positive_intersect(_p, _v, _lambdas, _Av);
                NT lambda = pbpair.first;
                int facet = pbpair.second;

                if (facet < 0 || lambda <= 0 || lambda == std::numeric_limits<NT>::max()) {
                    // Reset and try again
                    _p = p0;
                    _v = GetDirection<Point>::apply(n, rng);
                    break;
                }

                if (T <= lambda) {
                    _p += (T * _v);
                    _lambda_prev = T;
                    break;
                }

                _lambda_prev = dl * lambda;
                _p += (_lambda_prev * _v);
                T -= _lambda_prev;

                compute_reflection(_v, _p);
                it++;
            }
            
            if (debug_walk && j < 3) {
                // Check feasibility using corrected lazy evaluation
                VT p_rounded = _p.getCoefficients();
                VT p_original = _L_inv_dense * p_rounded;  // CORRECTED: x = L_inv * y
                VT Ap = _A * p_original;  // ← SPARSE multiply!
                VT slack = _b - Ap;
                int violated = (slack.array() < -1e-10).count();
                std::cout << "Step " << j << " feasibility: " << (violated == 0 ? "OK" : "VIOLATED") 
                         << " (violations: " << violated << "/" << slack.size() << ")" << std::endl;
                if (violated > 0) {
                    std::cout << "  Slack range: [" << slack.minCoeff() << ", " << slack.maxCoeff() << "]" << std::endl;
                }
            }
        }
        
        if (debug_walk) {
            std::cout << "=== SPARSE WALK " << walk_call_count << " END ===" << std::endl;
        }
        
        // Transform back to original space for output: x = L_inv * y
        VT p_rounded = _p.getCoefficients();
        VT p_original = _L_inv_dense * p_rounded;  // CORRECTED transformation
        p = Point(p_original);
    }

private:

    template<typename GenericPolytope>
    inline void initialize(GenericPolytope &P,
                        Point const& p_rounded,
                        RandomNumberGenerator &rng)
    {
        std::cout << "\n=== SPARSE INITIALIZATION ===" << std::endl;
        std::cout << "Polytope dimension: " << P.dimension() << std::endl;
        std::cout << "Starting point (rounded): " << p_rounded.getCoefficients().transpose() << std::endl;
        
        // Check if starting point is feasible using corrected lazy evaluation
        VT p_rounded_coeffs = p_rounded.getCoefficients();
        VT p_original = _L_inv_dense * p_rounded_coeffs;  // CORRECTED: x = L_inv * y
        VT Ap = _A * p_original;  // ← SPARSE multiply!
        VT slack = _b - Ap;
        std::cout << "Initial feasibility: slack range [" << slack.minCoeff() << ", " << slack.maxCoeff() << "]" << std::endl;
        int violated = (slack.array() < 0).count();
        std::cout << "Violated constraints: " << violated << " out of " << slack.size() << std::endl;
        
        if (violated > 0) {
            std::cout << "ERROR: Starting point is not feasible!" << std::endl;
            std::cout << "This indicates a bug in the coordinate transformation." << std::endl;
        }
        
        unsigned int n = P.dimension();
        const NT dl = 0.995;
        _lambdas.setZero(P.num_of_hyperplanes());
        _Av.setZero(P.num_of_hyperplanes());
        _p = p_rounded;  // Already in rounded space
        _v = GetDirection<Point>::apply(n, rng);

        NT T = rng.sample_urdist() * _Len;
        
        auto pbpair = line_positive_intersect(_p, _v, _lambdas, _Av);
        NT lambda = pbpair.first;
        int facet = pbpair.second;

        if (facet < 0) {
            _p += T * _v;
            _lambda_prev = T;
            std::cout << "WARNING: No positive intersection found in initialization" << std::endl;
            return;
        }

        if (T <= lambda) {
            _p += (T * _v);
            _lambda_prev = T;
            return;
        }
        
        _lambda_prev = dl * lambda;
        _p += (_lambda_prev * _v);
        T -= _lambda_prev;
        
        compute_reflection(_v, _p);
        
        // Continue initialization with reflections
        int it = 0;
        while (it <= 50*n && T > 0)
        {
            auto pbpair = line_positive_intersect(_p, _v, _lambdas, _Av);
            
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
        
        std::cout << "=== END SPARSE INITIALIZATION ===" << std::endl;
    }

    // Member variables for optimized lazy approach
    SparseRowMT _A;          // Original sparse A matrix (NEVER transformed!)
    VT _b;                   // Original b vector  
    SparseMT _L_inv;         // L^(-1) transformation matrix (sparse, for compatibility)
    MT _L_dense;             // L matrix (dense, for efficient triangular solves)
    MT _L_inv_dense;         // L^(-1) matrix (dense, for efficient operations)
    
    // Standard billiard walk members
    NT _Len;
    Point _p;      // Position in rounded space
    Point _v;      // Velocity in rounded space
    NT _lambda_prev;
    VT _lambdas;
    VT _Av;
    parameters _param;
};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP