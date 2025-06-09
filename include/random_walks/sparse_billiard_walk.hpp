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
    
    // Fixed Cholesky computation
    void compute_cholesky_factor(const SparseMT &H)
    {
        std::cout << "=== CHOLESKY FACTOR DEBUG ===" << std::endl;
        std::cout << "Hessian size: " << H.rows() << "x" << H.cols() << std::endl;
        std::cout << "Hessian nnz: " << H.nonZeros() << std::endl;
        
        // Check if H is positive definite
        Eigen::VectorXd eigenvals = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(H.toDense()).eigenvalues();
        std::cout << "Min eigenvalue: " << eigenvals.minCoeff() << std::endl;
        std::cout << "Max eigenvalue: " << eigenvals.maxCoeff() << std::endl;
        std::cout << "Condition number: " << eigenvals.maxCoeff() / eigenvals.minCoeff() << std::endl;
        
        // Compute Cholesky: H = L * L^T
        Eigen::SimplicialLLT<SparseMT, Eigen::Lower> chol(H);
        if (chol.info() != Eigen::Success)
            throw std::runtime_error("Cholesky decomposition failed");

        _L = chol.matrixL();  // Lower triangular L
        std::cout << "L size: " << _L.rows() << "x" << _L.cols() << std::endl;
        std::cout << "L nnz: " << _L.nonZeros() << std::endl;
        
        // _L_inv is L^T for the coordinate transformation: x_rounded = L^{-T} * x
        _L_inv = _L.transpose();
        std::cout << "L_inv (L^T) nnz: " << _L_inv.nonZeros() << std::endl;

        // CRITICAL FIX: Compute A_rounded = A * L^{-1} (NOT A * L^{-T})
        // Mathematical reasoning:
        // - Points transform as: x_rounded = L^{-T} * x_original  
        // - Constraints transform as: A_rounded * x_rounded ≤ b
        // - Substituting: A_rounded * (L^{-T} * x_original) ≤ b
        // - Therefore: A_rounded = A * L^{-1} * L^T = A * L^{-1}
        
        int m = _A.rows();
        int n = _A.cols();
        _row_norm.resize(m);
        _A_rounded.resize(m, n);
        
        // Compute A_rounded = A * L^{-1} by solving L * Y = A^T, then A_rounded = Y^T
        MT A_dense = MT(_A);  // Convert sparse A to dense for easier computation
        MT L_dense = MT(_L);  // Convert sparse L to dense
        
        // Solve L * A_rounded^T = A^T for A_rounded^T
        MT A_rounded_T = L_dense.template triangularView<Eigen::Lower>().solve(A_dense.transpose());
        _A_rounded = A_rounded_T.transpose();
        
        // Compute row norms BEFORE normalization
        for (int i = 0; i < m; ++i) {
            NT nrm = _A_rounded.row(i).norm();
            _row_norm(i) = (nrm > NT(1e-12)) ? nrm : NT(1);
        }
        
        // Scale b by row norms  
        _b_scaled = _b.array() / _row_norm.array();
        
        // Normalize A_rounded rows to have unit norm
        for (int i = 0; i < m; ++i) {
            _A_rounded.row(i) /= _row_norm(i);
        }

        std::cout << "Row norms range: [" << _row_norm.minCoeff() << ", " << _row_norm.maxCoeff() << "]" << std::endl;
        std::cout << "b_scaled range: [" << _b_scaled.minCoeff() << ", " << _b_scaled.maxCoeff() << "]" << std::endl;
        std::cout << "=== END CHOLESKY DEBUG ===" << std::endl;
    }
 

    // Lazy boundary oracle using sparse triangular solves (key innovation)
    // From TolisChal: "So, for example, in the boundary oracle instead of doing Ax = A_rounded*x 
    //                  we do Ax = A * L_inv.template triangularView<Eigen::Upper>().solve(x);"
    std::pair<NT, int>
    line_positive_intersect(Point const& r, Point const& v,
                            VT& Ar_out, VT& Av_out)
    {
        static int debug_call_count = 0;
        debug_call_count++;
        
        // Only print for first few calls to avoid spam
        bool debug_print = (debug_call_count <= 5);
        
        if (debug_print) {
            std::cout << "\n=== INTERSECTION DEBUG (call " << debug_call_count << ") ===" << std::endl;
            std::cout << "Input r: " << r.getCoefficients().transpose() << std::endl;
            std::cout << "Input v: " << v.getCoefficients().transpose() << std::endl;
        }

        // CORRECT: Use L^T (upper triangular) solve to transform to rounded coordinates
        // Mathematical reasoning: if H = L*L^T, then to transform x to rounded space,
        // we need L^{-T} * x, which is solved by L^T \ x (upper triangular solve)
        VT r_rounded = r.getCoefficients();
        VT v_rounded = v.getCoefficients();
        
        // Transform both position and direction to rounded coordinates
        _L_inv.template triangularView<Eigen::Upper>().solveInPlace(r_rounded);
        _L_inv.template triangularView<Eigen::Upper>().solveInPlace(v_rounded);
        
        if (debug_print) {
            std::cout << "r_rounded (L^T solve): " << r_rounded.transpose() << std::endl;
            std::cout << "v_rounded (L^T solve): " << v_rounded.transpose() << std::endl;
        }

        // Compute A * r_rounded and A * v_rounded using original sparse matrix A
        Ar_out.noalias() = _A * r_rounded;
        VT raw_Av = _A * v_rounded;
        Av_out = raw_Av;

        if (debug_print) {
            std::cout << "A * r_rounded: " << Ar_out.transpose() << std::endl;
            std::cout << "A * v_rounded: " << Av_out.transpose() << std::endl;
        }

        // Normalize by row norms (this makes constraints unit-norm in rounded space)
        Ar_out.array() /= _row_norm.array();
        Av_out.array() /= _row_norm.array();

        if (debug_print) {
            std::cout << "Normalized Ar: " << Ar_out.transpose() << std::endl;
            std::cout << "Normalized Av: " << Av_out.transpose() << std::endl;
            std::cout << "b_scaled: " << _b_scaled.transpose() << std::endl;
        }

        NT lambda_min = std::numeric_limits<NT>::max();
        int facet = -1;
        int positive_lambdas = 0;
        
        // Find ray-polytope intersection in rounded space
        for (int i = 0; i < Av_out.size(); ++i)
        {
            NT av = Av_out(i);
            if (std::abs(av) < NT(1e-12)) continue;

            NT lambda = (_b_scaled(i) - Ar_out(i)) / av;
            
            if (debug_print && i < 10) {  // Print first 10 constraints
                std::cout << "Constraint " << i << ": lambda=" << lambda 
                        << " (b_scaled=" << _b_scaled(i) 
                        << " - Ar=" << Ar_out(i) << ") / av=" << av << std::endl;
            }

            if (lambda > NT(1e-12)) {  // Use small positive threshold
                positive_lambdas++;
                if (lambda < lambda_min) {
                    lambda_min = lambda;
                    facet = i;
                    // Store normalized value for reflection
                    _param.inner_vi_ak = raw_Av(i) / _row_norm(i);
                    _param.facet_prev = i;
                }
            }
        }

        if (debug_print) {
            std::cout << "Positive lambdas found: " << positive_lambdas << std::endl;
            std::cout << "Min lambda: " << lambda_min << ", facet: " << facet << std::endl;
            std::cout << "=== END INTERSECTION DEBUG ===" << std::endl;
        }

        // Return -1 facet if no valid intersection found
        if (facet == -1) {
            lambda_min = std::numeric_limits<NT>::max();
        }

        return {lambda_min, facet};
    } 




    // Fixed reflection oracle
    void compute_reflection(Point& v, Point& u)
    {
        NT coef = -2.0 * _param.inner_vi_ak;
        int facet = _param.facet_prev;
        
        if (facet < 0) return; // Safety check
        
        // Use pre-computed normalized A_rounded row
        VT row_vec = _A_rounded.row(facet);
        
        Point a(coef * row_vec);
        v += a;
        u += a;
    }
    



public:

    // Main walk function (same interface as uniform billiard walk)
    // From conversation: "keep the structure and everything the same as for the uniform billiard walk, 
    //                     but change the logic according to all the previous comments"
    // Fixed main walk function with better error handling
    template<typename GenericPolytope>
    inline void apply(GenericPolytope &P,
                    Point& p,
                    unsigned int const& walk_length,
                    RandomNumberGenerator &rng)
    {
        unsigned int n = P.dimension();
        const NT dl = 0.995;

        for (auto j = 0u; j < walk_length; ++j)
        {
            NT T = rng.sample_urdist() * _Len;
            _v = GetDirection<Point>::apply(n, rng);

            Point p0 = _p;
            int it = 0;
            int consecutive_failures = 0;
            
            while (it < 50 * n)
            {
                // Use lazy boundary oracle
                auto pbpair = line_positive_intersect(_p, _v, _lambdas, _Av);

                NT lambda = pbpair.first;
                int facet = pbpair.second;

                // Improved error handling
                if (facet < 0 || lambda <= 0 || lambda == std::numeric_limits<NT>::max()) {
                    consecutive_failures++;
                    if (consecutive_failures > 5) {
                        // Reset to a safe state
                        _p = p0;
                        _v = GetDirection<Point>::apply(n, rng);
                        std::cout << "WARNING: Resetting due to consecutive failures" << std::endl;
                        break;
                    }
                    
                    // Try a small random step
                    _p += 0.01 * GetDirection<Point>::apply(n, rng);
                    continue;
                }
                
                consecutive_failures = 0; // Reset failure counter

                if (T <= lambda) {
                    _p += (T * _v);
                    _lambda_prev = T;
                    break;
                }

                _lambda_prev = dl * lambda;
                _p += (_lambda_prev * _v);
                T -= _lambda_prev;

                // Use lazy reflection oracle
                compute_reflection(_v, _p);

                it++;
            }
            
            if (it == 50 * n) {
                _p = p0; // Reset on timeout
            }
        }
        
        p = Point(_p.getCoefficients());
    }

    inline void update_delta(NT L)
    {
        _Len = L;
    }

private:

    template<typename GenericPolytope>
    inline void initialize(GenericPolytope &P,
                        Point const& p,
                        RandomNumberGenerator &rng)
    {
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

    // Core sparse data structures
    // From TolisChal: "For those we need the _L_inv, A_rounded and A_rounded_row_norms"
    // But: "A next todo would be to apply lazy computations for both A_rounded and A_rounded_row_norms 
    //       by computing and storing only the vectors and norms for the facets the walk hits"
    SparseRowMT _A;              // Original sparse A (never transformed)
    VT _b;                       // Original b vector  
    SparseMT _L_inv;             // L^T where H = L * L^T (not H^{-1}) 
    
    // Walk state (same as uniform billiard walk for compatibility)
    // From conversation: "keep the structure and everything the same as for the uniform billiard walk"
    NT _Len;
    Point _p;
    Point _v;
    NT _lambda_prev;
    VT _lambdas;
    VT _Av;

    SparseMT _L;                         // lower‑triangular   (H = L·Lᵀ) 
    VT       _b_scaled;              // b  divided by those norms
    VT       _row_norm;
    parameters _param; 
    MT _A_rounded;

};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP