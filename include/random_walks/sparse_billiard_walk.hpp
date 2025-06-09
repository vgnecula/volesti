// VolEsti (volume computation and sampling library)
// SPARSE BILLIARD WALK - Following Tolis's exact specifications with LAZY COMPUTATION

#ifndef RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP
#define RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <unordered_map>

#include "convex_bodies/hpolytope.h"
#include "sampling/sphere.hpp"
#include "generators/boost_random_number_generator.hpp"

// Sparse Billiard walk following Tolis's exact specifications with lazy computation
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
        NT  inner_vi_ak; // ⟨v,a_k⟩ / ‖a_k‖²  of the rounded facet  
        int  facet_prev;  // index of that facet
    }; 

    // Lazy computation cache for A_rounded rows and their norms
    struct LazyCache {
        std::unordered_map<int, VT> a_rounded_rows;      // facet_id -> normalized A_rounded row
        std::unordered_map<int, NT> a_rounded_row_norms; // facet_id -> original row norm
        
        // Statistics for monitoring
        int total_facets;
        int computed_facets;
        
        LazyCache() : total_facets(0), computed_facets(0) {}
        
        void print_stats() const {
            std::cout << "Lazy cache stats: " << computed_facets << "/" << total_facets 
                      << " facets computed (" << (100.0 * computed_facets / std::max(1, total_facets)) 
                      << "%)" << std::endl;
        }
    };

    // Constructor following Tolis's exact specification with lazy initialization
    template <typename GenericPolytope>
    Walk(GenericPolytope &P, 
         Point const& p, 
         RandomNumberGenerator &rng,
         parameters const& user_params,
         SparseMT const& Hessian) : _param(user_params)
    {
        // Use same billiard length as dense version for fair comparison
        _Len = _param.set_L ? _param.m_L : NT(6.0) * std::sqrt(static_cast<double>(P.dimension()));
 
        // Store original sparse A and b
        _A = P.get_mat();
        _b = P.get_vec();
        
        // Initialize lazy cache
        _lazy_cache.total_facets = _A.rows();
        
        // STEP 1: Compute transformations following Tolis's exact code
        compute_cholesky_and_transformations(Hessian);
        
        // STEP 2: Transform starting point to rounded space
        VT p_original = p.getCoefficients();
        VT p_rounded = forward_transform(p_original);  // y = L * x
        
        std::cout << "=== TOLIS SPARSE INITIALIZATION (LAZY) ===" << std::endl;
        std::cout << "Billiard length: " << _Len << std::endl;
        std::cout << "Total facets: " << _lazy_cache.total_facets << std::endl;
        std::cout << "L_inv is sparse: " << _L_inv.nonZeros() << " nnz" << std::endl;
        std::cout << "Starting point (original): " << p_original.transpose() << std::endl;
        std::cout << "Starting point (rounded): " << p_rounded.transpose() << std::endl;
        std::cout << "=== END TOLIS INITIALIZATION (LAZY) ===" << std::endl;
        
        Point p_rounded_point(p_rounded);
        initialize(P, p_rounded_point, rng);
    }

private:
    
    void compute_cholesky_and_transformations(const SparseMT &H)
    {
        std::cout << "=== TOLIS CHOLESKY COMPUTATION (LAZY) ===" << std::endl;
        
        // Follow Tolis's ACTUAL code, not his comment
        // His code does: Chol(H), even though comment says "factor of H^{-1}"
        Eigen::SimplicialLLT<SparseMT, Eigen::Lower> Chol(H);
        if (Chol.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Cholesky decomposition failed");
        }
        
        // Tolis's exact code: _L_inv = Chol.matrixL().transpose()
        _L_inv = Chol.matrixL().transpose();
        
        std::cout << "L_inv computed (sparse triangular): " << _L_inv.rows() << "x" << _L_inv.cols() 
                  << " with " << _L_inv.nonZeros() << " nnz" << std::endl;
        
        // NO LONGER compute A_rounded and A_rounded_row_norms upfront!
        // They will be computed lazily as needed
        
        std::cout << "Lazy computation enabled - A_rounded rows will be computed on demand" << std::endl;
        std::cout << "=== END TOLIS CHOLESKY (LAZY) ===" << std::endl;
    }
    
    // Lazy computation of A_rounded row and its norm for a specific facet
    std::pair<VT, NT> get_a_rounded_row_and_norm(int facet_id)
    {
        // Check if already computed
        auto row_it = _lazy_cache.a_rounded_rows.find(facet_id);
        if (row_it != _lazy_cache.a_rounded_rows.end()) {
            // Already computed, return cached values
            return {row_it->second, _lazy_cache.a_rounded_row_norms[facet_id]};
        }
        
        // Compute on demand following Tolis's exact formula
        // _A_rounded.row(i) = L_inv.triangularView.solve(A.row(i).transpose()).transpose()
        
        // Extract the i-th row of A (sparse)
        VT a_row = _A.row(facet_id).transpose();  // Convert row to column vector
        
        // Apply Tolis's transformation: L_inv^T \ a_row
        VT a_rounded_row = _L_inv.template triangularView<Eigen::Upper>().solve(a_row);
        
        // Compute norm before normalization
        NT row_norm = a_rounded_row.norm();
        
        // Normalize the row
        if (row_norm > NT(1e-12)) {
            a_rounded_row /= row_norm;
        } else {
            // Handle degenerate case
            row_norm = NT(1.0);
        }
        
        // Cache the results
        _lazy_cache.a_rounded_rows[facet_id] = a_rounded_row;
        _lazy_cache.a_rounded_row_norms[facet_id] = row_norm;
        _lazy_cache.computed_facets++;
        
        static bool debug_lazy = true;
        if (debug_lazy && _lazy_cache.computed_facets <= 5) {
            std::cout << "Lazily computed facet " << facet_id << ": norm=" << row_norm 
                      << ", ||a_rounded||=" << a_rounded_row.norm() << std::endl;
        }
        
        return {a_rounded_row, row_norm};
    }
    
    // Forward and inverse transforms remain the same
    VT forward_transform(const VT& x) const {
        return _L_inv.transpose().template triangularView<Eigen::Lower>() * x;
    }
    
    VT inverse_transform(const VT& y) const {
        return _L_inv.transpose().template triangularView<Eigen::Lower>().solve(y);
    }

    // Tolis's sparse oracle with lazy computation
    std::pair<NT, int>
    line_positive_intersect(Point const& r, Point const& v)
    {
        static int debug_call_count = 0;
        debug_call_count++;
        bool debug_print = (debug_call_count <= 3);
        
        VT r_rounded = r.getCoefficients();
        VT v_rounded = v.getCoefficients();
        
        // Tolis's exact sparse oracle specification:
        VT r_original = inverse_transform(r_rounded);
        VT v_original = inverse_transform(v_rounded);
        
        VT Ar = _A * r_original;  // Sparse matrix-vector multiply
        VT Av = _A * v_original;  // Sparse matrix-vector multiply

        if (debug_print) {
            VT slack = _b - Ar;
            int violated = (slack.array() < -1e-10).count();
            std::cout << "\n=== TOLIS SPARSE ORACLE (LAZY, call " << debug_call_count << ") ===" << std::endl;
            std::cout << "Feasibility: violated=" << violated << "/" << slack.size() << std::endl;
        }

        // Find intersection λ such that A * (r_original + λ*v_original) = b
        NT lambda_min = std::numeric_limits<NT>::max();
        int facet = -1;
        
        for (int i = 0; i < Av.size(); ++i)
        {
            NT av = Av(i);
            if (std::abs(av) < NT(1e-12)) continue;

            NT lambda = (_b(i) - Ar(i)) / av;

            if (lambda > NT(1e-12)) {
                if (lambda < lambda_min) {
                    lambda_min = lambda;
                    facet = i;
                    
                    // LAZY COMPUTATION: Only compute A_rounded row when we need it!
                    auto [a_rounded_row, row_norm] = get_a_rounded_row_and_norm(i);
                    
                    // Compute inner product using the lazily computed row
                    _param.inner_vi_ak = v_rounded.dot(a_rounded_row);
                    _param.facet_prev = i;
                }
            }
        }

        if (debug_print) {
            std::cout << "Best intersection: lambda=" << lambda_min << ", facet=" << facet << std::endl;
            if (facet >= 0) {
                std::cout << "inner_vi_ak: " << _param.inner_vi_ak << std::endl;
                auto [_, row_norm] = get_a_rounded_row_and_norm(facet);
                std::cout << "row_norm: " << row_norm << std::endl;
            }
            _lazy_cache.print_stats();
            std::cout << "=== END TOLIS SPARSE ORACLE (LAZY) ===" << std::endl;
        }

        if (facet == -1) {
            lambda_min = std::numeric_limits<NT>::max();
        }

        return {lambda_min, facet};
    }

    // Reflection using Tolis's exact specification with lazy computation
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
            std::cout << "\n=== TOLIS REFLECTION (LAZY) " << reflection_count << " ===" << std::endl;
            std::cout << "Facet: " << facet << std::endl;
            std::cout << "inner_vi_ak: " << _param.inner_vi_ak << std::endl;
        }
        
        // LAZY COMPUTATION: Get the A_rounded row (will be cached from oracle call)
        auto [a_rounded_row, row_norm] = get_a_rounded_row_and_norm(facet);
        
        // Tolis's exact reflection formula:
        Point a((-2.0 * _param.inner_vi_ak) * a_rounded_row);
        v += a;
        
        if (debug) {
            std::cout << "||a_rounded_row||: " << a_rounded_row.norm() << std::endl;
            std::cout << "||v_after||: " << v.length() << std::endl;
            std::cout << "=== END TOLIS REFLECTION (LAZY) ===" << std::endl;
        }
    }

public:

    // Walk operates entirely in rounded space
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
            std::cout << "\n=== TOLIS SPARSE WALK (LAZY) " << walk_call_count << " START ===" << std::endl;
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
                auto pbpair = line_positive_intersect(_p, _v);
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
                    break;
                }

                NT lambda_used = dl * lambda;
                _p += (lambda_used * _v);
                T -= lambda_used;

                compute_reflection(_v, _p);
                it++;
            }
            
            if (debug_walk && j < 3) {
                // Check feasibility using Tolis's lazy evaluation
                VT p_rounded = _p.getCoefficients();
                VT p_original = inverse_transform(p_rounded);
                VT Ap = _A * p_original;  // Sparse multiply
                VT slack = _b - Ap;
                int violated = (slack.array() < -1e-10).count();
                std::cout << "Step " << j << " feasibility: " << (violated == 0 ? "OK" : "VIOLATED") 
                         << " (violations: " << violated << "/" << slack.size() << ")" << std::endl;
            }
        }
        
        if (debug_walk) {
            std::cout << "Final lazy cache stats:" << std::endl;
            _lazy_cache.print_stats();
            std::cout << "=== TOLIS SPARSE WALK (LAZY) " << walk_call_count << " END ===" << std::endl;
        }
        
        // Transform back to original space using Tolis's method
        VT p_rounded = _p.getCoefficients();
        VT p_original = inverse_transform(p_rounded);
        p = Point(p_original);
    }

private:

    template<typename GenericPolytope>
    inline void initialize(GenericPolytope &P,
                        Point const& p_rounded,
                        RandomNumberGenerator &rng)
    {
        std::cout << "\n=== TOLIS INITIALIZATION (LAZY) ===" << std::endl;
        
        // Check feasibility using Tolis's sparse method
        VT p_rounded_coeffs = p_rounded.getCoefficients();
        VT p_original = inverse_transform(p_rounded_coeffs);
        VT Ap = _A * p_original;  // Sparse multiply
        VT slack = _b - Ap;
        int violated = (slack.array() < 0).count();
        std::cout << "Initial feasibility: " << (violated == 0 ? "OK" : "VIOLATED") << std::endl;
        
        if (violated > 0) {
            throw std::runtime_error("Starting point is not feasible in Tolis implementation!");
        }
        
        unsigned int n = P.dimension();
        const NT dl = 0.995;
        
        _p = p_rounded;  // Already in rounded space
        _v = GetDirection<Point>::apply(n, rng);

        NT T = rng.sample_urdist() * _Len;
        
        auto pbpair = line_positive_intersect(_p, _v);
        NT lambda = pbpair.first;
        int facet = pbpair.second;

        if (facet < 0) {
            _p += T * _v;
            return;
        }

        if (T <= lambda) {
            _p += (T * _v);
            return;
        }
        
        NT lambda_used = dl * lambda;
        _p += (lambda_used * _v);
        T -= lambda_used;
        
        compute_reflection(_v, _p);
        
        // Continue with reflections
        int it = 0;
        while (it <= 50*n && T > 0)
        {
            auto pbpair2 = line_positive_intersect(_p, _v);
            
            if (T <= pbpair2.first) {
                _p += (T * _v);
                break;
            } else if (it == 50*n) {
                NT final_lambda = rng.sample_urdist() * pbpair2.first;
                _p += (final_lambda * _v);
                break;
            }
            
            NT lambda_step = dl * pbpair2.first;
            _p += (lambda_step * _v);
            T -= lambda_step;
            
            compute_reflection(_v, _p);
            it++;
        }
        
        std::cout << "Initialization complete. ";
        _lazy_cache.print_stats();
        std::cout << "=== END TOLIS INITIALIZATION (LAZY) ===" << std::endl;
    }

    // Member variables following Tolis's exact specification with lazy computation
    SparseRowMT _A;                  // Original sparse A matrix  
    VT _b;                           // Original b vector  
    SparseMT _L_inv;                 // Sparse triangular L_inv (Cholesky factor of H^{-1})
    
    // REMOVED: Dense A_rounded and A_rounded_row_norms - now computed lazily!
    LazyCache _lazy_cache;           // Lazy computation cache
    
    // Standard billiard walk members
    NT _Len;
    Point _p;      // Position in rounded space
    Point _v;      // Velocity in rounded space
    parameters _param;
};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP