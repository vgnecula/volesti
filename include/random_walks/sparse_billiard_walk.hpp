// VolEsti (volume computation and sampling library)
// FIXED SPARSE BILLIARD WALK - Following Tolis's specifications exactly

#ifndef RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP
#define RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP

#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "convex_bodies/hpolytope.h"
#include "sampling/sphere.hpp"
#include "generators/boost_random_number_generator.hpp"

// Sparse Billiard walk with proper lazy rounding
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

    // Constructor following Tolis's specification
    template <typename GenericPolytope>
    Walk(GenericPolytope &P, 
         Point const& p, 
         RandomNumberGenerator &rng,
         parameters const& user_params,
         SparseMT const& Hessian) : _param(user_params)
    {
        // Conservative billiard length for rounded space
        _Len = _param.set_L ? _param.m_L : NT(6.0) * std::sqrt(static_cast<double>(P.dimension()));
 
        // Store original sparse A and b
        _A_sparse = P.get_mat();
        _b = P.get_vec();
        
        // STEP 1: Compute L_inv from Hessian
        compute_transformation_matrices(Hessian);
        
        // STEP 2: Compute A_rounded and A_rounded_row_norms (Tolis's requirement)
        compute_rounded_polytope_matrices();
        
        // STEP 3: Transform starting point to rounded space
        VT p_original = p.getCoefficients();
        VT p_rounded = _L_dense * p_original;  // y = L * x (forward transform)
        
        std::cout << "=== FIXED SPARSE INITIALIZATION ===" << std::endl;
        std::cout << "Billiard length: " << _Len << std::endl;
        std::cout << "A_rounded size: " << _A_rounded.rows() << "x" << _A_rounded.cols() << std::endl;
        std::cout << "A_rounded_row_norms size: " << _A_rounded_row_norms.size() << std::endl;
        std::cout << "Starting point (original): " << p_original.transpose() << std::endl;
        std::cout << "Starting point (rounded): " << p_rounded.transpose() << std::endl;
        std::cout << "=== END FIXED INITIALIZATION ===" << std::endl;
        
        Point p_rounded_point(p_rounded);
        initialize(P, p_rounded_point, rng);
    }

private:
    
    void compute_transformation_matrices(const SparseMT &H)
    {
        std::cout << "=== TRANSFORMATION MATRICES COMPUTATION ===" << std::endl;
        
        // Convert to dense for Cholesky
        MT H_dense = H.toDense();
        
        // Cholesky decomposition H = L * L^T
        Eigen::LLT<MT> llt(H_dense);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("Cholesky decomposition of H failed");
        
        // Get L and L^(-1)
        _L_dense = llt.matrixL();
        _L_inv_dense = _L_dense.template triangularView<Eigen::Lower>().solve(
            MT::Identity(H.rows(), H.cols())
        );
        
        std::cout << "L and L_inv computed successfully" << std::endl;
        std::cout << "=== END TRANSFORMATION COMPUTATION ===" << std::endl;
    }
    
    void compute_rounded_polytope_matrices()
    {
        std::cout << "=== COMPUTING A_ROUNDED AND ROW NORMS ===" << std::endl;
        
        int m = _A_sparse.rows();
        int n = _A_sparse.cols();
        
        // Convert sparse A to dense for transformation: A_rounded = A * L_inv
        MT A_dense = _A_sparse.toDense();
        _A_rounded = A_dense * _L_inv_dense;
        
        // Compute row norms of A_rounded
        _A_rounded_row_norms.resize(m);
        for (int i = 0; i < m; ++i) {
            _A_rounded_row_norms(i) = _A_rounded.row(i).norm();
            if (_A_rounded_row_norms(i) < NT(1e-14)) {
                throw std::runtime_error("Zero row norm in A_rounded");
            }
        }
        
        // Normalize A_rounded rows (Tolis's requirement)
        for (int i = 0; i < m; ++i) {
            _A_rounded.row(i) /= _A_rounded_row_norms(i);
        }
        
        std::cout << "A_rounded computed and normalized" << std::endl;
        std::cout << "Row norms range: [" << _A_rounded_row_norms.minCoeff() 
                  << ", " << _A_rounded_row_norms.maxCoeff() << "]" << std::endl;
        std::cout << "=== END A_ROUNDED COMPUTATION ===" << std::endl;
    }

    // Lazy oracle: use sparse A for constraint evaluation, A_rounded for inner products
    std::pair<NT, int>
    line_positive_intersect(Point const& r, Point const& v)
    {
        static int debug_call_count = 0;
        debug_call_count++;
        bool debug_print = (debug_call_count <= 3);
        
        VT r_rounded = r.getCoefficients();
        VT v_rounded = v.getCoefficients();
        
        // Transform back to original space for constraint evaluation: x = L_inv * y
        VT r_original = _L_inv_dense * r_rounded;
        VT v_original = _L_inv_dense * v_rounded;
        
        // Use sparse A for constraint evaluation (Tolis's specification)
        VT Ar = _A_sparse * r_original;  // Sparse matrix-vector multiply
        VT Av = _A_sparse * v_original;  // Sparse matrix-vector multiply

        if (debug_print) {
            VT slack = _b - Ar;
            int violated = (slack.array() < -1e-10).count();
            std::cout << "\n=== LAZY ORACLE (call " << debug_call_count << ") ===" << std::endl;
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
                    
                    // Compute inner product using A_rounded (Tolis's specification)
                    VT a_rounded_row = _A_rounded.row(i).transpose();
                    _param.inner_vi_ak = v_rounded.dot(a_rounded_row);
                    _param.facet_prev = i;
                }
            }
        }

        if (debug_print) {
            std::cout << "Best intersection: lambda=" << lambda_min << ", facet=" << facet << std::endl;
            if (facet >= 0) {
                std::cout << "inner_vi_ak: " << _param.inner_vi_ak << std::endl;
            }
            std::cout << "=== END LAZY ORACLE ===" << std::endl;
        }

        if (facet == -1) {
            lambda_min = std::numeric_limits<NT>::max();
        }

        return {lambda_min, facet};
    }

    // Reflection using A_rounded (Tolis's specification)
    void compute_reflection(Point& v, Point const&)
    {
        static int reflection_count = 0;
        reflection_count++;
        bool debug = (reflection_count <= 3);
        
        int facet = _param.facet_prev;
        if (facet < 0 || facet >= _A_rounded.rows()) {
            std::cout << "ERROR: Invalid facet " << facet << std::endl;
            return;
        }
        
        if (debug) {
            std::cout << "\n=== REFLECTION " << reflection_count << " ===" << std::endl;
            std::cout << "Facet: " << facet << std::endl;
            std::cout << "inner_vi_ak: " << _param.inner_vi_ak << std::endl;
        }
        
        // Reflection using normalized A_rounded (Tolis's exact specification)
        VT a_rounded_row = _A_rounded.row(facet).transpose();
        Point a_point((-2.0 * _param.inner_vi_ak) * a_rounded_row);
        v += a_point;
        
        if (debug) {
            std::cout << "||a_rounded_row||: " << a_rounded_row.norm() << std::endl;
            std::cout << "||v_after||: " << v.length() << std::endl;
            std::cout << "=== END REFLECTION ===" << std::endl;
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
            std::cout << "\n=== FIXED WALK " << walk_call_count << " START ===" << std::endl;
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
                // Check feasibility using lazy evaluation
                VT p_rounded = _p.getCoefficients();
                VT p_original = _L_inv_dense * p_rounded;
                VT Ap = _A_sparse * p_original;
                VT slack = _b - Ap;
                int violated = (slack.array() < -1e-10).count();
                std::cout << "Step " << j << " feasibility: " << (violated == 0 ? "OK" : "VIOLATED") 
                         << " (violations: " << violated << "/" << slack.size() << ")" << std::endl;
            }
        }
        
        if (debug_walk) {
            std::cout << "=== FIXED WALK " << walk_call_count << " END ===" << std::endl;
        }
        
        // Transform back to original space: x = L_inv * y
        VT p_rounded = _p.getCoefficients();
        VT p_original = _L_inv_dense * p_rounded;
        p = Point(p_original);
    }

private:

    template<typename GenericPolytope>
    inline void initialize(GenericPolytope &P,
                        Point const& p_rounded,
                        RandomNumberGenerator &rng)
    {
        std::cout << "\n=== FIXED INITIALIZATION ===" << std::endl;
        
        // Check feasibility of starting point
        VT p_rounded_coeffs = p_rounded.getCoefficients();
        VT p_original = _L_inv_dense * p_rounded_coeffs;
        VT Ap = _A_sparse * p_original;
        VT slack = _b - Ap;
        int violated = (slack.array() < 0).count();
        std::cout << "Initial feasibility: " << (violated == 0 ? "OK" : "VIOLATED") << std::endl;
        
        if (violated > 0) {
            throw std::runtime_error("Starting point is not feasible in fixed implementation!");
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
        
        std::cout << "=== END FIXED INITIALIZATION ===" << std::endl;
    }

    // Member variables following Tolis's specification
    SparseRowMT _A_sparse;          // Original sparse A matrix (for constraint evaluation only)
    VT _b;                          // Original b vector  
    MT _L_dense;                    // L matrix from Cholesky
    MT _L_inv_dense;                // L^(-1) matrix
    MT _A_rounded;                  // A_rounded = A * L_inv (dense, row-normalized)
    VT _A_rounded_row_norms;        // Row norms of original A_rounded
    
    // Standard billiard walk members
    NT _Len;
    Point _p;      // Position in rounded space
    Point _v;      // Velocity in rounded space
    parameters _param;
};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP