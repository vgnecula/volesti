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
    SparseBilliardWalk(double L)
            :   param(L, true)
    {}

    SparseBilliardWalk()
            :   param(0, false)
    {}

    struct parameters
    {
        parameters(double L, bool set)
                :   m_L(L), set_L(set)
        {}
        double m_L;
        bool set_L;
    };

    parameters param;

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

    // Constructor with Hessian matrix (core requirement from TolisChal)
    // From: "I would implement a new billiard walk struct that would take as input 
    //        a hessian matrix H and an unrounded polytope with a sparse matrix _A."
    template <typename GenericPolytope>
    Walk(GenericPolytope &P, 
         Point const& p, 
         RandomNumberGenerator &rng,
         parameters const& params,
         SparseMT const& Hessian)
    {
        _Len = params.set_L ? params.m_L : 2.0 * std::sqrt(static_cast<double>(P.dimension()));
        
        // From: "We don't apply the transformation on the polytope to preserve the sparsity of A"
        // Store original sparse A and b (never transform them)
        _A = P.get_mat();
        _b = P.get_vec();
        
        // Compute L_inv from Hessian as TolisChal specified
        compute_cholesky_factor(Hessian);
        
        initialize(P, p, rng);
    }

private:
    
    void compute_cholesky_factor(SparseMT const& H)
    {
        // From TolisChal: "Eigen::SimplicialLLT<Eigen::SparseMatrix<NT, Eigen::ColMajor>, Eigen::Lower> Chol(H);"
        // Use SimplicialLLT for sparse matrices
        Eigen::SimplicialLLT<SparseMT, Eigen::Lower> chol(H);
        if (chol.info() != Eigen::Success) {
            throw std::runtime_error("Cholesky decomposition failed");
        }
        
        // From TolisChal: "_L_inv = Chol.matrixL().transpose(); // _L is a Cholesky factor of H^{-1}"
        // L_inv = L^T where H^{-1} = L * L^T
        _L_inv = chol.matrixL().transpose();
    }

    // Lazy boundary oracle using sparse triangular solves (key innovation)
    // From TolisChal: "So, for example, in the boundary oracle instead of doing Ax = A_rounded*x 
    //                  we do Ax = A * L_inv.template triangularView<Eigen::Upper>().solve(x);"
    std::pair<NT, int> line_positive_intersect(Point const& r,
                                               Point const& v,
                                               VT& Ar,
                                               VT& Av) const
    {
        // Key insight from TolisChal's comment:
        // Instead of: Ax = A_rounded * x
        // We do: Ax = A * L_inv.triangularView<Eigen::Upper>().solve(x)
        
        VT r_coeffs = r.getCoefficients();
        VT v_coeffs = v.getCoefficients();
        
        // Solve L_inv * r_rounded = r_coeffs for r_rounded
        // This is equivalent to: r_rounded = L_inv^{-1} * r_coeffs
        VT r_rounded = r_coeffs;
        _L_inv.template triangularView<Eigen::Upper>().solveInPlace(r_rounded);
        
        // Solve L_inv * v_rounded = v_coeffs for v_rounded  
        // This is equivalent to: v_rounded = L_inv^{-1} * v_coeffs
        VT v_rounded = v_coeffs;
        _L_inv.template triangularView<Eigen::Upper>().solveInPlace(v_rounded);
        
        // From TolisChal: "The A_rounded is A*L"
        // Now compute A * r_rounded and A * v_rounded (lazy A_rounded)
        // This gives us A * L * x_rounded = A * x_original
        Ar.noalias() = _A * r_rounded;
        Av.noalias() = _A * v_rounded;
        
        // Standard intersection logic
        NT min_plus = std::numeric_limits<NT>::max();
        int facet = -1;
        
        for (int i = 0; i < _A.rows(); ++i) {
            if (std::abs(Av(i)) < NT(1e-14)) continue;
            
            NT lambda = (_b(i) - Ar(i)) / Av(i);
            if (lambda > 0 && lambda < min_plus) {
                min_plus = lambda;
                facet = i;
            }
        }
        
        return std::make_pair(min_plus, facet);
    }

    // Sparse reflection oracle with lazy rounding
    // From general comment: "implementing lazy boundary and reflection oracles"
    void compute_reflection(Point& v, Point const& p, int const& facet) const
    {
        // From TolisChal: "since the Hessian has a sparsity structure, the cholesky will also have zeros 
        //                  in the lower/upper part, so using sparse arithmetics would speedup the oracle computations"
        
        // Reflection in rounded space using lazy computation
        // Normal in original space: _A.row(facet)
        VT n_original = _A.row(facet).transpose().eval();
        
        // Transform normal to rounded space: n_rounded = L_inv^{-T} * n_original
        // Since L_inv is upper triangular: solve L_inv^T * n_rounded = n_original
        // From TolisChal: "we mainly use the matrix L computed from the cholesky decomposition 
        //                  of the Hessian to solve triangular linear systems"
        VT n_rounded = n_original;
        _L_inv.transpose().template triangularView<Eigen::Lower>().solveInPlace(n_rounded);
        
        // Transform velocity to rounded space
        VT v_coeffs = v.getCoefficients();
        VT v_rounded = v_coeffs;
        _L_inv.template triangularView<Eigen::Upper>().solveInPlace(v_rounded);
        
        // Reflect in rounded space: v_new = v - 2 * (v · n) * n / ||n||²
        NT n_norm2 = n_rounded.squaredNorm();
        if (n_norm2 < std::numeric_limits<NT>::epsilon()) {
            throw std::runtime_error("Normal vector has zero norm in compute_reflection");
        }
        
        NT dot_product = v_rounded.dot(n_rounded);
        VT v_reflected = v_rounded - 2.0 * (dot_product / n_norm2) * n_rounded;
        
        // Transform back to original space: v_original = L_inv * v_reflected
        // From general comment: "However, we apply the walk on the rounded polytope by implementing 
        //                        lazy boundary and reflection oracles"
        VT v_original = _L_inv * v_reflected;
        
        v = Point(v_original);
    }

public:

    // Main walk function (same interface as uniform billiard walk)
    // From conversation: "keep the structure and everything the same as for the uniform billiard walk, 
    //                     but change the logic according to all the previous comments"
    template<typename GenericPolytope>
    inline void apply(GenericPolytope &P,
                      Point& p,
                      unsigned int const& walk_length,
                      RandomNumberGenerator &rng)
    {
        unsigned int n = P.dimension();
        NT T = rng.sample_urdist() * _Len;
        const NT dl = 0.995;

        for (auto j=0u; j<walk_length; ++j)
        {
            T = rng.sample_urdist() * _Len;
            _v = GetDirection<Point>::apply(n, rng);

            Point p0 = _p;
            int it = 0;
            while (it < 50*n)
            {
                // Use lazy boundary oracle (key difference from uniform billiard walk)
                auto pbpair = line_positive_intersect(_p, _v, _lambdas, _Av);

                if (T <= pbpair.first) {
                    _p += (T * _v);
                    _lambda_prev = T;
                    break;
                }

                _lambda_prev = dl * pbpair.first;
                _p += (_lambda_prev * _v);
                T -= _lambda_prev;

                // Use lazy reflection oracle (key difference from uniform billiard walk)
                compute_reflection(_v, _p, pbpair.second);

                it++;
            }
            if (it == 50*n){
                _p = p0;
            }
        }
        p = _p;
        _p.set_to_origin();
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
        unsigned int n = P.dimension();
        const NT dl = 0.995;
        _lambdas.setZero(P.num_of_hyperplanes());
        _Av.setZero(P.num_of_hyperplanes());
        _p = p;
        _v = GetDirection<Point>::apply(n, rng);

        NT T = rng.sample_urdist() * _Len;
        Point p0 = _p;
        
        // Initial intersection
        auto pbpair = line_positive_intersect(_p, _v, _lambdas, _Av);
        
        if (T <= pbpair.first) {
            _p += (T * _v);
            _lambda_prev = T;
            return;
        }
        
        _lambda_prev = dl * pbpair.first;
        _p += (_lambda_prev * _v);
        T -= _lambda_prev;
        
        compute_reflection(_v, _p, pbpair.second);
        
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
            
            compute_reflection(_v, _p, pbpair.second);
            it++;
        }
    }

    // Core sparse data structures
    // From TolisChal: "For those we need the _L_inv, A_rounded and A_rounded_row_norms"
    // But: "A next todo would be to apply lazy computations for both A_rounded and A_rounded_row_norms 
    //       by computing and storing only the vectors and norms for the facets the walk hits"
    SparseRowMT _A;              // Original sparse A (never transformed)
    VT _b;                       // Original b vector  
    SparseMT _L_inv;             // L^T where H^{-1} = L * L^T
    
    // Walk state (same as uniform billiard walk for compatibility)
    // From conversation: "keep the structure and everything the same as for the uniform billiard walk"
    NT _Len;
    Point _p;
    Point _v;
    NT _lambda_prev;
    VT _lambdas;
    VT _Av;
};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP