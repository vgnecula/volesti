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
        Eigen::SimplicialLLT<SparseMT, Eigen::Lower> chol(H);
        if (chol.info() != Eigen::Success)
            throw std::runtime_error("Cholesky decomposition failed");

        _L = chol.matrixL();    
        _A_rounded = (_A * _L).template cast<NT>();      // dense copy

        // L⁻¹ (upper‑triangular, sparse)
        Eigen::SparseMatrix<NT, Eigen::ColMajor> I(_L.rows(), _L.cols());
        I.setIdentity();
        Eigen::SparseLU<SparseMT> solver;
        solver.analyzePattern(_L);
        solver.factorize(_L);
        _L_inv = solver.solve(I);
        _L_inv.prune([](int r,int c,NT){return r<=c;});

        // -------- row norms of A⋅L and scaled b ---------------------------------
        int m = _A.rows();
        _row_norm.resize(m);
        for (int i=0;i<m;++i){
            Eigen::SparseVector<NT> row = _A.row(i)*_L;    // A_i· L
            NT nrm = std::sqrt(row.squaredNorm());
            _row_norm(i) = (nrm>0)? nrm: NT(1);
        }
        _b_scaled = _b.array() / _row_norm.array();

        for (int i=0;i<_A_rounded.rows();++i)
            _A_rounded.row(i) /= _row_norm(i);           // unit normals
    }
 

    // Lazy boundary oracle using sparse triangular solves (key innovation)
    // From TolisChal: "So, for example, in the boundary oracle instead of doing Ax = A_rounded*x 
    //                  we do Ax = A * L_inv.template triangularView<Eigen::Upper>().solve(x);"
    std::pair<NT, int>
    line_positive_intersect(Point const& r, Point const& v,
                            VT& Ar_out, VT& Av_out)
    {
        // Solve L⁻¹ * r and v (to go into rounded space)
        VT r_solved = r.getCoefficients();
        _L_inv.template triangularView<Eigen::Upper>().solveInPlace(r_solved);

        VT v_solved = v.getCoefficients();
        _L_inv.template triangularView<Eigen::Upper>().solveInPlace(v_solved);

        // Compute A * r and A * v
        Ar_out.noalias() = _A * r_solved;

        VT raw_Av = _A * v_solved;    // We'll keep this unscaled for inner_vi_ak
        Av_out = raw_Av;              // Copy to be scaled for reflection and intersection

        Ar_out.array() /= _row_norm.array();
        Av_out.array()   /= _row_norm.array();

        NT lambda_min = std::numeric_limits<NT>::max();
        int facet = -1;

        for (int i = 0; i < Av_out.size(); ++i)
        {
            NT av = Av_out(i);
            if (std::abs(av) < NT(1e-12)) continue;  // Skip near-parallel

            NT lambda = (_b_scaled(i) - Ar_out(i)) / av;

            if (lambda > 0 && lambda < lambda_min)
            {
                lambda_min      = lambda;
                facet           = i;
                _param.inner_vi_ak = raw_Av(i) / _row_norm(i); // scaled
                _param.facet_prev  = i;
            }
        }

        return {lambda_min, facet};
    }



    // Sparse reflection oracle with lazy rounding
    // From general comment: "implementing lazy boundary and reflection oracles"
    void compute_reflection(Point& v, Point& u)
    {
        NT coef = -2.0 * _param.inner_vi_ak;
        int facet = _param.facet_prev;
        VT row = _A_rounded.row(facet);
        v += Point(coef * row);
        u += Point(coef * row);
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

                NT lambda = pbpair.first;
                int facet = pbpair.second;

                /* ========== NEW SAFETY GUARD ========== */
                if (facet < 0) {           // no positive intersection found
                    _p += T * _v;          // free flight the remaining distance
                    _lambda_prev = T;
                    std::cout << "WARNING: No positive intersection found" << std::endl;
                    break;                 // exit the while(it < 50*n) loop
                }

                if (T <= pbpair.first) {
                    _p += (T * _v);
                    _lambda_prev = T;
                    break;
                }

                _lambda_prev = dl * pbpair.first;
                _p += (_lambda_prev * _v);
                T -= _lambda_prev;

                // Use lazy reflection oracle (key difference from uniform billiard walk)
                compute_reflection(_v, _p);

                it++;
            }
            if (it == 50*n){
                _p = p0;
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
    SparseMT _L_inv;             // L^T where H^{-1} = L * L^T
    
    // Walk state (same as uniform billiard walk for compatibility)
    // From conversation: "keep the structure and everything the same as for the uniform billiard walk"
    NT _Len;
    Point _p;
    Point _v;
    NT _lambda_prev;
    VT _lambdas;
    VT _Av;

    SparseMT _L;                     // lower‑triangular   (H⁻¹ = L·Lᵀ)    // ‖A▭_{i·}‖  – cached once
    VT       _b_scaled;              // b  divided by those norms
    VT       _row_norm;
    parameters _param; 
    MT _A_rounded;

};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP