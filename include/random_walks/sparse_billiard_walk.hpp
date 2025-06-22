#ifndef RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP
#define RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "convex_bodies/hpolytope.h"
#include "sampling/sphere.hpp"
#include "generators/boost_random_number_generator.hpp"

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

        double m_L;
        bool   set_L;

        NT  inner_vi_ak; 
        int  facet_prev;
    }; 

    template <typename GenericPolytope>
    Walk(GenericPolytope &P, 
         Point const& p, 
         RandomNumberGenerator &rng,
         parameters const& user_params,
         SparseMT const& Hessian) : _param(user_params)
    {
        _Len = _param.set_L ? _param.m_L : NT(6.0) * std::sqrt(static_cast<double>(P.dimension()));
 
        // Store original sparse A and b
        _A = P.get_mat();
        _b = P.get_vec();
        
        compute_cholesky_and_transformations(Hessian);
        
        VT p_original = p.getCoefficients();
        VT p_rounded = _L_inv.transpose().template triangularView<Eigen::Lower>() * p_original; 
        
        Point p_rounded_point(p_rounded);
        initialize(P, p_rounded_point, rng);
    }

private:

    void compute_cholesky_and_transformations(const SparseMT &H)
    {   
        Eigen::SimplicialLLT<SparseMT, Eigen::Lower> Chol(H);
        if (Chol.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Cholesky decomposition failed");
        }
        
        _L_inv = Chol.matrixL().transpose();
        
        MT A_dense = _A.toDense();
        MT A_transposed = A_dense.transpose();
        MT temp = _L_inv.template triangularView<Eigen::Upper>().solve(A_transposed);
        _A_rounded = temp.transpose();
        
        _A_rounded_row_norms.setZero(_A_rounded.rows());
        NT* A_rounded_row_norms_data = _A_rounded_row_norms.data();
        for (int i = 0; i < _A_rounded.rows(); ++i) {
            NT row_norm = _A_rounded.row(i).norm();
            *A_rounded_row_norms_data = row_norm;
            _A_rounded.row(i) /= row_norm;
            A_rounded_row_norms_data++;
        }
    }

public:

    // walk operates entirely in rounded space
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

            _lambda_prev = 0;
            VT p_round = _p.getCoefficients();
            _Ar = _A * _L_inv.transpose().template triangularView<Eigen::Lower>().solve(p_round);

            Point p0 = _p;
            int it = 0;

            while (it < 50 * n)
            {
                std::pair<NT,int> pbpair;

                if (it == 0) {
                    pbpair = P.sparse_line_positive_intersect(_p, _v, _sparams);
                    VT v_round = _v.getCoefficients();
                    _Av  = _A * _L_inv.transpose().template triangularView<Eigen::Lower>().solve(v_round);
                } else {
                    pbpair = P.sparse_line_positive_intersect(_p, _v, _Ar, _Av, _lambda_prev, _sparams);
                }

                if (T <= pbpair.first) {
                    _p += T * _v;
                    _lambda_prev = T;
                    break;
                }

                _lambda_prev = dl * pbpair.first;
                _p += _lambda_prev * _v;
                T -= _lambda_prev;

                P.sparse_compute_reflection(_v, _sparams);
                // 1. recompute _Ar (A × current position in original coordinates)
                VT p_round    = _p.getCoefficients();
                _Ar = _A * _L_inv.transpose()
                            .template triangularView<Eigen::Lower>()
                            .solve(p_round);

                // 2. reset the relative displacement for the next segment
                _lambda_prev = 0;
                it++;
            }

            if (it == 50 * n)
                _p = p0;
        } 
        
        VT p_rounded = _p.getCoefficients();
        VT p_original = _L_inv.transpose().template triangularView<Eigen::Lower>().solve(p_rounded);
        p = Point(p_original);
    }

private:

    template<typename GenericPolytope>
    inline void initialize(GenericPolytope &P,
                        Point const& p_rounded,
                        RandomNumberGenerator &rng)
    {
        unsigned int n = P.dimension();
        const NT dl = 0.995;
        
        _p = p_rounded;  // Already in rounded space
        _v = GetDirection<Point>::apply(n, rng);
                
        _Ar.setZero(_A.rows());
        _Av.setZero(_A.rows());
        _lambda_prev = 0;
        
        NT T = rng.sample_urdist() * _Len;
        
        auto pbpair = P.sparse_line_positive_intersect(_p, _v, _sparams);
        
        if (pbpair.second < 0) {
            _p += T * _v;
            _lambda_prev = T;
            return;
        }
        
        if (T <= pbpair.first) {
            _p += (T * _v);
            _lambda_prev = T;
            return;
        }
        
        _lambda_prev = dl * pbpair.first;
        _p += (_lambda_prev * _v);
        T -= _lambda_prev;
        
        P.sparse_compute_reflection(_v, _sparams);
        // 1. recompute _Ar (A × current position in original coordinates)
        VT p_round    = _p.getCoefficients();
        _Ar = _A * _L_inv.transpose()
                    .template triangularView<Eigen::Lower>()
                    .solve(p_round);

        // 2. reset the relative displacement for the next segment
        _lambda_prev = 0;
        
        int it = 0;
        while (it <= 50*n && T > 0)
        {
            auto pbpair2 = P.sparse_line_positive_intersect(_p, _v, _Ar, _Av, _lambda_prev, _sparams);
            
            if (T <= pbpair2.first) {
                _p += (T * _v);
                _lambda_prev = T;
                break;
            } else if (it == 50*n) {
                _lambda_prev = rng.sample_urdist() * pbpair2.first;
                _p += (_lambda_prev * _v);
                break;
            }
            
            _lambda_prev = dl * pbpair2.first;
            _p += (_lambda_prev * _v);
            T -= _lambda_prev;
            
            P.sparse_compute_reflection(_v, _sparams);
            // 1. recompute _Ar (A × current position in original coordinates)
            VT p_round    = _p.getCoefficients();
            _Ar = _A * _L_inv.transpose()
                        .template triangularView<Eigen::Lower>()
                        .solve(p_round);

            // 2. reset the relative displacement for the next segment
            _lambda_prev = 0;
            it++;
        }
       
    }

    SparseRowMT _A; 
    VT _b;
    SparseMT _L_inv;
    MT _A_rounded;
    VT _A_rounded_row_norms;
    
    NT _Len;
    Point _p;
    Point _v;
    parameters _param;

    VT _Ar;
    VT _Av;
    NT _lambda_prev;

    /// Carries the **static** rounding data + the two mutable
    /// reflection scalars that HPolytope needs.
    struct SparseParams {
        // immutable references (point to the matrices owned by the walk)
        const SparseMT &L_inv;
        const MT       &A_rounded;
        const VT       &row_norms;
        // updated by HPolytope
        NT  inner_vi_ak = NT(0);
        int facet_prev  = -1;
    };

    // ➋  the walk owns one instance
    SparseParams _sparams { _L_inv, _A_rounded, _A_rounded_row_norms };
};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP