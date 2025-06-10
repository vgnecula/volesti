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
        
        // compute transformations following Tolis's exact code
        compute_cholesky_and_transformations(Hessian);
        
        // transform starting point to rounded space
        VT p_original = p.getCoefficients();
        VT p_rounded = forward_transform(p_original);  // y = L * x
        
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
    
    // Let's go back to the working transformation logic but use Tolis's matrices
    VT forward_transform(const VT& x) const {
        // Since _L_inv = L^T from Chol(H), we need L = _L_inv^T
        // Forward: y = L * x = _L_inv^T * x
        return _L_inv.transpose().template triangularView<Eigen::Lower>() * x;
    }
    
    VT inverse_transform(const VT& y) const {
        // Inverse: x = L^{-1} * y = (_L_inv^T)^{-1} * y = _L_inv^{-T} * y
        // This is equivalent to solving _L_inv^T * x = y
        return _L_inv.transpose().template triangularView<Eigen::Lower>().solve(y);
    }

    std::pair<NT, int>
    line_positive_intersect(Point const& r, Point const& v)
    {
        
        VT r_rounded = r.getCoefficients();
        VT v_rounded = v.getCoefficients();
        
        VT r_original = inverse_transform(r_rounded);
        VT v_original = inverse_transform(v_rounded);
        
        VT Ar = _A * r_original;
        VT Av = _A * v_original;


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
                    
                    // Compute inner product and rescale by row norm
                    VT a_rounded_row = _A_rounded.row(i).transpose();
                    _param.inner_vi_ak = v_rounded.dot(a_rounded_row);
                    // Note: _param.inner_vi_ak should be rescaled by A_rounded_row_norms, 
                    // but since A_rounded is already normalized, this is correct
                    _param.facet_prev = i;
                }
            }
        }


        if (facet == -1) {
            lambda_min = std::numeric_limits<NT>::max();
        }

        return {lambda_min, facet};
    }

    void compute_reflection(Point& v, Point const&)
    {
        
        int facet = _param.facet_prev;
        if (facet < 0 || facet >= _A_rounded.rows()) {
            std::cout << "ERROR: Invalid facet " << facet << std::endl;
            return;
        }
        
        VT a_rounded_row = _A_rounded.row(facet).transpose();
        Point a((-2.0 * _param.inner_vi_ak) * a_rounded_row);
        v += a;
        

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

            Point p0 = _p;
            int it = 0;
            
            while (it < 50 * n)
            {
                auto pbpair = line_positive_intersect(_p, _v);
                NT lambda = pbpair.first;
                int facet = pbpair.second;

                if (facet < 0 || lambda <= 0 || lambda == std::numeric_limits<NT>::max()) {
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
        
        
        // Check feasibility using Tolis's sparse method
        VT p_rounded_coeffs = p_rounded.getCoefficients();
        VT p_original = inverse_transform(p_rounded_coeffs);
        VT Ap = _A * p_original;  // Sparse multiply
        VT slack = _b - Ap;
        int violated = (slack.array() < 0).count();
        
        if (violated > 0) {
            throw std::runtime_error("Starting point is not feasible in the implementation!");
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
};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP