// VolEsti (volume computation and sampling library)
// CORRECTED Sparse Billiard Walk for uniform distribution with lazy rounding

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

    // Constructor following Tolis's specification
    template <typename GenericPolytope>
    Walk(GenericPolytope &P, 
         Point const& p, 
         RandomNumberGenerator &rng,
         parameters const& user_params,
         SparseMT const& Hessian) : _param(user_params)
    {
        // Conservative billiard length for rounded space
        _Len = _param.set_L ? _param.m_L : NT(6) * std::sqrt(static_cast<double>(P.dimension()));
 
        // Store original sparse A and b (never transform them for sparsity)
        _A = P.get_mat();
        _b = P.get_vec();
        
        // STEP 1: Compute L_inv from Hessian as Tolis specified
        compute_cholesky_factor(Hessian);
        
        // STEP 2: Compute A_rounded and A_rounded_row_norms as Tolis specified
        compute_rounded_constraint_matrix();
        
        initialize(P, p, rng);
    }

private:
    
    void compute_cholesky_factor(const SparseMT &H)
    {
        std::cout << "=== CHOLESKY FACTOR COMPUTATION ===" << std::endl;
        std::cout << "Hessian size: " << H.rows() << "x" << H.cols() << std::endl;
        
        // Following Tolis's comment: "The Hessian can be used to round the polytope 
        // by computing the cholesky L_inv of H"
        Eigen::MatrixXd H_dense = H.toDense();
        
        // Compute Cholesky decomposition: H = L_inv^T * L_inv
        Eigen::LLT<Eigen::MatrixXd> llt_H(H_dense);
        if (llt_H.info() != Eigen::Success)
            throw std::runtime_error("Cholesky decomposition of H failed");
        
        // L_inv is the Cholesky factor: H = L_inv^T * L_inv
        Eigen::MatrixXd L_inv_dense = llt_H.matrixL().transpose(); // Upper triangular
        _L_inv = L_inv_dense.sparseView();
        
        std::cout << "L_inv computed, nnz: " << _L_inv.nonZeros() << std::endl;
        std::cout << "=== END CHOLESKY COMPUTATION ===" << std::endl;
    }

    void compute_rounded_constraint_matrix()
    {
        std::cout << "=== COMPUTING A_ROUNDED ===" << std::endl;
        
        // Following Tolis: "A_rounded and A_rounded_row_norms should be computed 
        // inside the billiard walk struct"
        
        // A_rounded = A * L_inv^(-1) = A * L_inv.inverse()
        // Since L_inv is upper triangular, we solve A * L_inv^(-1) = A * (L_inv^(-1))
        // This means: A_rounded^T = L_inv^(-T) * A^T
        
        Eigen::MatrixXd A_dense = _A.toDense();
        Eigen::MatrixXd L_inv_dense = _L_inv.toDense();
        
        // Compute A_rounded = A * L_inv^(-1)
        // Since L_inv is upper triangular: L_inv^(-1) can be computed by solving
        Eigen::MatrixXd L_inv_inv = L_inv_dense.inverse();
        _A_rounded = A_dense * L_inv_inv;
        
        // Compute row norms for normalization
        int m = _A_rounded.rows();
        _A_rounded_row_norms.resize(m);
        
        for (int i = 0; i < m; ++i) {
            NT row_norm = _A_rounded.row(i).norm();
            _A_rounded_row_norms(i) = row_norm;
            
            // Normalize the row (as Tolis specified: "row-wise normalized")
            if (row_norm > NT(1e-14)) {
                _A_rounded.row(i) /= row_norm;
            }
        }
        
        std::cout << "A_rounded computed, size: " << _A_rounded.rows() << "x" << _A_rounded.cols() << std::endl;
        std::cout << "Row norms range: [" << _A_rounded_row_norms.minCoeff() 
                  << ", " << _A_rounded_row_norms.maxCoeff() << "]" << std::endl;
        std::cout << "=== END A_ROUNDED COMPUTATION ===" << std::endl;
    }

    // CORRECTED: Follow Tolis's oracle specification exactly
    std::pair<NT, int>
    line_positive_intersect(Point const& r, Point const& v,
                            VT& Ar_out, VT& Av_out)
    {
        static int debug_call_count = 0;
        debug_call_count++;
        bool debug_print = (debug_call_count <= 3);
        
        if (debug_print) {
            std::cout << "\n=== CORRECTED ORACLE (call " << debug_call_count << ") ===" << std::endl;
        }

        // Following Tolis's specification exactly:
        // "Ax.noalias() = A * L_inv.template triangularView<Eigen::Upper>().solve(r.getCoefficients());"
        // "Av.noalias() = A * L_inv.template triangularView<Eigen::Upper>().solve(v.getCoefficients());"
        
        VT r_coeffs = r.getCoefficients();
        VT v_coeffs = v.getCoefficients();
        
        // Solve L_inv * x = r  =>  x = L_inv^(-1) * r
        VT r_transformed = _L_inv.template triangularView<Eigen::Upper>().solve(r_coeffs);
        VT v_transformed = _L_inv.template triangularView<Eigen::Upper>().solve(v_coeffs);
        
        // Apply original sparse A to transformed coordinates
        Ar_out = _A * r_transformed;
        Av_out = _A * v_transformed;

        if (debug_print) {
            VT slack = _b - Ar_out;
            int violated = (slack.array() < -1e-10).count();
            std::cout << "Feasibility check: violated=" << violated << "/" << slack.size() << std::endl;
            std::cout << "Slack range: [" << slack.minCoeff() << ", " << slack.maxCoeff() << "]" << std::endl;
        }

        // Find intersection λ such that A * (r_transformed + λ*v_transformed) = b
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
                    
                    // Following Tolis: store rescaled inner product for reflection
                    // "params.inner_vi_ak = *Av_data;"
                    // "params.inner_v_ak /= A_rounded_row_norms.coeff(facet);"
                    _param.inner_vi_ak = av / _A_rounded_row_norms(i);
                    _param.facet_prev = i;
                }
            }
        }

        if (debug_print) {
            std::cout << "Best intersection: lambda=" << lambda_min << ", facet=" << facet << std::endl;
            std::cout << "=== END CORRECTED ORACLE ===" << std::endl;
        }

        if (facet == -1) {
            lambda_min = std::numeric_limits<NT>::max();
        }

        return {lambda_min, facet};
    } 

    // CORRECTED: Follow Tolis's reflection specification
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
            std::cout << "\n=== CORRECTED REFLECTION " << reflection_count << " ===" << std::endl;
            std::cout << "Facet: " << facet << std::endl;
        }
        
        // Following Tolis's specification exactly:
        // "Point a((-2.0 * params.inner_vi_ak) * A_rounded.row(params.facet_prev));"
        // "v+=a"
        
        VT a_rounded_row = _A_rounded.row(facet).transpose();
        NT coeff = -2.0 * _param.inner_vi_ak;
        Point a(coeff * a_rounded_row);
        
        if (debug) {
            std::cout << "inner_vi_ak: " << _param.inner_vi_ak << std::endl;
            std::cout << "||a_rounded_row||: " << a_rounded_row.norm() << std::endl;
            std::cout << "||v_before||: " << v.getCoefficients().norm() << std::endl;
        }
        
        // Apply reflection: v += a
        v += a;
        
        if (debug) {
            std::cout << "||v_after||: " << v.getCoefficients().norm() << std::endl;
            std::cout << "=== END CORRECTED REFLECTION ===" << std::endl;
        }
    }

public:

    // CORRECTED: Apply walk with proper error handling
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
            std::cout << "\n=== CORRECTED WALK " << walk_call_count << " START ===" << std::endl;
            std::cout << "Walk length: " << walk_length << std::endl;
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
                // Check feasibility in transformed space
                VT p_transformed = _L_inv.template triangularView<Eigen::Upper>().solve(_p.getCoefficients());
                VT Ap = _A * p_transformed;
                VT slack = _b - Ap;
                int violated = (slack.array() < -1e-10).count();
                std::cout << "Step " << j << " feasibility: " << (violated == 0 ? "OK" : "VIOLATED") 
                         << " (violations: " << violated << "/" << slack.size() << ")" << std::endl;
            }
        }
        
        if (debug_walk) {
            std::cout << "=== CORRECTED WALK " << walk_call_count << " END ===" << std::endl;
        }
        
        p = Point(_p.getCoefficients());
    }

private:

    template<typename GenericPolytope>
    inline void initialize(GenericPolytope &P,
                        Point const& p,
                        RandomNumberGenerator &rng)
    {
        std::cout << "\n=== CORRECTED INITIALIZATION ===" << std::endl;
        std::cout << "Polytope dimension: " << P.dimension() << std::endl;
        std::cout << "Starting point: " << p.getCoefficients().transpose() << std::endl;
        
        // Check if starting point is feasible in transformed space
        VT p_transformed = _L_inv.template triangularView<Eigen::Upper>().solve(p.getCoefficients());
        VT Ap = _A * p_transformed;
        VT slack = _b - Ap;
        std::cout << "Initial feasibility: slack range [" << slack.minCoeff() << ", " << slack.maxCoeff() << "]" << std::endl;
        int violated = (slack.array() < 0).count();
        std::cout << "Violated constraints: " << violated << " out of " << slack.size() << std::endl;
        
        unsigned int n = P.dimension();
        const NT dl = 0.995;
        _lambdas.setZero(P.num_of_hyperplanes());
        _Av.setZero(P.num_of_hyperplanes());
        _p = p;
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
        
        std::cout << "=== END CORRECTED INITIALIZATION ===" << std::endl;
    }

    // Member variables following Tolis's specification
    SparseRowMT _A;                  // Original sparse A matrix (unrounded)
    VT _b;                           // Original b vector  
    SparseMT _L_inv;                 // L_inv where H = L_inv^T * L_inv (upper triangular)
    MT _A_rounded;                   // A_rounded = A * L_inv^(-1) (dense, row-normalized)
    VT _A_rounded_row_norms;         // Row norms of A_rounded before normalization
    
    // Standard billiard walk members
    NT _Len;
    Point _p;
    Point _v;
    NT _lambda_prev;
    VT _lambdas;
    VT _Av;
    parameters _param;
};

};

#endif // RANDOM_WALKS_SPARSE_BILLIARD_WALK_HPP