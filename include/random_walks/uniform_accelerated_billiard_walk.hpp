// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis

// Contributed and/or modified by Alexandros Manochis, as part of Google Summer of Code 2020 program.
// Contributed and/or modified by Luca Perju, as part of Google Summer of Code 2024 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef RANDOM_WALKS_ACCELERATED_IMPROVED_BILLIARD_WALK_HPP
#define RANDOM_WALKS_ACCELERATED_IMPROVED_BILLIARD_WALK_HPP

#include "sampling/sphere.hpp"
#include <Eigen/Eigen>


// Billiard walk which accelarates each step for uniform distribution

struct AcceleratedBilliardWalk
{
    AcceleratedBilliardWalk(double L)
            :   param(L, true)
    {}

    AcceleratedBilliardWalk()
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

    struct update_parameters
    {
        update_parameters()
                :   facet_prev(0), hit_ball(false), inner_vi_ak(0.0), ball_inner_norm(0.0)
        {}
        int facet_prev;
        bool hit_ball;
        double inner_vi_ak;
        double ball_inner_norm;
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
        typedef typename Eigen::Matrix<typename Point::FT, Eigen::Dynamic, Eigen::Dynamic> MT;
        typedef typename Point::FT NT;

        template <typename GenericPolytope>
        Walk(GenericPolytope &P, Point const& p, RandomNumberGenerator &rng)
        {
            if(!P.is_normalized()) {
                P.normalize();
            }
            _update_parameters = update_parameters();
            _L = compute_diameter<GenericPolytope>
                ::template compute<NT>(P);
            _AA.noalias() = (MT)(P.get_mat() * P.get_mat().transpose());
            _rho = 1000 * P.dimension(); // upper bound for the number of reflections (experimental)
            initialize(P, p, rng);
        }

        template <typename GenericPolytope>
        Walk(GenericPolytope &P, Point const& p, RandomNumberGenerator &rng,
             parameters const& params)
        {
            if(!P.is_normalized()) {
                P.normalize();
            }
            _update_parameters = update_parameters();
            _L = params.set_L ? params.m_L
                              : compute_diameter<GenericPolytope>
                                ::template compute<NT>(P);
            _AA.noalias() = (MT)(P.get_mat() * P.get_mat().transpose());
            _rho = 1000 * P.dimension(); // upper bound for the number of reflections (experimental)
            initialize(P, p, rng);
        }

        template
                <
                        typename GenericPolytope
                >
        inline void apply(GenericPolytope &P,
                          Point &p,   // a point to start
                          unsigned int const& walk_length,
                          RandomNumberGenerator &rng)
        {
            unsigned int n = P.dimension();
            NT T;
            const NT dl = 0.995;
            int it;

            for (auto j=0u; j<walk_length; ++j)
            {
                T = -std::log(rng.sample_urdist()) * _L;
                _v = GetDirection<Point>::apply(n, rng);
                Point p0 = _p;

                it = 0;
                std::pair<NT, int> pbpair;
                if(!was_reset) {
                    pbpair = P.line_positive_intersect(_p, _v, _lambdas, _Av, _lambda_prev, _update_parameters);
                } else {
                    pbpair = P.line_first_positive_intersect(_p, _v, _lambdas, _Av, _update_parameters);
                    was_reset = false;
                }
                
                if (T <= pbpair.first) {
                    _p += (T * _v);
                    _lambda_prev = T;
                    continue;
                }

                _lambda_prev = dl * pbpair.first;
                _p += (_lambda_prev * _v);
                T -= _lambda_prev;
                P.compute_reflection(_v, _p, _update_parameters);
                it++;

                while (it < _rho)
                {
                    std::pair<NT, int> pbpair
                            = P.line_positive_intersect(_p, _v, _lambdas, _Av, _lambda_prev,
                                                        _AA, _update_parameters);
                    if (T <= pbpair.first) {
                        _p += (T * _v);
                        _lambda_prev = T;
                        break;
                    }
                    _lambda_prev = dl * pbpair.first;
                    _p += (_lambda_prev * _v);
                    T -= _lambda_prev;
                    P.compute_reflection(_v, _p, _update_parameters);
                    it++;
                }
                if (it == _rho) {
                    _p = p0;
                    was_reset = true;
                }
            }
            p = _p;
        }


        template
        <
            typename GenericPolytope
        >
        inline void get_starting_point(GenericPolytope &P,
                           Point const& center,
                           Point &q,
                           unsigned int const& walk_length,
                           RandomNumberGenerator &rng)
        {
            unsigned int n = P.dimension();
            NT radius = P.InnerBall().second;

            q = GetPointInDsphere<Point>::apply(n, radius, rng);
            q += center;
            initialize(P, q, rng);

            apply(P, q, walk_length, rng);
        }


        template
        <
            typename GenericPolytope
        >
        inline void parameters_burnin(GenericPolytope &P, 
                                     Point const& center,
                                     unsigned int const& num_points,
                                     unsigned int const& walk_length,
                                     RandomNumberGenerator &rng)
        {
            Point p(P.dimension());
            std::vector<Point> pointset;
            pointset.push_back(center);
            pointset.push_back(_p);
            NT rad = NT(0), max_dist, Lmax = get_delta(), radius = P.InnerBall().second;

            for (int i = 0; i < num_points; i++) 
            {
                Point p = GetPointInDsphere<Point>::apply(P.dimension(), radius, rng);
                p += center;
                initialize(P, p, rng);

                apply(P, p, walk_length, rng);
                max_dist = get_max_distance(pointset, p, rad);
                if (max_dist > Lmax) {
                    Lmax = max_dist;
                }
                if (2.0*rad > Lmax) {
                    Lmax = 2.0 * rad;
                }
                pointset.push_back(p);
            }

            if (Lmax > _L) {
                if (P.dimension() <= 2500) {
                    update_delta(Lmax);
                }
                else{
                    update_delta(2.0 * get_delta());
                }
            }
            pointset.clear();
        }

        

        inline void update_delta(NT L)
        {
            _L = L;
        }

        NT get_delta()
        {
            return _L;
        }

    private :

        template
                <
                        typename GenericPolytope
                >
        inline void initialize(GenericPolytope &P,
                               Point const& p,
                               RandomNumberGenerator &rng)
        {
            unsigned int n = P.dimension();
            const NT dl = 0.995;
            was_reset = false;
            _lambdas.setZero(P.num_of_hyperplanes());
            _Av.setZero(P.num_of_hyperplanes());
            _p = p;
            _v = GetDirection<Point>::apply(n, rng);

            NT T = -std::log(rng.sample_urdist()) * _L;
            Point p0 = _p;
            int it = 0;

            std::pair<NT, int> pbpair
                    = P.line_first_positive_intersect(_p, _v, _lambdas, _Av, _update_parameters);
            if (T <= pbpair.first) {
                _p += (T * _v);
                _lambda_prev = T;
                return;
            }
            _lambda_prev = dl * pbpair.first;
            _p += (_lambda_prev * _v);
            T -= _lambda_prev;
            P.compute_reflection(_v, _p, _update_parameters);

            while (it <= _rho)
            {
                std::pair<NT, int> pbpair
                        = P.line_positive_intersect(_p, _v, _lambdas, _Av, _lambda_prev, _AA, _update_parameters);
                if (T <= pbpair.first) {
                    _p += (T * _v);
                    _lambda_prev = T;
                    break;
                } else if (it == _rho) {
                    _lambda_prev = rng.sample_urdist() * pbpair.first;
                    _p += (_lambda_prev * _v);
                    break;
                }
                _lambda_prev = dl * pbpair.first;
                _p += (_lambda_prev * _v);
                T -= _lambda_prev;
                P.compute_reflection(_v, _p, _update_parameters);
                it++;
            }
        }

        inline double get_max_distance(std::vector<Point> &pointset, Point const& q, double &rad) 
        {
            double dis = -1.0, temp_dis;
            int jj = 0;
            for (auto vecit = pointset.begin(); vecit!=pointset.end(); vecit++, jj++) 
            {
                temp_dis = (q.getCoefficients() - (*vecit).getCoefficients()).norm();
                if (temp_dis > dis) {
                    dis = temp_dis;
                }
                if (jj == 0) {
                    if (temp_dis > rad) {
                        rad = temp_dis;
                    }
                }
            }
            return dis;
        }

        double _L;
        Point _p;
        Point _v;
        NT _lambda_prev;
        MT _AA;
        unsigned int _rho;
        update_parameters _update_parameters;
        typename Point::Coeff _lambdas;
        typename Point::Coeff _Av;
        bool was_reset;
    };

};


#endif
