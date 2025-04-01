// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2018 Vissarion Fisikopoulos
// Copyright (c) 2018 Apostolos Chalkis

// Licensed under GNU LGPL.3, see LICENCE file

// VolEsti example

#include "Eigen/Eigen"
//#define VOLESTI_DEBUG
#include <fstream>
#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "random_walks/random_walks.hpp"

#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_balls.hpp"

#include "volume/exact_vols.h"
#include "generators/known_polytope_generators.h"

#include <fstream>
#include <iostream>
#include <chrono>
#include <random>

#include "misc/misc.h"
#include "cartesian_geom/cartesian_kernel.h"
#include "convex_bodies/hpolytope.h"

#include "generators/h_polytopes_generator.h"

int main()
{
    typedef double NT;
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef boost::mt19937    RNGType;
    typedef HPolytope<Point> Hpolytope;
    typedef VPolytope<Point> Vpolytope;

    std::cout << "Volume algorithm: Cooling Balls" << std::endl << std::endl;

    Hpolytope HP = generate_cube<Hpolytope>(10, false);

    //Compute chebychev ball
    std::pair<Point,NT> CheBall;
    CheBall = HP.ComputeInnerBall();

    // Setup the parameters
    int n = HP.dimension();
    int walk_len=10 + n/10;
    int n_threads=1;
    NT e=1, err=0.1;
    NT C=2.0,ratio,frac=0.1,delta=-1.0;
    int N = 500 * ((int) C) + ((int) (n * n / 2));
    int W = 6*n*n+800;
    ratio = 1.0-1.0/(NT(n));

    int rnum = std::pow(e,-2) * 400 * n * std::log(n);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    RNGType rng(seed);
    boost::normal_distribution<> rdist(0,1);
    boost::random::uniform_real_distribution<>(urdist);
    boost::random::uniform_real_distribution<> urdist1(-1,1);

    double tstart;

    ////////////////////////////////////////////////////////////////
    /// H-Polytopes
    ///
    ///
    ///

    std::cout << "Volume estimation on H-polytopes (cube-10)" << std::endl;

    // Estimate the volume

    typedef BoostRandomNumberGenerator<boost::mt11213b, NT> RNG;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "BallWalk (cube) = "
              << volume_cooling_balls<BallWalk, RNG>(HP, e, walk_len).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;
    tstart = (double)clock()/(double)CLOCKS_PER_SEC;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "RDHRWalk (cube) = "
              << volume_cooling_balls<RDHRWalk, RNG>(HP, e, walk_len).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "CDHRWalk (cube) = "
              << volume_cooling_balls<CDHRWalk, RNG>(HP, e, walk_len).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "BirdWalk (cube) = "
              << volume_cooling_balls<BilliardWalk, RNG>(HP, e, walk_len).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

#ifdef VOLESTI_OLD_IMPLEMENTATION

    // OLD Implementation

    NT diameter;
    HP.comp_diam(diameter, CheBall.second);

    NT lb = 0.1, ub = 0.15, p = 0.75, rmax = 0.0, alpha = 0.2;
    int W2 = 500, NNu = 150, nu =10;
    bool win2 = false;
    vars_ban <NT> var_ban(lb, ub, p, rmax, alpha, W2, NNu, nu, win2);

    {
        vars<NT, RNGType> var(rnum,n,walk_len,n_threads,err,e,0,0,0,
                              CheBall.second,diameter,rng,
                              urdist,urdist1,-1.0,false,false,false,
                              false,false,true,false,false,false);

        tstart = (double)clock()/(double)CLOCKS_PER_SEC;
        std::cout << "OLD Ball = " << vol_cooling_balls(HP, var, var_ban, CheBall)
                  << " , ";
        std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;
    }
    {
        vars<NT, RNGType> var(rnum,n,walk_len,n_threads,err,e,0,0,0,
                              CheBall.second,diameter,rng,
                              urdist,urdist1,-1.0,false,false,false,
                              false,false,false,true,false,false);

        tstart = (double)clock()/(double)CLOCKS_PER_SEC;
        std::cout << "OLD RDHR = " << vol_cooling_balls(HP, var, var_ban, CheBall)
                  << " , ";
        std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;
    }
    {
        vars<NT, RNGType> var(rnum,n,walk_len,n_threads,err,e,0,0,0,
                              CheBall.second,diameter,rng,
                              urdist,urdist1,-1.0,false,false,false,
                              false,false,false,false,true,false);

        tstart = (double)clock()/(double)CLOCKS_PER_SEC;
        std::cout << "OLD CDHR = " << vol_cooling_balls(HP, var, var_ban, CheBall)
                  << " , ";
        std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;
    }
    {
        vars<NT, RNGType> var(rnum,n,walk_len,n_threads,err,e,0,0,0,
                              CheBall.second,diameter,rng,
                              urdist,urdist1,-1.0,false,false,false,
                              false,false,false,false,false,true);

        tstart = (double)clock()/(double)CLOCKS_PER_SEC;
        std::cout << "OLD Blrd = " << vol_cooling_balls(HP, var, var_ban, CheBall)
                  << " , ";
        std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;
    }
#endif

    ////////////////////////////////////////////////////////////////
    /// V-Polytopes
    ///
    ///
    ///

    std::cout << "Volume estimation on V-polytopes (cross-10)" << std::endl;

    Vpolytope VP = generate_cross<Vpolytope>(10, true);

    // NEW IMPLEMENTATIOM

    // Estimate the volume

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Ball (cross) = "
              << volume_cooling_balls<BallWalk, RNG>(VP).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "RDHR (cross) = "
              << volume_cooling_balls<RDHRWalk, RNG>(VP).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "CDHR (cross) = "
              << volume_cooling_balls<CDHRWalk, RNG>(VP).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Blrd (cross) = "
              << volume_cooling_balls<BilliardWalk, RNG>(VP).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    // Run the line intersection benchmark
    std::cout << "\n--- Benchmarking line intersection for sparse H-polytope" << std::endl;
    
    // Create a sparse random matrix
    const int dim = 100;
    const int num_constraints = 200;
    Eigen::SparseMatrix<NT> A(num_constraints, dim);
    A.reserve(Eigen::VectorXi::Constant(num_constraints, 5)); // Reserve space for 5 non-zeros per row
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<NT> dis(-1.0, 1.0);
    std::uniform_int_distribution<int> row_dis(0, num_constraints-1);
    std::uniform_int_distribution<int> col_dis(0, dim-1);
    
    // Fill with random non-zero elements (about 5% sparsity)
    for(int i = 0; i < num_constraints * dim / 20; i++) {
        int row = row_dis(gen);
        int col = col_dis(gen);
        A.insert(row, col) = dis(gen);
    }
    A.makeCompressed();
    
    // Create right-hand side vector
    Eigen::VectorXd b(num_constraints);
    for(int i = 0; i < num_constraints; i++) {
        b(i) = dis(gen);
    }
    
    // Create H-polytope from sparse matrix
    Hpolytope P(dim, A, b);
    P.normalize();

    // Generate random points and directions for testing
    const int num_tests = 1000;
    std::vector<Point> points(num_tests);
    std::vector<Point> directions(num_tests);
    
    for(int i = 0; i < num_tests; i++) {
        std::vector<NT> coords(dim);
        for(int j = 0; j < dim; j++) {
            coords[j] = dis(gen);
        }
        points[i] = Point(dim, coords);
        
        for(int j = 0; j < dim; j++) {
            coords[j] = dis(gen);
        }
        directions[i] = Point(dim, coords);
    }

    std::cout << "Polytope dimensions: " << P.dimension() << "D, " << P.num_of_hyperplanes() << " constraints" << std::endl;
    std::cout << "Matrix sparsity: " << (double)A.nonZeros() / (A.rows() * A.cols()) * 100 << "%" << std::endl;
    std::cout << "Number of tests: " << num_tests << std::endl << std::endl;

    // Benchmark dense version
    std::cout << "Testing dense line intersection..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < num_tests; i++) {
        P.line_intersect(points[i], directions[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto dense_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Dense version took: " << dense_duration.count() << " microseconds" << std::endl;
    std::cout << "Average time per test: " << (double)dense_duration.count() / num_tests << " microseconds" << std::endl << std::endl;
    
    // Benchmark sparse version
    std::cout << "Testing sparse line intersection..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < num_tests; i++) {
        P.sparse_line_intersect(points[i], directions[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto sparse_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Sparse version took: " << sparse_duration.count() << " microseconds" << std::endl;
    std::cout << "Average time per test: " << (double)sparse_duration.count() / num_tests << " microseconds" << std::endl << std::endl;

    std::cout << "Performance comparison:" << std::endl;
    std::cout << "Speedup: " << (double)dense_duration.count() / sparse_duration.count() << "x" << std::endl;
    std::cout << "Time saved per test: " << (double)(dense_duration.count() - sparse_duration.count()) / num_tests << " microseconds" << std::endl;

    return 0;
}
