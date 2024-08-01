// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2024 Vissarion Fisikopoulos
// Copyright (c) 2018-2024 Apostolos Chalkis
// Copyright (c) 2024 Vladimir Necula

// Contributed and/or modified by Vladimir Necula, as part of Google Summer of
// Code 2024 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "generators/known_polytope_generators.h"
#include "random_walks/random_walks.hpp"
#include "volume/volume_cooling_nonspherical_gaussians_crhmc.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include "misc/misc.h"

const unsigned int FIXED_SEED = 42;

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef BoostRandomNumberGenerator<boost::mt11213b, NT, FIXED_SEED> RandomNumberGenerator;
typedef HPolytope<Point> HPOLYTOPE;

void calculateAndVerifyVolume(HPOLYTOPE& polytope) {

    int walk_len = 10;
    NT e = 0.1;
    RandomNumberGenerator rng;
    std::cout << "Before " << std::endl;
    NT volume = non_spherical_crhmc_volume_cooling_gaussians<HPOLYTOPE, RandomNumberGenerator>(polytope, rng, e, walk_len);
    //NT volume = volume_cooling_gaussians<GaussianCDHRWalk,HPOLYTOPE, RandomNumberGenerator>(polytope, rng, e, walk_len);
    std::cout << "Volume " << volume << std::endl;
}

int main() {
    // Set global seed
    boost::random::mt19937 global_gen(FIXED_SEED);

    Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A(5, 2);
    Eigen::Matrix<NT, Eigen::Dynamic, 1> b(5);

    A << 1.8339, 3.5784,
        -2.2588, 2.7694,
        0.8622, -1.3499,
        -1.3077, 0.7254,
        -0.4336, -0.0631;

    b << 3.4903,
        1.0752,
        0.4714,
        0.7534,
        0.3853;

    HPolytope<Point> Pin(2, A, b);
    
    Pin.print();

    calculateAndVerifyVolume(Pin);
    

/*   
    HPOLYTOPE cube = generate_cube<HPOLYTOPE>(4, false);
    std::cout << std::endl << "Cube: " << std::endl;
    cube.print();
    calculateAndVerifyVolume(cube);

    HPOLYTOPE cross = generate_cross<HPOLYTOPE>(3, false);
    std::cout << std::endl << "Cross: " << std::endl;
    cross.print();
    calculateAndVerifyVolume(cross);

    HPOLYTOPE birkhoff = generate_birkhoff<HPOLYTOPE>(3);
    std::cout << std::endl << "Birkhoff: " << std::endl;
    birkhoff.print();
    calculateAndVerifyVolume(birkhoff);

    HPOLYTOPE simplex = generate_simplex<HPOLYTOPE>(2, false);
    std::cout << std::endl << "Simplex: " << std::endl;
    simplex.print();
    calculateAndVerifyVolume(simplex);

    HPOLYTOPE cube = generate_cube<HPOLYTOPE>(3, false);
    std::cout << std::endl << "Cube: " << std::endl;
    cube.print();
    calculateAndVerifyVolume(cube);
*/
    return 0;
}
