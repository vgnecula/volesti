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
#include "volume/volume_cooling_gaussians.hpp"
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

NT nonspherical_crhmc_volume(HPOLYTOPE& polytope) {
    int walk_len = 10;
    NT e = 0.1;
    RandomNumberGenerator rng;
    // NT volume = volume_cooling_gaussians<GaussianBallWalk, RandomNumberGenerator>(polytope, e, walk_len);
    NT volume = non_spherical_crhmc_volume_cooling_gaussians<HPOLYTOPE, RandomNumberGenerator>(polytope, rng, e, walk_len);
    return volume;
}

NT spherical_gaussians_volume(HPOLYTOPE& polytope) {
    int walk_len = 10;
    NT e = 0.1;
    RandomNumberGenerator rng;
    NT volume = volume_cooling_gaussians<GaussianCDHRWalk, RandomNumberGenerator>(polytope, e, walk_len);
    return volume;
}

int main() {


    boost::random::mt19937 global_gen(FIXED_SEED);

    HPOLYTOPE cube3 = generate_cube<HPOLYTOPE>(3, false);
    std::cout << "Cube3 \n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With Gaussian CDHR: " << spherical_gaussians_volume(cube3) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With CRHMC: " << nonspherical_crhmc_volume(cube3) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Expected Volume: " << std::pow(2, 3) << "\n\n";

    HPOLYTOPE cube4 = generate_cube<HPOLYTOPE>(4, false);
    std::cout << "Cube4 \n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With Gaussian CDHR: " << spherical_gaussians_volume(cube4) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With CRHMC: " << nonspherical_crhmc_volume(cube4) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Expected Volume: " << std::pow(2, 4) << "\n\n";

    HPOLYTOPE skinnycube3 = generate_skinny_cube<HPOLYTOPE>(3, false);
    std::cout << "SkinnyCube3 \n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With Gaussian CDHR: " << spherical_gaussians_volume(skinnycube3) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With CRHMC: " << nonspherical_crhmc_volume(skinnycube3) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Expected Volume: " << 200 * std::pow(2, 2) << "\n\n";


    HPOLYTOPE cube5 = generate_cube<HPOLYTOPE>(5, false);
    std::cout << "Cube5 \n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With Gaussian CDHR: " << spherical_gaussians_volume(cube5) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With CRHMC: " << nonspherical_crhmc_volume(cube5) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Expected Volume: " << std::pow(2, 5) << "\n\n";

    //--------------------------------------------------------------------------------
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

    std::cout << "Random hploy \n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With Gaussian CDHR: " << spherical_gaussians_volume(Pin) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Calculated Volume With CRHMC: " << nonspherical_crhmc_volume(Pin) << "\n";

/*
    HPOLYTOPE skinnycube10 = generate_skinny_cube<HPOLYTOPE>(10);
    std::cout << "SkinnyCube10 \n";
    std::cout << "Calculated Volume: " << calculateAndVerifyVolume(skinnycube10) << "\n";
    std::cout << "Expected Volume: " << 200 * std::pow(2, 9) << "\n\n";

    HPOLYTOPE cube50 = generate_cube<HPOLYTOPE>(50, false);
    std::cout << "Cube50 \n";
    std::cout << "Calculated Volume: " << calculateAndVerifyVolume(cube50) << "\n";
    std::cout << "Expected Volume: " << std::pow(2, 50) << "\n\n";

    HPOLYTOPE skinnycube50 = generate_skinny_cube<HPOLYTOPE>(50);
    std::cout << "SkinnyCube50 \n";
    std::cout << "Calculated Volume: " << calculateAndVerifyVolume(skinnycube50) << "\n";
    std::cout << "Expected Volume: " << 200 * std::pow(2, 49) << "\n\n";
*/
    return 0;
}
