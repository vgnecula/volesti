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

NT calculateAndVerifyVolume(HPOLYTOPE& polytope) {

    int walk_len = 10;
    NT e = 0.1;
    RandomNumberGenerator rng;
    NT volume = non_spherical_crhmc_volume_cooling_gaussians<HPOLYTOPE, RandomNumberGenerator>(polytope, rng, e, walk_len);
    return volume;
}

int main() {


    boost::random::mt19937 global_gen(FIXED_SEED);

    HPOLYTOPE cube10 = generate_cube<HPOLYTOPE>(10, false);
    std::cout << "Cube10 \n";
    std::cout << "Calculated Volume: " << calculateAndVerifyVolume(cube10) << "\n";
    std::cout << "Expected Volume: " << std::pow(2, 10) << "\n\n";

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
