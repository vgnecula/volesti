#include "Eigen/Eigen"
#include <vector>
#include "cartesian_geom/cartesian_kernel.h"
#include "convex_bodies/hpolytope.h"
#include "generators/known_polytope_generators.h"

#include "random_walks/random_walks.hpp"
#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_gaussians_crhmc.hpp"
#include "volume/volume_cooling_balls.hpp"

#include <iostream>
#include <fstream>
#include "misc/misc.h"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef BoostRandomNumberGenerator<boost::mt19937, NT, 3> RNGType;
typedef HPolytope<Point> HPOLYTOPE;

void calculateAndVerifyVolume(HPOLYTOPE& polytope, const std::string& description, NT expectedVolume, NT epsilon) {
    int walk_len = 100;
    NT e = 0.1;

    std::cout << "Calculating volume for " << description << ":\n";
    NT volume = volume_cooling_gaussians<GaussianHamiltonianMonteCarloExactWalk, RNGType>(polytope, e, walk_len);

    std::cout << "Estimated volume: " << volume << "\n";
    std::cout << "Expected volume: " << expectedVolume << "\n";
    NT error = std::abs(volume - expectedVolume);
    if (error <= epsilon) {
        std::cout << "Result is within the acceptable error margin (" << epsilon << ").\n\n";
    } else {
        std::cout << "Error (" << error << ") is larger than acceptable margin (" << epsilon << ").\n\n";
    }
}

int main() {
    const NT epsilon = 0.1;

    // 3-dimensional cube
    HPOLYTOPE cube = generate_cube<HPOLYTOPE>(3, false);
    calculateAndVerifyVolume(cube, "3-dimensional cube", 8.0, epsilon);

    // 3-dimensional cross polytope
    HPOLYTOPE crossPolytope = generate_cross<HPOLYTOPE>(3, false);
    calculateAndVerifyVolume(crossPolytope, "3-dimensional cross polytope", 4.0 / 6.0, epsilon);

    // 3-dimensional simplex
    HPOLYTOPE simplex = generate_simplex<HPOLYTOPE>(3, false);
    calculateAndVerifyVolume(simplex, "3-dimensional simplex", 1.0 / 6.0, epsilon);

    // birkhoff polytope dimension 3 
    HPOLYTOPE birkhoffPolytope = generate_birkhoff<HPOLYTOPE>(3);
    calculateAndVerifyVolume(birkhoffPolytope, "Birkhoff polytope (dim 3)", -1, epsilon);  // Theoretical volume not easily available

    return 0;
}