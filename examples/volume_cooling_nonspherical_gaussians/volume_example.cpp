#include "generators/known_polytope_generators.h"
#include "random_walks/random_walks.hpp"
#include "volume/volume_cooling_nonspherical_gaussians_crhmc.hpp"
#include <iostream>
#include <fstream>
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
    NT volume = non_spherical_crhmc_volume_cooling_gaussians<HPOLYTOPE, RandomNumberGenerator>(polytope, rng, e, walk_len);
    //NT volume = volume_cooling_gaussians<GaussianCDHRWalk,HPOLYTOPE, RandomNumberGenerator>(polytope, rng, e, walk_len);
    std::cout << "Volume " << volume << std::endl;
}

int main() {
    // Set global seed
    boost::random::mt19937 global_gen(FIXED_SEED);

    HPOLYTOPE cube = generate_cube<HPOLYTOPE>(4, false);
    std::cout << std::endl << "Cube: " << std::endl;
    cube.print();
    calculateAndVerifyVolume(cube);

/*   
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