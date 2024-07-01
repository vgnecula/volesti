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

template <int simdLen>
void calculateAndVerifyVolume(HPOLYTOPE& polytope) {
    int walk_len = 100;
    NT e = 0.1;

    NT volume = volume_cooling_gaussians<CRHMCWalk, RNGType, HPOLYTOPE, 4>(polytope, e, walk_len);

    std::cout << "Volume " << volume << std::endl;
}

int main() {
    
    HPOLYTOPE HP = generate_simplex<HPOLYTOPE>(2,false);
    std::cout << "HPoly: " << std::endl;
    HP.print();
    calculateAndVerifyVolume<4>(HP);

    return 0;
}