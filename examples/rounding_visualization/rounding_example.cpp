// Fix for the SVD rounding issue in 2D
#include "Eigen/Eigen"
#include <vector>
#include "cartesian_geom/cartesian_kernel.h"
#include "hpolytope.h"
#include "known_polytope_generators.h"
#include "random_walks/random_walks.hpp"
#include "preprocess/inscribed_ellipsoid_rounding.hpp"
#include "preprocess/min_sampling_covering_ellipsoid_rounding.hpp"
#include "preprocess/svd_rounding.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

typedef double NT;
typedef Cartesian <NT> Kernel;
typedef typename Kernel::Point Point;
typedef BoostRandomNumberGenerator<boost::mt19937, NT, 5> RNGType;
typedef HPolytope <Point> HPOLYTOPE;
typedef typename HPOLYTOPE::MT MT;
typedef typename HPOLYTOPE::VT VT;

// Function to save points to a file for visualization
void save_points_to_file(const std::vector<Point>& points, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& p : points) {
        for (int i = 0; i < p.dimension(); ++i) {
            file << p[i] << " ";
        }
        file << std::endl;
    }
    file.close();
}

// Function to generate sample points from a polytope
std::vector<Point> generate_sample_points(HPOLYTOPE& P, int num_points, RNGType& rng) {
    std::vector<Point> points;
    PushBackWalkPolicy push_back_policy;
    std::list<Point> randPoints;
    
    // Get a point in the Chebychev ball
    std::pair<Point, NT> InnerBall = P.ComputeInnerBall();
    Point p = InnerBall.first;
    
    // Generate random points using CDHR walk
    RandomPointGenerator<typename CDHRWalk::Walk<HPOLYTOPE, RNGType>>::apply(P, p, num_points, 10, randPoints, push_back_policy, rng);
    
    // Convert list to vector
    for (const auto& point : randPoints) {
        points.push_back(point);
    }
    
    return points;
}

int main() {
    // Create a skinny polytope (cube with high aspect ratio)
    int dim = 3;  // Changed from 2 to 3 for SVD rounding
    HPOLYTOPE P = generate_skinny_cube<HPOLYTOPE>(dim);
    
    // Initialize random number generator
    RNGType rng(dim);
    
    // Generate sample points before rounding
    std::vector<Point> original_points = generate_sample_points(P, 1000, rng);
    save_points_to_file(original_points, "original_points.txt");
    
    // 1. Apply inscribed ellipsoid rounding
    std::pair<Point, NT> InnerBall = P.ComputeInnerBall();
    auto [T1, shift1, round_val1] = inscribed_ellipsoid_rounding<MT, VT, NT>(P, InnerBall.first);
    std::vector<Point> rounded_points1 = generate_sample_points(P, 1000, rng);
    save_points_to_file(rounded_points1, "inscribed_ellipsoid_points.txt");
    
    // 2. Apply min sampling covering ellipsoid rounding
    P = generate_skinny_cube<HPOLYTOPE>(dim);  // Reset polytope
    InnerBall = P.ComputeInnerBall();
    auto [T2, shift2, round_val2] = min_sampling_covering_ellipsoid_rounding<typename CDHRWalk::Walk<HPOLYTOPE, RNGType>, MT, VT>(P, InnerBall, 10 + 10 * dim, rng);
    std::vector<Point> rounded_points2 = generate_sample_points(P, 1000, rng);
    save_points_to_file(rounded_points2, "min_sampling_points.txt");
    
    // 3. Apply SVD rounding
    P = generate_skinny_cube<HPOLYTOPE>(dim);  // Reset polytope
    InnerBall = P.ComputeInnerBall();
    int num_rounding_steps = 1;
    auto [T3, shift3, round_val3] = svd_rounding<typename CDHRWalk::Walk<HPOLYTOPE, RNGType>, MT, VT>(P, InnerBall, num_rounding_steps, rng);
    std::vector<Point> rounded_points3 = generate_sample_points(P, 1000, rng);
    save_points_to_file(rounded_points3, "svd_points.txt");
    
    
    std::cout << "Sample points have been saved to files for visualization." << std::endl;
    std::cout << "You can use Python with matplotlib to visualize these points." << std::endl;
    
    return 0;
}