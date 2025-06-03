cmake_minimum_required(VERSION 3.10)
project(SparseBilliardWalkBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable NLP oracles since we don't need them
option(DISABLE_NLP_ORACLES "Disable non-linear oracles (used in collocation)" ON)
if(DISABLE_NLP_ORACLES)
  add_definitions(-DDISABLE_NLP_ORACLES)
endif()

# Include cmake modules from the repository
include("../../external/cmake-files/Eigen.cmake")
GetEigen()

include("../../external/cmake-files/Boost.cmake")
GetBoost()

include("../../external/cmake-files/LPSolve.cmake")
GetLPSolve()

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS random)

# Include directories - updated paths to point to root directory
include_directories(${CMAKE_SOURCE_DIR}/../../include)
include_directories(${CMAKE_SOURCE_DIR}/../../external)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add executable
add_executable(sparse_billiard_benchmark sparse_billiard_walk_benchmark.cpp)

# Link libraries
target_link_libraries(sparse_billiard_benchmark 
    lp_solve
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)

# Compiler flags
target_compile_options(sparse_billiard_benchmark PRIVATE 
    -O3 
    -DEIGEN_NO_DEBUG
    -march=native
    -funroll-loops
)

# Add OpenMP support if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sparse_billiard_benchmark OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(sparse_billiard_benchmark PRIVATE DISABLE_NLP_ORACLES) 