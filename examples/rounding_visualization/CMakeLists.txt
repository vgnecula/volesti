# VolEsti (volume computation and sampling library)
# Copyright (c) 2012-2024 Vissarion Fisikopoulos
# Copyright (c) 2018-2024 Apostolos Chalkis
# Licensed under GNU LGPL.3, see LICENCE file

cmake_minimum_required(VERSION 3.11)

project(VolEsti)

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(DISABLE_NLP_ORACLES "Disable non-linear oracles (used in collocation)" ON)
if(DISABLE_NLP_ORACLES)
  add_definitions(-DDISABLE_NLP_ORACLES)
endif()

# Eigen
option(BUILTIN_EIGEN "Use eigen from ../../external" OFF)
include("../../external/cmake-files/Eigen.cmake")
GetEigen()

if(CMAKE_VERSION VERSION_LESS "3.12.0")
  add_compile_options(-DEIGEN_NO_DEBUG)
else()
  add_compile_definitions(EIGEN_NO_DEBUG)
endif()

# Boost
include("../../external/cmake-files/Boost.cmake")
GetBoost()

# lp_solve
include("../../external/cmake-files/LPSolve.cmake")
GetLPSolve()

# Include directories
include_directories(BEFORE
  ../../external
  ../../external/minimum_ellipsoid
  ../../include
  ../../include/convex_bodies
  ../../include/random_walks
  ../../include/annealing
  ../../include/ode_solvers
  ../../include/root_finders
  ../../include/samplers
  ../../include/lp_oracles
  ../../include/nlp_oracles
  ../../include/misc
  ../../include/optimization
  ../../include/preprocess
  ../../include/volume
  ../../include/generators
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(rounding_example rounding_example.cpp)

# Compiler optimization
target_compile_options(rounding_example PRIVATE -O3)

# Define macros
target_compile_definitions(rounding_example PRIVATE BOOST_NO_AUTO_PTR)

# Link libraries
target_link_libraries(rounding_example PUBLIC lp_solve)
