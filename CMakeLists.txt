cmake_minimum_required(VERSION 3.20)
project(wfoptim)

set(CMAKE_CXX_STANDARD 14)

# local include directories
# Eigen include
include_directories(/usr/include/eigen3/Eigen)

include_directories(${CMAKE_SOURCE_DIR}/external/StaticHSL/include)
include_directories(${CMAKE_SOURCE_DIR}/external/StaticIPOPT/include/coin-or)
include_directories(${CMAKE_SOURCE_DIR}/external/StaticMumps/include)

link_directories(
    ${CMAKE_SOURCE_DIR}/external/StaticHSL/lib
    ${CMAKE_SOURCE_DIR}/external/StaticIPOPT/lib
    ${CMAKE_SOURCE_DIR}/external/StaticMumps/lib
)

# custom dynamic library for windfarmoptimization class
# compile the shared library for once
# add_library(wfclass STATIC Turbine.cpp Turbulence.cpp
#         combination.cpp Deflection.cpp
#         Farm.cpp MyNLP.cpp
#         Rotation.cpp Velocity.cpp
#         Toolset.cpp wrappedfunc.cpp)

# # set static library output directory
# set_target_properties(wfclass PROPERTIES
#         ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/internal/lib"
# )

# Declare wfclass as an IMPORTED shared library so it can be linked later
# add_library(wfclass STATIC IMPORTED)
# set_target_properties(wfclass PROPERTIES
#         IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/internal/lib/libwfclass.a"
# )

add_executable(wfoptim cpp_example.cpp Turbine.cpp Turbulence.cpp
         combination.cpp Deflection.cpp
         Farm.cpp MyNLP.cpp
         Rotation.cpp Velocity.cpp
         Toolset.cpp wrappedfunc.cpp
)
find_package(OpenMP REQUIRED)
target_link_libraries(wfoptim OpenMP::OpenMP_CXX)
target_link_libraries(wfoptim -Wl,--start-group
    ipopt coinhsl
    coinmumps lapack
    blas
    -Wl,--end-group
    pthread gfortran m)