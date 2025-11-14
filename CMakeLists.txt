cmake_minimum_required(VERSION 3.20)
project(wfoptim)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3" )
# local include directories
# Eigen include
include_directories(/usr/include/eigen3/Eigen)
# MKL include
#include_directories(/home/legion/code_space_mkl/mkl/mkl/latest/include)

include_directories(${CMAKE_SOURCE_DIR}/include/coin-or)

# link_directories(
#     ${CMAKE_SOURCE_DIR}/external/StaticHSL/lib
#     ${CMAKE_SOURCE_DIR}/external/StaticIPOPT/lib
#     ${CMAKE_SOURCE_DIR}/external/StaticMumps/lib
# )

# MKL lib
# link_directories(/home/legion/code_space_mkl/mkl/mkl/latest/lib/intel64)

# custom dynamic library for windfarmoptimization class
# compile the shared library for once
add_library(wfclass SHARED Turbine.cpp Turbulence.cpp
        combination.cpp Deflection.cpp
        Farm.cpp Rotation.cpp
        Velocity.cpp wrappedfunc.cpp)

# # set shared library output directory
set_target_properties(wfclass PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/internal/lib"
)

# Declare wfclass as an IMPORTED shared library so it can be linked later
# add_library(wfclass STATIC IMPORTED)
# set_target_properties(wfclass PROPERTIES
#         IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/internal/lib/libwfclass.a"
# )

# 链接库
# add_library(hsl0 SHARED IMPORTED)
# set_target_properties(hsl0 PROPERTIES
#         IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/external/hsl/lib/libcoinhsl.so"
# )
# add_library(ipopt0 SHARED IMPORTED)
# set_target_properties(ipopt0 PROPERTIES
#         IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/external/ipopt/lib/libipopt.so"
# )
# add_library(coinmumps0 SHARED IMPORTED)
# set_target_properties(coinmumps0 PROPERTIES
#         IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/external/mumps/lib/libcoinmumps.so."
# )
# add_library(hsl STATIC IMPORTED)
# set_target_properties(hsl PROPERTIES 
#         IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/external/libcoinhsl.a"
# )
# add_library(ipopt STATIC IMPORTED)
# set_target_properties(ipopt PROPERTIES 
#         IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/external/libsolver.a"
# )
# add_library(coinmumps STATIC IMPORTED)
# set_target_properties(coinmumps PROPERTIES 
#         IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/external/libcoinmumps.a"
# )

add_library(hsl0 SHARED IMPORTED)
set_target_properties(hsl0 PROPERTIES 
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/external/libcoinhsl.so"
)
add_library(solver SHARED IMPORTED)
set_target_properties(solver PROPERTIES 
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/external/libsolver.so"
)
add_library(coinmumps0 SHARED IMPORTED)
set_target_properties(coinmumps0 PROPERTIES 
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/external/libcoinmumps.so"
)

# add_executable(wfoptim cpp_example.cpp Turbine.cpp Turbulence.cpp
#          combination.cpp Deflection.cpp
#          Farm.cpp MyNLP.cpp
#          Rotation.cpp Velocity.cpp
#          Toolset.cpp wrappedfunc.cpp
# )

add_executable(wfoptim cpp_example.cpp)
find_package(OpenMP REQUIRED)
target_link_libraries(wfoptim wfclass OpenMP::OpenMP_CXX)
# target_link_libraries(wfoptim hsl INSTALL_RPATH "${LIBDIR}"
#     BUILD_WITH_INSTALL_RPATH TRUE)
target_link_libraries(wfoptim -Wl,--start-group
    solver hsl0
    coinmumps0 metis lapack
    blas
    -Wl,--end-group
    # mkl_intel_lp64
    # mkl_core
    # mkl_sequential
    pthread gfortran m)

# set(DIR1 "${CMAKE_SOURCE_DIR}/external/ipopt/lib")
# set(DIR2 "${CMAKE_SOURCE_DIR}/external/mumps/lib")
# set(DIR3 "${CMAKE_SOURCE_DIR}/external/hsl/lib")

# target_link_libraries(wfoptim ipopt0
#     pthread gfortran m
# )

# set_target_properties(wfoptim PROPERTIES
#     INSTALL_RPATH "${DIR1}:${DIR2}:${DIR3}"
#     BUILD_WITH_INSTALL_RPATH TRUE
# )