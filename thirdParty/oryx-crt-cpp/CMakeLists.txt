cmake_minimum_required(VERSION 3.29)

project(external-oryx-crt-cpp NONE)
include(ExternalProject)

list(APPEND CMAKE_ARGS
    "-DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"    
    "-DCMAKE_CXX_STANDARD=23"
    "-DORYX_CRT_BUILD_SHARED_LIBS=OFF"
    "-DORYX_CRT_INSTALL=ON"
)

ExternalProject_add(
    oryx-crt-cpp
    PREFIX oryx-crt-cpp
    GIT_REPOSITORY https://github.com/BestITUserEUW/oryx-crt-cpp.git
    GIT_TAG v0.4.0
    CMAKE_ARGS "${CMAKE_ARGS}"
    GIT_SHALLOW ON
)