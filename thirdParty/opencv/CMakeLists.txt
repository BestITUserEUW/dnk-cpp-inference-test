cmake_minimum_required(VERSION 3.29)

project(external-opencv NONE)
include(ExternalProject)

set(OCV_VERSION 4.13.0)

list(APPEND CMAKE_ARGS
    "-DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DBUILD_LIST=core,highgui,improc,imgcodecs,videoio"
    "-DBUILD_SHARED_LIBS=OFF"
    "-DOPENCV_ENABLE_NONFREE=ON"
    "-DENABLE_FAST_MATH=ON"
    "-DWITH_JPEG=ON"
    "-DWITH_FFMPEG=ON"
    "-DWITH_PROTOBUF=OFF"
    "-DWITH_ADE=OFF"
)

if(DEFINED CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND CMAKE_ARGS "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
endif()

message(STATUS "Preparing project ${PROJECT_NAME} with args:")

foreach(arg ${CMAKE_ARGS})
    message(STATUS "-- ${arg}")
endforeach()

ExternalProject_add(
    opencv
    PREFIX opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG ${OCV_VERSION}
    CMAKE_ARGS "${CMAKE_ARGS}"
    GIT_SHALLOW ON
)