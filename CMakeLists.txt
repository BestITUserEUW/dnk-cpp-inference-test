cmake_policy(SET CMP0146 OLD)
cmake_minimum_required(VERSION 3.24)

project(dnk-inference-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(DEPS_BUILD_PATH "${PROJECT_BINARY_DIR}/thirdParty" CACHE PATH "Install path for the dependencies")
set(DEPS_INSTALL_PATH "${DEPS_BUILD_PATH}/install" CACHE PATH "Install path for the dependencies")

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

link_directories(
    ${DEPS_INSTALL_PATH}/lib
)

add_subdirectory(thirdParty)

find_package(OpenCV REQUIRED)
find_package(oryx-crt-cpp REQUIRED)

find_library(LIB_DENKFLOW denkflow 
    HINTS 
        "${CMAKE_SOURCE_DIR}/external/lib"
)

add_executable(${PROJECT_NAME}
    src/main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/external/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Threads::Threads
    oryx::oryx-crt-cpp
    opencv_core 
    opencv_imgproc 
    opencv_highgui
    ${LIB_DENKFLOW}
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall 
    -Wextra 
    -Wuninitialized 
    -Wno-unused-function 
    -Wunused-variable
    $<$<CONFIG:Release>:-g0> 
    $<$<CONFIG:Release>:-O3>
)