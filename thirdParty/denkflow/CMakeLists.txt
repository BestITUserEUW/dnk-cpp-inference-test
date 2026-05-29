cmake_minimum_required(VERSION 3.29)

project(external-denkflow NONE)

set(ARCH "")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARCH "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(ARCH "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ARCH "x86-64")
endif()

set(VERSION "0.8.1")
set(URL "https://github.com/DENKweitGmbH/DENKflow-C-API/releases/download/${VERSION}")
set(LOC_ARCHIVE_FPATH "${CMAKE_CURRENT_BINARY_DIR}/libdenkflow.zip")

file(DOWNLOAD "${URL}/denkflow.h" "${CMAKE_INSTALL_PREFIX}/include/denkflow.h")

file(DOWNLOAD
    "${URL}/libdenkflow_v${VERSION}_lin_${ARCH}.zip"
    ${LOC_ARCHIVE_FPATH}
)

file(ARCHIVE_EXTRACT
    INPUT ${LOC_ARCHIVE_FPATH}
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
)