cmake_minimum_required(VERSION 3.12)
SET(MODULE_NAME, "ppt-shared")
project(${MODULE_NAME} VERSION 0.0.1 LANGUAGES CXX)

file(GLOB_RECURSE SOURCES "src/*.[hc]pp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

add_library(${MODULE_NAME} SHARED ${SOURCES} ${HEADERS})

target_include_directories(${SHARED_LIB_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
