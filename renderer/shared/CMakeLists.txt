cmake_minimum_required(VERSION 3.12)
SET(MODULE_NAME, "path-prediction-tracer::shared")

file(GLOB_RECURSE SOURCES "src/*.[hc]pp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

add_executable(${MODULE_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${SHARED_LIB_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
