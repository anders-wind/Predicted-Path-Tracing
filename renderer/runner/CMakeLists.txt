cmake_minimum_required(VERSION 3.12)
SET(MODULE_NAME "ppt-runner")
SET(MODULE_NAME_CONFIG "ppt-runner-config")
message(STATUS "Project: " ${MODULE_NAME})

project(${MODULE_NAME}
    VERSION 0.0.1
    LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.cuh" "include/*.hpp")

add_executable(${MODULE_NAME} ${SOURCES})

target_link_libraries (${MODULE_NAME} PUBLIC ppt-shared ppt-path-tracer glut GLU GL)

target_compile_features(${MODULE_NAME} 
    PUBLIC cxx_std_14)

target_include_directories(${MODULE_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(${MODULE_NAME} 
    PRIVATE -D_EXPORT)
    
set_target_properties(${MODULE_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
