cmake_minimum_required(VERSION 3.12)
SET(SOLUTION_NAME "ppt")
SET(MODULE_NAME "ppt-app")
SET(MODULE_NAME_CONFIG "ppt-app-config")
message(STATUS "Project: " ${MODULE_NAME})

project(${MODULE_NAME}
    VERSION 0.0.1
    LANGUAGES C CXX)

include(GNUInstallDirs)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

add_executable(${MODULE_NAME} ${SOURCES})

target_link_libraries (${MODULE_NAME} 
    PUBLIC
    ppt-shared
    ppt-path-tracer 
    ppt-dataset-creator
    GL
    GLU
    glfw
    GLEW::GLEW
    stb
    imgui)

target_compile_features(${MODULE_NAME} 
    PUBLIC cxx_std_17)

target_include_directories(${MODULE_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(${MODULE_NAME} 
    PRIVATE -D_EXPORT)

