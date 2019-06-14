cmake_minimum_required(VERSION 3.12)
SET(SOLUTION_NAME "ppt")
SET(MODULE_NAME "ppt-path-tracer")
SET(MODULE_NAME_CONFIG "ppt-path-tracer-config")
message(STATUS "Project: " ${MODULE_NAME})

project(${MODULE_NAME}
    VERSION 0.0.1
    LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.cuh" "include/*.hpp")

add_library(${MODULE_NAME} 
    SHARED ${SOURCES} ${HEADERS})

target_link_libraries (${MODULE_NAME} PUBLIC ppt-shared)

target_compile_features(${MODULE_NAME} 
    PUBLIC cxx_std_14)

target_include_directories(${MODULE_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_definitions(${MODULE_NAME} 
    PRIVATE -D_EXPORT)
    
set_target_properties(${MODULE_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

install(
    TARGETS ${MODULE_NAME}
    EXPORT ${MODULE_NAME_CONFIG}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    EXPORT ${MODULE_NAME_CONFIG}
    NAMESPACE ${MODULE_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${SOLUTION_NAME})

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${SOLUTION_NAME})