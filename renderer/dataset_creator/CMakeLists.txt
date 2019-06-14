cmake_minimum_required(VERSION 3.12)
SET(SOLUTION_NAME "ppt")
SET(MODULE_NAME "ppt-dataset-creator")
SET(MODULE_NAME_CONFIG "ppt-dataset-creator-config")
message(STATUS "Project: " ${MODULE_NAME})

project(${MODULE_NAME}
    VERSION 0.0.1
    LANGUAGES C CXX)

include(GNUInstallDirs)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

add_library(${MODULE_NAME} 
    SHARED ${SOURCES} ${HEADERS})

target_link_libraries (${MODULE_NAME} PUBLIC ppt-shared ppt-path-tracer stdc++fs)

target_compile_features(${MODULE_NAME} 
    PUBLIC cxx_std_14)

target_include_directories(${MODULE_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(${MODULE_NAME} 
    PRIVATE -D_EXPORT)
    
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