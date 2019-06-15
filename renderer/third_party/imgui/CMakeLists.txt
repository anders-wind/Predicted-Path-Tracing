cmake_minimum_required(VERSION 3.12)
SET(MODULE_NAME "imgui")
SET(MODULE_NAME_CONFIG "imgui-config")
message(STATUS "Project: " ${MODULE_NAME})

project(${MODULE_NAME}
    VERSION 0.0.1
    LANGUAGES C CXX)

include(GNUInstallDirs)

file(GLOB_RECURSE SOURCES "include/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")
list(REMOVE_ITEM SOURCES "include/main.cpp")

add_library(${MODULE_NAME} 
    SHARED ${SOURCES} ${HEADERS})

target_compile_features(${MODULE_NAME} 
    PUBLIC cxx_std_14)

target_include_directories(${MODULE_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(${MODULE_NAME} 
    PRIVATE -D_EXPORT)
    
target_link_libraries(${MODULE_NAME} glfw)

# install(
#     TARGETS ${MODULE_NAME}
#     EXPORT ${MODULE_NAME_CONFIG}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# install(
#     EXPORT ${MODULE_NAME_CONFIG}
#     NAMESPACE ${MODULE_NAME}::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${MODULE_NAME})

# install(
#     DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
#     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${MODULE_NAME})