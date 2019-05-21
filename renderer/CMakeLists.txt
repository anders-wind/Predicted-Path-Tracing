CMAKE_MINIMUM_REQUIRED_VERSION(3.14)
project(renderer LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    # Set up wrapper scripts
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    message("Using ccache")
else()
    message("Cannot find ccache")
endif()

add_subdirectory(shared)
add_subdirectory(model-loader)
add_subdirectory(dataset-creator)
add_subdirectory(runner)