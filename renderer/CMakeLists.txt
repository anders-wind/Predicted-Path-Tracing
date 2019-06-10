cmake_minimum_required(VERSION 3.12)
project(ppt
    VERSION 0.0.1
    LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

find_program(OpenGl REQUIRED)

if (WIN32)
	set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/nvcc.exe")
endif (WIN32)

# Add third parties
add_subdirectory(third_party)


add_subdirectory(shared) 
add_subdirectory(path_tracer)
add_subdirectory(dataset_creator)
# add_subdirectory(model_loader)
add_subdirectory(runner)