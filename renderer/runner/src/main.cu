#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <path_tracer/camera.cuh>
#include <path_tracer/cuda_renderer.cuh>
#include <path_tracer/hitable.cuh>
#include <path_tracer/hitable_list.cuh>
#include <path_tracer/material.cuh>
#include <path_tracer/ray.cuh>
#include <path_tracer/sphere.cuh>
#include <random>
#include <shared/random_helpers.cuh>
#include <shared/vec3.cuh>
#include <string>
#include <vector>

int main()
{
    using namespace ppt::shared;
    using namespace ppt::path_tracer;
    int w = 1200;
    int h = 600;
    int s = 400;
    try
    {
        std::cout << "A" << std::endl;

        auto colors = cuda_renderer::cuda_ray_render(w, h, s);

        std::cout << "B" << std::endl;

        cuda_renderer::write_ppm_image(colors, w, h, "render");

        std::cout << "C" << std::endl;
    }
    catch (...)
    {
        std::cout << "failed" << std::endl;
    }
}