#include <iostream>
#include <path_tracer/cuda_renderer.cuh>
#include <string>

int main()
{
    using namespace ppt::shared;
    using namespace ppt::path_tracer;
    int w = 1280;
    int h = 720;
    int s = 256;
    try
    {
        auto colors = cuda_renderer::cuda_ray_render(w, h, s);
        cuda_renderer::write_ppm_image(colors, w, h, "render");
    }
    catch (...)
    {
        std::cout << "failed" << std::endl;
    }
}