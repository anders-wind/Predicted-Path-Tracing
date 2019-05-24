#include <fstream>
#include <iostream>
#include <path_tracer/cuda_renderer.cuh>
#include <string>

int main()
{
    using namespace ppt::shared;
    using namespace ppt::path_tracer;
    int w = 640;
    int h = 360;
    int s = 1;
    std::string filename = "render";

    try
    {
        auto renderer = cuda_renderer(w, h);
        auto render = renderer.ray_trace(s);
        auto ppm = render.get_ppm_representation();

        std::ofstream myfile;
        myfile.open(filename + ".ppm");
        myfile << ppm;
        myfile.close();
    }
    catch (...)
    {
        std::cout << "failed" << std::endl;
    }
}