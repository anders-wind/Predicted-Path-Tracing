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
    int samples[4] = { 1, 2, 4, 8 };
    std::string filename = "render";

    try
    {
        auto renderer = cuda_renderer(w, h);
        auto render_datapoint = renderer.ray_trace_datapoint(samples);
        auto ppm = render_datapoint.get_ppm_representation(render_datapoint.target);

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