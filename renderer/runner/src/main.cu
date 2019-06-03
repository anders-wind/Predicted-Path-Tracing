#include <dataset_creator/dataset_repository.hpp>
#include <fstream>
#include <iostream>
#include <path_tracer/cuda_renderer.cuh>
#include <string>

int main()
{
    using namespace ppt::shared;
    using namespace ppt::path_tracer;
    using namespace ppt::dataset_creator;

    int w = 640;
    int h = 360;
    int samples[4] = { 1, 10, 100, 1000 };
    std::string filename = "render";
    auto repository = dataset_repository("/home/anders/Documents/datasets/ppt/640x360_run01");

    try
    {
        auto renderer = cuda_renderer(w, h, 5);
        auto render_datapoint = renderer.ray_trace_datapoint(samples);
        repository.save_datapoint(render_datapoint, "heyo");
    }
    catch (...)
    {
        std::cout << "failed" << std::endl;
    }
}