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

    // settings
    int w = 640;
    int h = 360;
    int samples[4] = { 1, 10, 100, 1000 };
    size_t number_of_images = 10;
    std::string filename = "trial_run";
    auto repository = dataset_repository("/home/anders/Documents/datasets/ppt/640x360_run02");

    // run
    try
    {
        auto renderer = cuda_renderer(w, h);
        auto render_datapoints = renderer.ray_trace_datapoints(samples, number_of_images);
        repository.save_datapoints(render_datapoints, filename);
        repository.save_ppms(render_datapoints, filename);
    }
    catch (...)
    {
        std::cout << "failed" << std::endl;
    }
}