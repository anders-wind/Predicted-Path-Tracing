#include <dataset_creator/dataset_repository.hpp>
#include <fstream>
#include <iostream>
#include <path_tracer/cuda_renderer.cuh>
#include <shared/sample_service.cuh>
#include <string>

int main()
{
    // settings
    int w = 640;
    int h = 360;
    size_t number_of_images = 2;
    std::string filename = "trial";

    // services
    const auto sampler = std::make_shared<ppt::shared::sample_service>();
    auto repository = ppt::dataset_creator::dataset_repository(
        std::string(getenv("HOME")) + "/Documents/datasets/ppt/640x360_run07");

    auto renderer = ppt::path_tracer::cuda_renderer(w, h, sampler);

    // ray trace
    auto render_datapoints = renderer.ray_trace_datapoints(number_of_images);
    repository.save_datapoints(render_datapoints, filename);
    repository.save_ppms(render_datapoints, filename);
}