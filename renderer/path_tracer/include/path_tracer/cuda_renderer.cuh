#pragma once
#include "camera.cuh"
#include "hitable.cuh"
#include "hitable_list.cuh"
#include "material.cuh"
#include "objects/plane.cuh"
#include "objects/sphere.cuh"
#include "ray.cuh"
#include "render.cuh"
#include "render_datapoint.cuh"
#include <shared/cuda_helpers.cuh>
#include <shared/random_helpers.cuh>
#include <shared/sample_service.cuh>
#include <shared/scoped_timer.cuh>
#include <shared/vecs/vec3.cuh>
#include <shared/vecs/vec8.cuh>
#include <vector>

namespace ppt
{
namespace path_tracer
{

class cuda_renderer
{
    private:
    curandState* d_rand_state;
    hitable** d_list;
    hitable** d_world;
    camera* d_camera;
    const std::shared_ptr<shared::sample_service> _sampler;

    public:
    int num_threads_x = 16;
    int num_threads_y = 16;
    const dim3 blocks;
    const dim3 threads;
    const size_t w;
    const size_t h;

    cuda_renderer(int w, int h, std::shared_ptr<shared::sample_service> sampler);

    ~cuda_renderer();

    public: // Methods
    std::shared_ptr<render> ray_trace(int samples, int sample_sum) const;
    void ray_trace(int samples, int sample_sum, render& ray_traced_image) const;

    std::vector<path_tracer::render_datapoint> ray_trace_datapoints(size_t number_of_images);

    path_tracer::render_datapoint ray_trace_datapoint() const;

    path_tracer::render_datapoint ray_trace_datapoint(render& ray_traced_image) const;

    // private:
    void reset_image(render& ray_traced_image) const;
    void update_world();
};

} // namespace path_tracer
} // namespace ppt