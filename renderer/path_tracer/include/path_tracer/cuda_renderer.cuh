#pragma once
#include "camera.cuh"
#include "objects/hitable.cuh"
#include "render.cuh"
#include "render_datapoint.cuh"
#include <shared/cuda_helpers.cuh>
#include <shared/matrix_probability_stats.cuh>
#include <shared/sample_service.cuh>
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
    const std::shared_ptr<shared::matrix_probability_stats<vec3>> _matrix_probability_stats;

    public:
    const dim3 blocks;
    const dim3 threads;
    const size_t w;
    const size_t h;

    cuda_renderer(int w,
                  int h,
                  const dim3& blocks,
                  const dim3& threads,
                  std::shared_ptr<shared::sample_service> sampler,
                  std::shared_ptr<shared::matrix_probability_stats<vec3>> matrix_probability_stats);

    ~cuda_renderer();

    public: // Methods
    std::shared_ptr<render> ray_trace(int samples, int sample_sum, bool fast = true) const;
    void ray_trace(int samples, int sample_sum, render& ray_traced_image, bool fast = true) const;

    std::vector<path_tracer::render_datapoint> ray_trace_datapoints(size_t number_of_images);

    path_tracer::render_datapoint ray_trace_datapoint() const;

    path_tracer::render_datapoint ray_trace_datapoint(render& ray_traced_image) const;

    // private:
    void reset_image(render& ray_traced_image) const;
    void update_world();
    float variance_sum() const
    {
        return _matrix_probability_stats->get_variance_sum();
    }
};

} // namespace path_tracer
} // namespace ppt