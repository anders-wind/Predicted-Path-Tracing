#pragma once
#include <array>
#include <cuda.h>
#include <shared/cuda_helpers.cuh>
#include <shared/scoped_lock.cuh>
#include <shared/vecs/vec3.cuh>
#include <shared/vecs/vec5.cuh>
#include <shared/vecs/vec8.cuh>
#include <string>
#include <vector>

namespace ppt
{
namespace path_tracer
{

/**
 * Render class contains logic for handeling the memory of a render,
 * as well as utility functions for serializing the render.
 */
class render
{
    private:
    ppt::shared::vec3* d_color_matrix;
    ppt::shared::vec8* d_image_matrix;
    std::shared_ptr<std::mutex> lock = std::make_shared<std::mutex>();
    int* d_samples;
    float* d_variance;

    public:
    const size_t w;
    const size_t h;
    const size_t render_color_bytes;
    const size_t render_image_bytes;
    const size_t sample_bytes;
    const size_t variance_bytes;

    render(int w, int h);

    // move operator
    render(render&& other);

    render(const render& other) = delete;
    render& operator=(const render& other) = delete;
    render& operator=(render&& other) = delete;

    ~render();

    // todo think about how we can return as ref?
    ppt::shared::vec8* get_image_matrix()
    {
        return d_image_matrix;
    }

    ppt::shared::vec3* get_color_matrix()
    {
        return d_color_matrix;
    }

    int* get_sample_matrix()
    {
        return d_samples;
    }

    float* get_variance_matrix()
    {
        return d_variance;
    }

    std::vector<ppt::shared::vec3> get_vector3_representation() const;

    std::vector<ppt::shared::vec8> get_vector8_representation() const;

    std::vector<unsigned char> get_byte_representation() const;

    std::unique_ptr<shared::scoped_lock> get_scoped_lock()
    {
        return std::make_unique<shared::scoped_lock>(lock);
    }
};

} // namespace path_tracer
} // namespace ppt