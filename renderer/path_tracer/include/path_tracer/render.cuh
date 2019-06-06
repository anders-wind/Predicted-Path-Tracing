#pragma once
#include <cuda.h>
#include <shared/cuda_helpers.cuh>
#include <shared/vecs/vec3.cuh>
#include <shared/vecs/vec5.cuh>
#include <string>
#include <vector>

namespace ppt
{
namespace path_tracer
{

using namespace ppt::shared;
/**
 * Render class contains logic for handeling the memory of a render,
 * as well as utility functions for serializing the render.
 */
class render
{
    private:
    vec3* d_color_matrix; // d for device only
    vec5* m_image_matrix; // m for managed

    public:
    const size_t w;
    const size_t h;
    const size_t render_color_bytes;
    const size_t render_image_bytes;

    render(int w, int h)
      : w(w), h(h), render_color_bytes(w * h * sizeof(vec3)), render_image_bytes(w * h * sizeof(vec5))
    {
        checkCudaErrors(cudaMallocManaged((void**)&d_color_matrix, render_color_bytes));
        checkCudaErrors(cudaMallocManaged((void**)&m_image_matrix, render_image_bytes));
    }

    // move operator
    render(render&& other)
      : m_image_matrix{ other.m_image_matrix }
      , w(w)
      , h(h)
      , render_color_bytes(render_color_bytes)
      , render_image_bytes(render_image_bytes)
    {
        other.d_color_matrix = nullptr;
        other.m_image_matrix = nullptr;
    }

    // For now delete copy and assignment to make sure we do not do it anywhere
    render(const render& other) = delete;
    render& operator=(const render& other) = delete;
    render& operator=(render&& other)
    {
        cudaFree(m_image_matrix);
        m_image_matrix = other.m_image_matrix;
        other.m_image_matrix = nullptr;

        cudaFree(d_color_matrix);
        d_color_matrix = other.d_color_matrix;
        other.d_color_matrix = nullptr;

        return *this;
    }

    ~render()
    {
        cudaFree(m_image_matrix);
    }

    // todo think about how we can return as ref?
    vec5* get_image_matrix()
    {
        return m_image_matrix;
    }

    vec3* get_color_matrix()
    {
        return d_color_matrix;
    }

    std::vector<vec3> get_vector3_representation() const;

    std::vector<vec5> get_vector5_representation() const;
};

} // namespace path_tracer
} // namespace ppt