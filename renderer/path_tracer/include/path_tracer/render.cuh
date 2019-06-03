#pragma once
#include <cuda.h>
#include <shared/cuda_helpers.cuh>
#include <shared/vec3.cuh>
#include <shared/vec5.cuh>
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
    vec3* m_image_matrix; // m for managed

    public:
    const size_t w;
    const size_t h;
    const size_t render_image_bytes;

    render(int w, int h) : w(w), h(h), render_image_bytes(w * h * sizeof(vec3))
    {
        checkCudaErrors(cudaMallocManaged((void**)&m_image_matrix, render_image_bytes));
    }

    // move operator
    render(render&& other)
      : m_image_matrix{ other.m_image_matrix }, w(w), h(h), render_image_bytes(render_image_bytes)
    {
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

        return *this;
    }

    ~render()
    {
        cudaFree(m_image_matrix);
    }

    // todo think about how we can return as ref?
    vec3* get_image_matrix()
    {
        return m_image_matrix;
    }

    std::vector<vec3> get_vector3_representation() const;

    std::vector<vec5> get_vector5_representation() const;
};

} // namespace path_tracer
} // namespace ppt