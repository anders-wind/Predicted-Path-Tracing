#pragma once
#include <array>
#include <cuda.h>
#include <shared/cuda_helpers.cuh>
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

    public:
    const size_t w;
    const size_t h;
    const size_t render_color_bytes;
    const size_t render_image_bytes;

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

    std::vector<ppt::shared::vec3> get_vector3_representation() const;

    std::vector<ppt::shared::vec8> get_vector8_representation() const;

    std::vector<std::vector<std::array<unsigned char, 4>>> get_2d_byte_representation() const;
};

} // namespace path_tracer
} // namespace ppt