#include "path_tracer/render.cuh"
#include <shared/scoped_timer.cuh>
#include <sstream>

namespace ppt
{
namespace path_tracer
{

using vec3 = ppt::shared::vec3;
using vec5 = ppt::shared::vec5;
using vec8 = ppt::shared::vec8;

#define RM(row, col, w) row* w + col
#define CM(row, col, h) col* h + row


template <typename T>
void get_vector_representation(std::vector<vec8> h_image_matrix, size_t w, size_t h, std::vector<T>& colors)
{
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            const auto pixel_index = RM(i, j, w);
            colors[pixel_index] = T(h_image_matrix[pixel_index].e);
        }
    }
}

std::vector<vec3> render::get_vector3_representation() const
{
    auto colors = std::vector<vec3>(w * h);
    auto h_image_matrix = std::vector<vec8>(w * h);
    h_image_matrix.resize(w * h);
    auto bytes = sizeof(vec8) * w * h;
    checkCudaErrors(cudaMemcpy(&h_image_matrix[0], d_image_matrix, bytes, cudaMemcpyDeviceToHost));
    get_vector_representation<vec3>(h_image_matrix, w, h, colors);

    return colors;
}

std::vector<vec8> render::get_vector8_representation() const
{
    auto colors = std::vector<vec8>(w * h);
    auto h_image_matrix = std::vector<vec8>(w * h);
    h_image_matrix.resize(w * h);
    auto bytes = sizeof(vec8) * w * h;
    checkCudaErrors(cudaMemcpy(&h_image_matrix[0], d_image_matrix, bytes, cudaMemcpyDeviceToHost));
    return h_image_matrix;
}


} // namespace path_tracer
} // namespace ppt