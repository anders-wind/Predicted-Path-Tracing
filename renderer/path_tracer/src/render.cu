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

render::render(int w, int h)
  : w(w)
  , h(h)
  , render_color_bytes(w * h * sizeof(ppt::shared::vec3))
  , render_image_bytes(w * h * sizeof(ppt::shared::vec8))
{
    checkCudaErrors(cudaMalloc((void**)&d_color_matrix, render_color_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_image_matrix, render_image_bytes));
}

render::render(render&& other)
  : d_image_matrix{ other.d_image_matrix }
  , w(w)
  , h(h)
  , render_color_bytes(render_color_bytes)
  , render_image_bytes(render_image_bytes)
{
    other.d_color_matrix = nullptr;
    other.d_image_matrix = nullptr;
}

render::~render()
{
    cudaFree(d_color_matrix);
    cudaFree(d_image_matrix);
}

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
    auto h_image_matrix = get_vector8_representation();
    get_vector_representation<vec3>(h_image_matrix, w, h, colors);

    return colors;
}

std::vector<vec8> render::get_vector8_representation() const
{
    auto h_image_matrix = std::vector<vec8>(w * h);
    h_image_matrix.resize(w * h);
    auto bytes = sizeof(vec8) * w * h;
    checkCudaErrors(cudaMemcpy(&h_image_matrix[0], d_image_matrix, bytes, cudaMemcpyDeviceToHost));
    return h_image_matrix;
}

std::vector<std::vector<std::array<unsigned char, 4>>> render::get_2d_byte_representation() const
{
    auto h_image_matrix = get_vector8_representation();

    auto result = std::vector<std::vector<std::array<unsigned char, 4>>>(h);
    for (auto i = 0; i < h; i++)
    {
        result[i].resize(w);
        for (auto j = 0; j < w; j++)
        {
            auto idx = RM(i, j, w);
            const auto e = h_image_matrix[idx].e;
            result[i][j][0] = e[0] * 255.f;
            result[i][j][1] = e[1] * 255.f;
            result[i][j][2] = e[2] * 255.f;
            result[i][j][3] = 1.0f * 255.f;
        }
    }
    return result;
}

} // namespace path_tracer
} // namespace ppt