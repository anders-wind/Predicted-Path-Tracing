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
  , sample_bytes(w * h * sizeof(unsigned int))
  , variance_bytes(w * h * sizeof(float))
{
    checkCudaErrors(cudaMalloc((void**)&d_color_matrix, render_color_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_image_matrix, render_image_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_samples, sample_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_variance, variance_bytes));
}

render::render(render&& other)
  : d_image_matrix(std::move(other.d_image_matrix))
  , d_color_matrix(std::move(other.d_color_matrix))
  , w(w)
  , h(h)
  , render_color_bytes(render_color_bytes)
  , render_image_bytes(render_image_bytes)
  , sample_bytes(sample_bytes)
  , variance_bytes(variance_bytes)
{
    other.d_color_matrix = nullptr;
    other.d_image_matrix = nullptr;
    other.d_samples = nullptr;
    other.d_variance = nullptr;
}

render::~render()
{
    checkCudaErrors(cudaFree(d_color_matrix));
    checkCudaErrors(cudaFree(d_image_matrix));
    checkCudaErrors(cudaFree(d_samples));
    checkCudaErrors(cudaFree(d_variance));
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

std::vector<unsigned char> render::get_byte_representation() const
{
    auto h_image_matrix = get_vector8_representation();

    auto result = std::vector<unsigned char>(h * w * 4);

    for (auto i = 0; i < h; i++)
    {
        for (auto j = 0; j < w; j++)
        {
            const auto idx = i * w * 4 + j * 4;
            const auto& e = h_image_matrix[(h - i - 1) * w + j];
            result[idx + 0] = e[0] * 255.f;
            result[idx + 1] = e[1] * 255.f;
            result[idx + 2] = e[2] * 255.f;
            result[idx + 3] = 1.0f * 255.f;
        }
    }
    return result;
}

} // namespace path_tracer
} // namespace ppt