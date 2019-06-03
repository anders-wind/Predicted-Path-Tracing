#include "path_tracer/render.cuh"
#include <shared/scoped_timer.cuh>
#include <sstream>

namespace ppt
{
namespace path_tracer
{

using namespace ppt::shared;
#define RM(row, col, w) row* w + col
#define CM(row, col, h) col* h + row


template <typename T>
void get_vector_representation(vec3* m_image_matrix, size_t w, size_t h, std::vector<T>& colors)
{
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            const size_t pixel_index = RM(i, j, w);
            colors[pixel_index] = m_image_matrix[pixel_index];
        }
    }
}

std::vector<vec3> render::get_vector3_representation() const
{
    auto colors = std::vector<vec3>(w * h);
    get_vector_representation(m_image_matrix, w, h, colors);
    return colors;
}

std::vector<vec5> render::get_vector5_representation() const
{
    auto colors = std::vector<vec5>(w * h);
    get_vector_representation(m_image_matrix, w, h, colors);
    return colors;
}


} // namespace path_tracer
} // namespace ppt