#include "path_tracer/render.cuh"
#include <shared/scoped_timer.cuh>
#include <sstream>

namespace ppt
{
namespace path_tracer
{

#define RM(row, col, w) row* w + col
#define CM(row, col, h) col* h + row
using namespace ppt::shared;

std::string render::get_ppm_representation() const
{
    const auto colors = get_vector_representation();
    return get_ppm_representation(colors);
}

std::string render::get_ppm_representation(const std::vector<rgb>& colors) const
{
    const auto timer = scoped_timer("write_ppm_image");

    std::stringstream ss;
    ss << "P3\n" << w << " " << h << "\n255\n";

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            ss << colors[RM(i, j, w)] << std::endl;
        }
    }

    return ss.str();
}

std::vector<rgb> render::get_vector_representation() const
{
    auto colors = std::vector<rgb>(w * h);
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            const size_t pixel_index = RM(i, j, w);
            colors[pixel_index] = m_image_matrix[pixel_index];
        }
    }
    return colors;
}

} // namespace path_tracer
} // namespace ppt