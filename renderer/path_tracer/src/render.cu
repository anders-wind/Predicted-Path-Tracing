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

std::string render::get_ppm_representation() const
{
    const auto colors = get_vector3_representation();
    return get_ppm_representation(colors);
}

std::string render::get_ppm_representation(const std::vector<vec3>& colors) const
{
    const auto timer = shared::scoped_timer("get_ppm_representation");

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

std::vector<vec3> render::get_vector3_representation() const
{
    auto colors = std::vector<vec3>(w * h);
    get_vector3_representation(colors);
    return colors;
}

void render::get_vector3_representation(std::vector<vec3>& colors) const
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

std::vector<vec5> render::get_vector5_representation() const
{
    auto colors = std::vector<vec5>(w * h);
    get_vector5_representation(colors);
    return colors;
}

void render::get_vector5_representation(std::vector<vec5>& colors) const
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

} // namespace path_tracer
} // namespace ppt