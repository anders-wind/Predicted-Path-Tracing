#pragma once
#include "scoped_timer.cuh"
#include "vecs/vec3.cuh"
#include "vecs/vec5.cuh"
#include <sstream>
#include <vector>

namespace ppt
{
namespace shared
{

#define RM(row, col, w) row* w + col
#define CM(row, col, h) col* h + row
class render_datapoint
{
    public:
    std::array<std::vector<vec5>, 3> renders;
    std::vector<vec3> target;

    size_t w;
    size_t h;

    render_datapoint(int w, int h) : w(w), h(h)
    {
    }

    size_t constexpr renders_size() const
    {
        return 3;
    }

    std::string get_render_string(std::vector<vec5> render) const
    {
        std::stringstream ss;
        ss << "x,y,z,v,w" << std::endl;
        for (const auto& vec : render)
        {
            ss << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << ", " << vec[4] << std::endl;
        }
        return ss.str();
    }

    std::string get_render_string(int idx) const
    {
        if (idx >= 3)
        {
            throw std::runtime_error("index out of bounds");
        }
        return get_render_string(renders[idx]);
    }

    std::string get_target_string() const
    {
        std::stringstream ss;
        ss << "x,y,z" << std::endl;
        for (const auto& vec : target)
        {
            ss << vec[0] << ", " << vec[1] << ", " << vec[2] << std::endl;
        }
        return ss.str();
    }

    template <typename T> std::string get_ppm_representation(const std::vector<T>& colors) const
    {
        std::stringstream ss;
        ss << "P3\n" << w << " " << h << "\n255\n";

        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                auto color = colors[RM(i, j, w)];
                ss << static_cast<unsigned int>(color.e[0] * 255.99) << " "
                   << static_cast<unsigned int>(color.e[1] * 255.99) << " "
                   << static_cast<unsigned int>(color.e[2] * 255.99) << std::endl;
            }
        }

        return ss.str();
    }
};
} // namespace shared
} // namespace ppt