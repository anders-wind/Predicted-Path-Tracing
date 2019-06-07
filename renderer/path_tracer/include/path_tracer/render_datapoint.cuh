#pragma once
#include "render.cuh"
#include <shared/scoped_timer.cuh>
#include <shared/vecs/vec3.cuh>
#include <shared/vecs/vec5.cuh>
#include <shared/vecs/vec8.cuh>
#include <sstream>
#include <vector>

namespace ppt
{
namespace path_tracer
{
using namespace shared;
#define RM(row, col, w) row* w + col
#define CM(row, col, h) col* h + row

class render_datapoint
{
    constexpr static int input_datapoints = 3;

    public:
    std::array<std::vector<vec8>, input_datapoints> renders;
    std::vector<vec3> target;

    size_t w;
    size_t h;

    render_datapoint(int w, int h) : w(w), h(h)
    {
    }

    void set_result(const render& render_holder, int idx)
    {
        if (idx < input_datapoints)
        {
            renders[idx] = render_holder.get_vector8_representation();
        }
        else if (idx == input_datapoints)
        {
            target = render_holder.get_vector3_representation();
        }
        else
        {
            std::cerr << input_datapoints << " " << idx << std::endl;
            throw std::runtime_error("Out of bounds datapoint");
        }
    }


    size_t constexpr renders_size() const
    {
        return 3;
    }

    std::string get_render_string(std::vector<vec8> render) const
    {
        std::stringstream ss;
        ss << "r, g, b, c, d, nx, ny, nz" << std::endl;
        ss << std::setprecision(7);
        for (const auto& vec : render)
        {
            ss << vec[0] << ", ";
            ss << vec[1] << ", ";
            ss << vec[2] << ", ";
            ss << vec[3] << ", ";
            ss << vec[4] << ", ";
            ss << vec[5] << ", ";
            ss << vec[6] << ", ";
            ss << vec[7] << std::endl;
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
        ss << "x, y, z" << std::endl;
        ss << std::setprecision(7);
        for (const auto& vec : target)
        {
            ss << vec[0] << ", ";
            ss << vec[1] << ", ";
            ss << vec[2] << std::endl;
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
} // namespace path_tracer
} // namespace ppt