#pragma once
#include "vec3.cuh"
#include "vec5.cuh"
#include <sstream>
#include <vector>

namespace ppt
{
namespace shared
{
class render_datapoint
{
    public:
    std::vector<vec5> renders[3];
    std::vector<vec3> target;

    render_datapoint()
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
            ss << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << ", " << vec[4]
               << ", " << std::endl;
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
            ss << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << std::endl;
        }
        return ss.str();
    }
};
} // namespace shared
} // namespace ppt