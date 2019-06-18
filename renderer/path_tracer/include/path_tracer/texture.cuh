#pragma once
#include <shared/vecs/vec3.cuh>

namespace ppt
{
namespace path_tracer
{

class texture
{
    public:
    __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public texture
{
    private:
    vec3 color;

    public:
    // constant_texture() = default;
    __device__ constant_texture(vec3 c) : color(c)
    {
    }

    __device__ virtual vec3 value(float u, float v, const vec3& p) const override
    {
        return color;
    }
};

} // namespace path_tracer
} // namespace ppt