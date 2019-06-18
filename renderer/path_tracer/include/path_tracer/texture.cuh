#pragma once
#include <shared/perlin.cuh>
#include <shared/vecs/vec3.cuh>

namespace ppt
{
namespace path_tracer
{

class texture
{
    public:
    __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
    __device__ virtual ~texture()
    {
    }
};

class constant_texture : public texture
{
    private:
    const vec3 color;

    public:
    // constant_texture() = default;
    __device__ constant_texture(vec3 c) : color(c)
    {
    }

    __device__ virtual vec3 value(float, float, const vec3&) const override
    {
        return color;
    }
};

class checker_texture : public texture
{
    private:
    const texture* const even;
    const texture* const odd;

    public:
    // constant_texture() = default;
    __device__ checker_texture(texture* t0, texture* t1) : even(t0), odd(t1)
    {
    }
    __device__ ~checker_texture()
    {
        delete even;
        delete odd;
    }

    __device__ virtual vec3 value(float u, float v, const vec3& p) const override
    {
        float sines = sin(10 * p[0]) * sin(10 * p[1]) * sin(10 * p[2]);
        if (sines < 0)
        {
            return odd->value(u, v, p);
        }
        else
        {
            return even->value(u, v, p);
        }
    }
};

class noise_texture : public texture
{
    private:
    shared::perlin noise;

    public:
    // constant_texture() = default;
    __device__ noise_texture()
    {
    }

    __device__ virtual vec3 value(float, float, const vec3& p) const override
    {
        return vec3(1) * noise.noise(p);
    }
};


} // namespace path_tracer
} // namespace ppt