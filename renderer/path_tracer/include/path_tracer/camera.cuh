#pragma once
#include "ray.cuh"
#include <random>
#include <shared/vec3.cuh>

namespace ppt
{
namespace path_tracer
{

struct camera
{
    private:
    vec3 _origin;
    vec3 _lower_left_corner;
    vec3 _horizontal;
    vec3 _vertical;

    public:
    __host__ __device__ camera(const vec3& lower_left_corner, const vec3& horizontal, const vec3& vertical, const vec3& origin)
      : _lower_left_corner(lower_left_corner), _horizontal(horizontal), _vertical(vertical), _origin(origin)
    {
    }

    __device__ ray get_ray(float u, float v) const
    {
        return ray(_origin, _lower_left_corner + (_horizontal * u) + (_vertical * v) - _origin);
    }
};


class camera_factory
{
    public:
    camera_factory() = default;

    __host__ __device__ camera make_16_9_camera() const
    {
        const auto lower_left_corner = vec3(-8.0, -4.5, -2.5);
        const auto horizontal = vec3(16, 0.0, 0.0);
        const auto vertical = vec3(0.0, 9.0, 0.0);
        const auto origin = vec3(0.0, 0.0, 0.0);
        return camera(lower_left_corner, horizontal, vertical, origin);
    }

    __host__ __device__ camera make_4_3_camera() const
    {
        const auto lower_left_corner = vec3(-2.0, -1.5, -1.5);
        const auto horizontal = vec3(4, 0.0, 0.0);
        const auto vertical = vec3(0.0, 3.0, 0.0);
        const auto origin = vec3(0.0, 0.0, 0.0);
        return camera(lower_left_corner, horizontal, vertical, origin);
    }

    __host__ __device__ camera make_square_camera() const
    {
        const auto lower_left_corner = vec3(-2.0, -2.0, -1.5);
        const auto horizontal = vec3(4, 0.0, 0.0);
        const auto vertical = vec3(0.0, 4.0, 0.0);
        const auto origin = vec3(0.0, 0.0, 0.0);
        return camera(lower_left_corner, horizontal, vertical, origin);
    }
};

} // namespace path_tracer
} // namespace ppt
