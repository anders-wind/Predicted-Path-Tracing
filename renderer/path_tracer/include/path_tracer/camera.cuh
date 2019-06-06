#pragma once
#include "ray.cuh"
#include <random>
#include <shared/vecs/vec3.cuh>

namespace ppt
{
namespace path_tracer
{

struct camera
{
    public:
    const vec3 _origin;
    const vec3 _lower_left_corner;
    const vec3 _horizontal;
    const vec3 _vertical;
    const float _min_depth;
    const float _max_depth;

    __host__ __device__ camera(const vec3& lower_left_corner,
                               const vec3& horizontal,
                               const vec3& vertical,
                               const vec3& origin,
                               float min_depth,
                               float max_depth)
      : _lower_left_corner(lower_left_corner)
      , _horizontal(horizontal)
      , _vertical(vertical)
      , _origin(origin)
      , _min_depth(min_depth)
      , _max_depth(max_depth)
    {
    }

    __host__ __device__ camera(const camera& camera)
      : _lower_left_corner(camera._lower_left_corner)
      , _horizontal(camera._horizontal)
      , _vertical(camera._vertical)
      , _origin(camera._origin)
      , _max_depth(camera._max_depth)
      , _min_depth(camera._min_depth)
    {
    }

    __host__ __device__ camera operator=(const camera& other)
    {
        return camera(other);
    }

    __device__ float get_min_depth() const
    {
        return _min_depth;
    }

    __device__ float get_max_depth() const
    {
        return _max_depth;
    }


    __device__ ray get_ray(float u, float v) const
    {
        return ray(_origin, _lower_left_corner + (_horizontal * u) + (_vertical * v) - _origin);
    }
};


class camera_factory
{
    public:
    float min_depth = 0.00001f;
    float max_depth = 10000.0f;
    camera_factory() = default;

    __host__ __device__ camera make_16_9_camera() const
    {
        const auto lower_left_corner = vec3(-8.0, -4.5, -2.5);
        const auto horizontal = vec3(16, 0.0, 0.0);
        const auto vertical = vec3(0.0, 9.0, 0.0);
        const auto origin = vec3(0.0, 0.0, 0.0);
        return camera(lower_left_corner, horizontal, vertical, origin, min_depth, max_depth);
    }

    __host__ __device__ camera make_4_3_camera() const
    {
        const auto lower_left_corner = vec3(-2.0, -1.5, -1.5);
        const auto horizontal = vec3(4, 0.0, 0.0);
        const auto vertical = vec3(0.0, 3.0, 0.0);
        const auto origin = vec3(0.0, 0.0, 0.0);
        return camera(lower_left_corner, horizontal, vertical, origin, min_depth, max_depth);
    }

    __host__ __device__ camera make_square_camera() const
    {
        const auto lower_left_corner = vec3(-2.0, -2.0, -1.5);
        const auto horizontal = vec3(4, 0.0, 0.0);
        const auto vertical = vec3(0.0, 4.0, 0.0);
        const auto origin = vec3(0.0, 0.0, 0.0);
        return camera(lower_left_corner, horizontal, vertical, origin, min_depth, max_depth);
    }
};

} // namespace path_tracer
} // namespace ppt
