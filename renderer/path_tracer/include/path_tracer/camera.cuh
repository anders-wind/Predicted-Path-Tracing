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
    vec3 _origin;
    vec3 _lower_left_corner;
    vec3 _horizontal;
    vec3 _vertical;
    float _min_depth;
    float _max_depth;

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

    __host__ __device__ camera make_16_9_camera(const vec3 look_from, const vec3 look_at, const vec3 v_up, float vfov) const
    {
        vec3 u, v, w;
        const float ratio = (16.0f / 9.0f);
        const float theta = vfov * M_PI / 180.0f;
        const float half_height = std::tan(theta / 2);
        const float half_width = ratio * half_height;

        w = unit_vector(look_from - look_at);
        u = unit_vector(cross(v_up, w));
        v = cross(w, u);

        const auto origin = look_from;
        auto lower_left_corner = vec3(-half_width, -half_height, -1.0f);
        lower_left_corner = look_from - (half_width * u) - (half_height * v) - w;

        const auto horizontal = (2.0f * half_width) * u;
        const auto vertical = (2.0f * half_height) * v;
        return camera(lower_left_corner, horizontal, vertical, origin, min_depth, max_depth);
    }

    __host__ __device__ camera make_4_3_camera(float vfov) const
    {
        const float ratio = (4.0f / 3.0f);
        const float theta = vfov * M_PI / 180.0f;
        const float half_height = std::tan(theta / 2);
        const float half_width = ratio * half_height;
        const auto lower_left_corner = vec3(-half_width, -half_height, -1.0);
        const auto horizontal = vec3(2 * half_width, 0.0, 0.0);
        const auto vertical = vec3(0.0, 2 * half_height, 0.0);
        const auto origin = vec3(0.0, 0.0, 0.0);
        return camera(lower_left_corner, horizontal, vertical, origin, min_depth, max_depth);
    }

    __host__ __device__ camera make_square_camera(float vfov) const
    {
        const float ratio = (1.0f / 1.0f);
        const float theta = vfov * M_PI / 180.0f;
        const float half_height = std::tan(theta / 2);
        const float half_width = ratio * half_height;
        const auto lower_left_corner = vec3(-half_width, -half_height, -1.0);
        const auto horizontal = vec3(2 * half_width, 0.0, 0.0);
        const auto vertical = vec3(0.0, 2 * half_height, 0.0);
        const auto origin = vec3(0.0, 0.0, 0.0);
        return camera(lower_left_corner, horizontal, vertical, origin, min_depth, max_depth);
    }
};

} // namespace path_tracer
} // namespace ppt
