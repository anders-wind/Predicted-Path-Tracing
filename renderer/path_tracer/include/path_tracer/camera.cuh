#pragma once
#include "ray.cuh"
#include <random>
#include <shared/random_helpers.cuh>
#include <shared/vecs/vec3.cuh>

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
    float _lens_radius;
    vec3 _u, _v, _w;

    public:
    float _min_depth;
    float _max_depth;


    __host__ __device__ camera(const vec3& lower_left_corner,
                               const vec3& horizontal,
                               const vec3& vertical,
                               const vec3& origin,
                               float lens_radius,
                               const vec3& u,
                               const vec3& v,
                               const vec3& w,
                               float min_depth,
                               float max_depth)
      : _origin(origin)
      , _lower_left_corner(lower_left_corner)
      , _horizontal(horizontal)
      , _vertical(vertical)
      , _lens_radius(lens_radius)
      , _u(u)
      , _v(v)
      , _w(w)
      , _min_depth(min_depth)
      , _max_depth(max_depth)
    {
    }

    __host__ __device__ camera(const camera& other)
      : _origin(other._origin)
      , _lower_left_corner(other._lower_left_corner)
      , _horizontal(other._horizontal)
      , _vertical(other._vertical)
      , _lens_radius(other._lens_radius)
      , _u(other._u)
      , _v(other._v)
      , _w(other._w)
      , _min_depth(other._min_depth)
      , _max_depth(other._max_depth)
    {
    }


    __device__ ray get_ray(float u, float v) const
    {
        return ray(_origin, _lower_left_corner + (_horizontal * u) + (_vertical * v) - _origin);
    }

    __device__ ray get_ray(float u, float v, curandState* local_rand_state) const
    {
        const auto rd = shared::random_in_unit_disk(local_rand_state) * _lens_radius;
        const auto offset = (u * rd[0]) + (v * rd[1]);
        return ray(_origin + offset, _lower_left_corner + (_horizontal * u) + (_vertical * v) - _origin - offset);
    }
};

constexpr float radian = M_PI / 180.0f;

class camera_factory
{
    public:
    float min_depth;
    float max_depth;

    __host__ __device__ camera_factory(float min_depth = 0.00001f, float max_depth = 10000.0f)
      : min_depth(min_depth)
      , max_depth(max_depth){

      };

    __host__ __device__ camera make_16_9_camera(const vec3& look_from,
                                                const vec3& look_at,
                                                float vfov,
                                                float aperture,
                                                float focus_dist) const
    {
        const float ratio = (16.0f / 9.0f);
        const vec3 v_up = vec3(0, 1, 0);
        return make_camera(look_from, look_at, v_up, vfov, ratio, aperture, focus_dist);
    }

    __host__ __device__ camera
    make_4_3_camera(const vec3& look_from, const vec3& look_at, float vfov, float aperture, float focus_dist) const
    {
        const float ratio = (4.0f / 3.0f);
        const vec3 v_up = vec3(0, 1, 0);
        return make_camera(look_from, look_at, v_up, vfov, ratio, aperture, focus_dist);
    }

    __host__ __device__ camera make_square_camera(const vec3& look_from,
                                                  const vec3& look_at,
                                                  float vfov,
                                                  float aperture,
                                                  float focus_dist) const
    {
        const float ratio = 1.0f;
        const vec3 v_up = vec3(0, 1, 0);
        return make_camera(look_from, look_at, v_up, vfov, ratio, aperture, focus_dist);
    }

    __host__ __device__ camera make_camera(const vec3& look_from,
                                           const vec3& look_at,
                                           const vec3& v_up,
                                           float vfov,
                                           float ratio,
                                           float aperture,
                                           float focus_dist) const
    {
        vec3 u, v, w;
        const float lens_radius = aperture / 2.0f;
        const float theta = vfov * radian;
        const float half_height = std::tan(theta / 2.0f);
        const float half_width = ratio * half_height;

        w = vec3::unit_vector(look_from - look_at);
        u = vec3::unit_vector(cross(v_up, w));
        v = cross(w, u);

        const auto origin = look_from;
        const auto lower_left_corner = look_from - (half_width * focus_dist * u) -
                                       (half_height * focus_dist * v) - (focus_dist * w);
        const auto horizontal = (2.0f * half_width) * focus_dist * u;
        const auto vertical = (2.0f * half_height) * focus_dist * v;
        return camera(lower_left_corner, horizontal, vertical, origin, lens_radius, u, v, w, min_depth, max_depth);
    }
};

} // namespace path_tracer
} // namespace ppt
