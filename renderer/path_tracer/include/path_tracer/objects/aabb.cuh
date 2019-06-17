#pragma once
#include "hitable.cuh"

namespace ppt
{
namespace path_tracer
{
// todo bench if these are faster than std::fmax()
inline float ffmin(float a, float b)
{
    return a < b ? a : b;
}
inline float ffmax(float a, float b)
{
    return a > b ? a : b;
}

/**
 * Axis Alignbed Bounding Box
 */
class aabb : hitable
{
    private:
    vec3 _min;
    vec3 _max;

    public:
    __device__ __host__ aabb(const vec3& min, const vec3& max) : _min(min), _max(max)
    {
    }

    __device__ __host__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        auto hit_anything = true;
        auto origin = r.origin();
        auto direction = r.direction();

        for (auto a = 0; a < 3; a++)
        {
            // concept:
            // float t0 = std::fmin((_min[a] - origin[a]) / direction[a], (_max[a] - origin[a]) / direction[a]);
            // float t1 = std::fmax((_min[a] - origin[a]) / direction[a], (_max[a] - origin[a]) / direction[a]);
            // t_min = std::fmax(t0, t_min);
            // t_min = std::fmin(t1, t_max);

            // performance improved
            float inv_d = 1.0f / direction[a];
            float t0 = (_min[a] - origin[a]) * inv_d;
            float t1 = (_max[a] - origin[a]) * inv_d;

            if (inv_d < 0.0f)
            {
                // swap
                auto temp = t0;
                t0 = t1;
                t1 = temp;
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;

            hit_anything &= t_max <= t_min;
        }
        return hit_anything;
    }
};
} // namespace path_tracer
} // namespace ppt