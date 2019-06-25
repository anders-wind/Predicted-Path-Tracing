#pragma once
#include "hitable.cuh"
#include <math.h>

namespace ppt
{
namespace path_tracer
{
constexpr float TO_RADIANS = M_PI / 180.0f;

struct rotate_y : public hitable
{
    private:
    hitable* inner;

    float angle;
    float sin_theta;
    float cos_theta;
    bool has_box;
    aabb bbox;

    public:
    __device__ rotate_y(hitable* inner, float angle) : inner(inner), angle(angle)
    {
        float radians = TO_RADIANS * angle;
        sin_theta = sin(radians);
        cos_theta = cos(radians);
        has_box = inner->bounding_box(0, 1, bbox);

        vec3 min = vec3(std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max());
        vec3 max = vec3(std::numeric_limits<float>::min(),
                        std::numeric_limits<float>::min(),
                        std::numeric_limits<float>::min());

        for (auto i = 0; i < 2; i++)
        {
            const float x = i * bbox.max()[0] + (1 - i) * bbox.min()[0];

            for (auto j = 0; j < 2; j++)
            {
                const float y = j * bbox.max()[1] + (1 - j) * bbox.min()[1];

                for (auto k = 0; k < 2; k++)
                {
                    const float z = k * bbox.max()[2] + (1 - k) * bbox.min()[2];
                    const float newx = cos_theta * x + sin_theta * z;
                    const float newz = -sin_theta * x + cos_theta * z;
                    vec3 tester(newx, y, newz);
                    for (auto c = 0; c < 3; c++)
                    {
                        if (tester[c] > max[c])
                        {
                            max[c] == tester[c];
                        }
                        if (tester[c] < min[c])
                        {
                            min[c] = tester[c];
                        }
                    }
                }
            }
        }
        bbox = aabb(min, max);
    }

    __device__ ~rotate_y()
    {
        delete inner;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        vec3 ori = r.origin();
        vec3 dir = r.direction();
        ori[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
        ori[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];
        dir[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
        dir[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];
        ray rotated_r(ori, dir);
        if (inner->hit(rotated_r, t_min, t_max, out))
        {
            vec3 p = out.p;
            vec3 normal = out.normal;

            p[0] = cos_theta * out.p[0] + sin_theta * out.p[2];
            p[2] = -sin_theta * out.p[0] + cos_theta * out.p[2];

            normal[0] = cos_theta * out.normal[0] + sin_theta * out.normal[0];
            normal[2] = -sin_theta * out.normal[0] + cos_theta * out.normal[0];

            out.p = p;
            out.normal = normal;
            return true;
        }
        return false;
    }

    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        box = bbox;
        return has_box;
    }
}; // namespace path_tracer


} // namespace path_tracer
} // namespace ppt
