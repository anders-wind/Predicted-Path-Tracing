#pragma once

#include "hitable.cuh"
#include "path_tracer/material.cuh"
#include "path_tracer/ray.cuh"
#include <math.h>
#include <memory>
#include <shared/vecs/vec3.cuh>

namespace ppt
{
namespace path_tracer
{

struct xy_rect : public hitable
{
    float x0, x1, y0, y1, k;
    material* _material;

    __device__ xy_rect(float x0, float x1, float y0, float y1, float k, material* material)
      : x0(x0), x1(x1), y0(y0), y1(y1), k(k), _material(material)
    {
    }

    __device__ ~xy_rect()
    {
        delete _material;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        auto dir = r.direction();
        auto ori = r.origin();
        float t = (k - ori[2]) / dir[2];
        if (t < t_min || t > t_max)
        {
            return false;
        }
        float x = ori[0] + t * dir[0];
        float y = ori[1] + t * dir[1];
        if ((x < x0 || x > x1) || (y < y0 || y > y1))
        {
            return false;
        }
        out.u = (x - x0) / (x1 - x0);
        out.v = (y - y0) / (y1 - y0);
        out.t = t;
        out.mat_ptr = _material;
        out.p = r.point_at_parameter(t);
        out.normal = vec3(0, 0, 1);
        return true;
    }

    __device__ virtual bool bounding_box(float, float, aabb& box) const override
    {
        box = aabb(vec3(x0, y0, k - 0.00001f), vec3(x1, y1, k + 0.000001f));
        return true;
    }
};
} // namespace path_tracer
} // namespace ppt