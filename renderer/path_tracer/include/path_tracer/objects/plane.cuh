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

struct plane : public hitable
{
    vec3 _pos;
    vec3 _normal;
    material* _material;

    __device__ plane(const vec3& pos, const vec3& normal, material* material)
      : _pos(pos), _normal(normal), _material(material)
    {
        _normal.make_unit_vector();
    }

    __device__ ~plane()
    {
        delete _material;
    }

    __device__ bool hit(const ray& r, float, float, hit_record& out) const override
    {
        const float divider = dot(r._direction, _normal);
        if (divider == 0.0f)
        {
            return false;
        }

        const float t = dot(_normal, (_pos - r._origin)) / divider;
        if (t < 0.0f)
        {
            return false;
        }

        out.t = t;
        out.mat_ptr = _material;
        out.p = r.point_at_parameter(t);
        out.normal = _normal;

        return true;
    }

    __device__ virtual bool bounding_box(float, float, aabb&) const override
    {
        return false;
    }
};
} // namespace path_tracer
} // namespace ppt