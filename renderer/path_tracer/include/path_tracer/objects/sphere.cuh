#pragma once

#include "hitable.cuh"
#include "path_tracer/ray.cuh"
#include <math.h>
#include <memory>
#include <shared/vecs/vec3.cuh>

namespace ppt
{
namespace path_tracer
{

struct sphere : public hitable
{
    vec3 _center;
    float _radius;
    material* _material;

    // sphere() = default;
    __device__ sphere(const vec3& center, float radius, material* material)
      : _center(center), _radius(radius), _material(material)
    {
    }

    __device__ ~sphere()
    {
        delete _material;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        vec3 oc = r.origin() - _center;
        const auto a = dot(r.direction(), r.direction());
        const auto b = dot(oc, r.direction());
        const auto c = dot(oc, oc) - _radius * _radius;
        const auto discriminant = b * b - a * c;

        if (discriminant > 0.0f)
        {
            auto temp = (-b - sqrt(b * b - a * c)) / a;
            if (temp < t_max && temp > t_min)
            {
                out.t = temp;
                out.p = r.point_at_parameter(temp);
                out.normal = ((out.p - _center) / _radius);
                out.mat_ptr = _material;
                out.normal.make_unit_vector();
                return true;
            }

            temp = (-b + sqrt(b * b - a * c)) / a;
            if (temp < t_max && temp > t_min)
            {
                out.t = temp;
                out.p = r.point_at_parameter(temp);
                out.normal = (_center - out.p) / _radius;
                out.mat_ptr = _material;
                out.normal.make_unit_vector();
                return false;
            }
        }
        return false;
    }

    __device__ virtual bool bounding_box(float, float, aabb& box) const override
    {
        box = aabb(_center - _radius, _center + _radius);
        return true;
    }
};

} // namespace path_tracer
} // namespace ppt