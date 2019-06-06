#pragma once

#include "path_tracer/hitable.cuh"
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

    __device__ __host__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        vec3 oc = r.origin() - _center;
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - _radius * _radius;
        float discriminant = b * b - a * c;

        if (discriminant > 0)
        {
            float temp = (-b - sqrt(b * b - a * c)) / a;
            if (temp < t_max && temp > t_min)
            {
                out.t = temp;
                out.p = r.point_at_parameter(temp);
                out.normal = (out.p - _center) / _radius;
                out.mat_ptr = _material;
                return true;
            }

            temp = (-b + sqrt(b * b - a * c)) / a;
            if (temp < t_max && temp > t_min)
            {
                out.t = temp;
                out.p = r.point_at_parameter(temp);
                out.normal = (out.p - _center) / _radius;
                out.mat_ptr = _material;
                return true;
            }
        }
        return false;
    }

    __device__ vec3 normal(const vec3& position) const
    {
        return _center - position;
    }
};

} // namespace path_tracer
} // namespace ppt