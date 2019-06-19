#pragma once
#include "path_tracer/ray.cuh"

namespace ppt
{
namespace path_tracer
{

// forward decleration
struct material;
class aabb;

struct hit_record
{
    float t;
    float u;
    float v;
    vec3 p;
    vec3 normal = vec3(0.0f);
    material* mat_ptr;
};

struct hitable
{
    __device__ virtual ~hitable()
    {
    }
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& out) const = 0;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
};

} // namespace path_tracer
} // namespace ppt
