#pragma once

#include "flip_normals.cuh"
#include "hitable.cuh"
#include "hitable_list.cuh"
#include "path_tracer/material.cuh"
#include "path_tracer/ray.cuh"
#include "rect.cuh"
#include <shared/vecs/vec3.cuh>

namespace ppt
{
namespace path_tracer
{

struct box : public hitable
{
    private:
    vec3 p_min, p_max;
    material* _material;
    hitable* rects;

    public:
    __device__ box(const vec3& p_min, const vec3& p_max, material* material)
      : p_min(p_min), p_max(p_max), _material(material)
    {
        hitable** list = new hitable*[6];
        list[0] = new xy_rect(p_min[0], p_max[0], p_min[1], p_max[1], p_max[2], material);
        list[1] = new flip_normals(new xy_rect(p_min[0], p_max[0], p_min[1], p_max[1], p_min[2], material));
        list[2] = new xz_rect(p_min[0], p_max[0], p_min[2], p_max[2], p_max[1], material);
        list[3] = new flip_normals(new xz_rect(p_min[0], p_max[0], p_min[2], p_max[2], p_min[1], material));
        list[4] = new yz_rect(p_min[1], p_max[1], p_min[2], p_max[2], p_max[0], material);
        list[5] = new flip_normals(new yz_rect(p_min[1], p_max[1], p_min[2], p_max[2], p_min[0], material));
        rects = new hitable_list(list, 6);
    }

    __device__ ~box()
    {
        delete _material;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        return rects->hit(r, t_min, t_max, out);
    }

    __device__ virtual bool bounding_box(float, float, aabb& box) const override
    {
        box = aabb(p_min, p_max);
        return true;
    }
};
} // namespace path_tracer
} // namespace ppt