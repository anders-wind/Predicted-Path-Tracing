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
    hitable** rects;

    public:
    __device__ box(const vec3& p_min, const vec3& p_max, material* material)
      : p_min(p_min), p_max(p_max), _material(material)
    {
        rects = new hitable*[6];
        rects[0] = new xy_rect(p_min[0], p_max[0], p_min[1], p_max[1], p_max[2], material);
        rects[1] = new flip_normals(new xy_rect(p_min[0], p_max[0], p_min[1], p_max[1], p_min[2], material));
        rects[2] = new xz_rect(p_min[0], p_max[0], p_min[2], p_max[2], p_max[1], material);
        rects[3] = new flip_normals(new xz_rect(p_min[0], p_max[0], p_min[2], p_max[2], p_min[1], material));
        rects[4] = new yz_rect(p_min[1], p_max[1], p_min[2], p_max[2], p_max[0], material);
        rects[5] = new flip_normals(new yz_rect(p_min[1], p_max[1], p_min[2], p_max[2], p_min[0], material));
    }

    __device__ ~box()
    {
        delete _material;
        for (auto i = 0; i < 6; i++)
        {
            delete rects[i];
        }
        delete[] rects;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        hit_record temp_closest;
        auto hit_anything = false;
        out.t = t_max;

        const auto& hitable_x = r.direction()[0] < 0 ? rects[4] : rects[5];
        if (hitable_x->hit(r, t_min, out.t, temp_closest))
        {
            hit_anything = true;
            out = temp_closest;
        }
        const auto& hitable_y = r.direction()[1] < 0 ? rects[2] : rects[3];
        if (hitable_y->hit(r, t_min, out.t, temp_closest))
        {
            hit_anything = true;
            out = temp_closest;
        }
        const auto& hitable_z = r.direction()[2] < 0 ? rects[0] : rects[1];
        if (hitable_z->hit(r, t_min, out.t, temp_closest))
        {
            hit_anything = true;
            out = temp_closest;
        }

        return hit_anything;
    }

    __device__ virtual bool bounding_box(float, float, aabb& box) const override
    {
        box = aabb(p_min, p_max);
        return true;
    }
};
} // namespace path_tracer
} // namespace ppt