#pragma once
#include "hitable.cuh"

namespace ppt
{
namespace path_tracer
{

struct translate : public hitable
{
    private:
    hitable* inner;
    vec3 displacement;

    public:
    __device__ translate(hitable* inner, const vec3& displacement)
      : inner(inner), displacement(displacement)
    {
    }

    __device__ ~translate()
    {
        delete inner;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        ray moved_r(r.origin() - displacement, r.direction());
        auto result = inner->hit(moved_r, t_min, t_max, out);
        out.p += displacement;
        return result;
    }

    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        auto result = inner->bounding_box(t0, t1, box);
        box = aabb(box.min() + displacement, box.max() + displacement);
        return result;
    }
};


} // namespace path_tracer
} // namespace ppt
