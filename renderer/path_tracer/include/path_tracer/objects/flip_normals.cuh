#pragma once
#include "hitable.cuh"

namespace ppt
{
namespace path_tracer
{

struct flip_normals : public hitable
{
    private:
    hitable* inner;

    public:
    __device__ flip_normals(hitable* inner) : inner(inner)
    {
    }

    __device__ ~flip_normals()
    {
        delete inner;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        auto result = inner->hit(r, t_min, t_max, out);
        out.normal *= -1;
        return result;
    }

    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        return inner->bounding_box(t0, t1, box);
    }
};


} // namespace path_tracer
} // namespace ppt
