#pragma once
#include "hitable.cuh"
#include <memory>
#include <vector>

namespace ppt
{
namespace path_tracer
{

struct hitable_list : public hitable
{
    private:
    hitable** _hitables;
    size_t num_elements;

    public:
    __device__ __host__ hitable_list(hitable** list, size_t num_elements)
    {
        _hitables = list;
        this->num_elements = num_elements;
    }

    __device__ __host__ ~hitable_list()
    {
        for (auto i = 0; i < num_elements; i++)
        {
            delete _hitables[i];
        }
        delete _hitables;
    }

    __device__ __host__ int get_num_elements() const
    {
        return num_elements;
    }

    __device__ __host__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        hit_record temp_closest;
        auto hit_anything = false;
        out.t = t_max;
        for (auto i = 0; i < num_elements; i++)
        {
            const auto& hitable = _hitables[i];
            if (hitable->hit(r, t_min, out.t, temp_closest))
            {
                hit_anything = true;
                out = temp_closest;
            }
        }
        return hit_anything;
    }

    __device__ __host__ virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        if (num_elements <= 0)
        {
            return false;
        }

        aabb temp_box;
        bool first_true = _hitables[0]->bounding_box(t0, t1, temp_box);
        if (!first_true)
        {
            return false;
        }
        else
        {
            box = temp_box;
        }
        for (auto i = 1; i < num_elements; i++)
        {
            if (_hitables[i]->bounding_box(t0, t1, temp_box))
            {
                box = aabb(box, temp_box);
            }
            else
            {
                return false;
            }
        }
        return true;
    }
};

} // namespace path_tracer
} // namespace ppt
