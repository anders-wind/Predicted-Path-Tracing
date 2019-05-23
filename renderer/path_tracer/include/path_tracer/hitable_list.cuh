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
    hitable** _hitables;
    size_t num_elements;

    __device__ hitable_list(){};
    __device__ hitable_list(hitable** list, size_t num_elements)
    {
        _hitables = list;
        this->num_elements = num_elements;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& out) const override
    {
        hit_record temp_closest;
        bool hit_anything = false;
        auto closest_so_far = t_max;
        for (auto i = 0; i < num_elements; i++)
        {
            const auto& hitable = _hitables[i];
            if (hitable->hit(r, t_min, closest_so_far, temp_closest))
            {
                hit_anything = true;
                closest_so_far = temp_closest.t;
                out = temp_closest;
            }
        }
        return hit_anything;
    }
};

} // namespace path_tracer
} // namespace ppt
