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

    void** d_this;

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
};

} // namespace path_tracer
} // namespace ppt
