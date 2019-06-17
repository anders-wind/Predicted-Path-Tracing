#pragma once
#include "hitable.cuh"
#include <shared/cuda_helpers.cuh>

namespace ppt
{
namespace path_tracer
{
// todo bench if these are faster than std::fmax()
inline float ffmin(float a, float b)
{
    return a < b ? a : b;
}
inline float ffmax(float a, float b)
{
    return a > b ? a : b;
}

/**
 * Axis Alignbed Bounding Box
 */
class aabb
{
    private:
    vec3 _min;
    vec3 _max;

    public:
    __device__ __host__ aabb()
    {
    }

    __device__ __host__ aabb(const vec3& min, const vec3& max) : _min(min), _max(max)
    {
    }

    __device__ __host__ aabb(const aabb& box0, const aabb& box1)
    {
        _min = vec3::min(box0._min, box1._min);
        _max = vec3::max(box0._max, box1._max);
    }

    __device__ __host__ vec3 min() const
    {
        return _min;
    }
    __device__ __host__ vec3 max() const
    {
        return _max;
    }

    __device__ __host__ bool hit(const ray& r, float t_min, float t_max) const
    {
        auto hit_anything = true;
        auto origin = r.origin();
        auto direction = r.direction();

        for (auto a = 0; a < 3; a++)
        {
            // concept:
            float t0 = std::fmin((_min[a] - origin[a]) / direction[a], (_max[a] - origin[a]) / direction[a]);
            float t1 = std::fmax((_min[a] - origin[a]) / direction[a], (_max[a] - origin[a]) / direction[a]);
            t_min = std::fmax(t0, t_min);
            t_min = std::fmin(t1, t_max);

            // performance improved
            // float inv_d = 1.0f / direction[a];
            // float t0 = (_min[a] - origin[a]) * inv_d;
            // float t1 = (_max[a] - origin[a]) * inv_d;

            // if (inv_d < 0.0f)
            // {
            //     // swap
            //     auto temp = t0;
            //     t0 = t1;
            //     t1 = temp;
            // }
            // t_min = t0 > t_min ? t0 : t_min;
            // t_max = t1 < t_max ? t1 : t_max;

            hit_anything &= t_max <= t_min;
        }
        return hit_anything;
    }
};

// Todo make a quick sort instead
__device__ __host__ void box_bubble_sort(hitable** arr, int n, int index)
{
    int i, j;
    aabb* boxes = new aabb[n];
    for (auto i = 0; i < n; i++)
    {
        boxes[i] = aabb();
        if (!arr[i]->bounding_box(0, 0, boxes[i]))
        {
            printf("bounding box fail");
        }
    }
    for (i = 0; i < n - 1; i++)
    {
        // Last i elements are already in place
        for (j = 0; j < n - i - 1; j++)
        {
            if (boxes[j].min()[index] < boxes[j + 1].min()[index])
            {
                const auto temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    delete[] boxes;
}
__device__ __host__ void box_x_bubble_sort(hitable** arr, int n)
{
    box_bubble_sort(arr, n, 0);
}
__device__ __host__ void box_y_bubble_sort(hitable** arr, int n)
{
    box_bubble_sort(arr, n, 1);
}
__device__ __host__ void box_z_bubble_sort(hitable** arr, int n)
{
    box_bubble_sort(arr, n, 2);
}

} // namespace path_tracer
} // namespace ppt