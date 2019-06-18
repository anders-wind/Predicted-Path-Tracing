#pragma once
#include "hitable.cuh"
#include "hitable_list.cuh"
namespace ppt
{
namespace path_tracer
{

/**
 * Bounding Volume Hierarchy
 */
class bvh_node : public hitable
{
    private:
    const hitable* left;
    const hitable* right;

    aabb box;

    public:
    __device__ __host__ bvh_node(hitable** l, int n, int axis = 0)
    {
        box_bubble_sort(l, n, axis % 3);
        const int half = n / 2;

        if (n == 1)
        {
            left = right = l[0];
        }
        else if (n == 2)
        {
            left = l[0];
            right = l[1];
        }
        else if (n < 25)
        {
            left = new hitable_list(l, half);
            right = new hitable_list(l + half, half);
        }
        else
        {
            left = new bvh_node(l, half, axis + 1);
            right = new bvh_node(l + half, n - half, axis + 1);
        }

        aabb box_left, box_right;
        if (!left->bounding_box(0, 0, box_left) || !right->bounding_box(0, 0, box_right))
        {
            printf("no bounding box in bvh_node construction\n");
        }
        box = aabb(box_left, box_right);
        printf("min: %f, %f, %f\n", box.min()[0], box.min()[1], box.min()[2]);
        printf("max: %f, %f, %f\n", box.max()[0], box.max()[1], box.max()[2]);
        printf("n: %d\n\n", n);
    }

    __device__ __host__ ~bvh_node()
    {
        if (left)
        {
            delete left;
        }
        if (right)
        {
            delete right;
        }
    }

    __device__ __host__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        if (box.hit(r, t_min, t_max))
        {
            hit_record left_rec, right_rec;
            const auto hit_left = left->hit(r, t_min, t_max, left_rec);
            const auto hit_right = right->hit(r, t_min, t_max, right_rec);
            if (hit_left && hit_right)
            {
                rec = left_rec.t < right_rec.t ? left_rec : right_rec;
                return true;
            }
            else if (hit_left)
            {
                rec = left_rec;
                return true;
            }
            else if (hit_right)
            {
                rec = right_rec;
                return true;
            }
        }
        return false;
    }

    __device__ __host__ virtual bool bounding_box(float t0, float t1, aabb& b) const override
    {
        b = this->box;
        return true;
    }
};


} // namespace path_tracer
} // namespace ppt