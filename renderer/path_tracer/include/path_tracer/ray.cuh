#pragma once

#include <shared/vecs/vec3.cuh>

namespace ppt
{
namespace path_tracer
{
using vec3 = ppt::shared::vec3;

struct ray
{
    vec3 _origin; // origin
    vec3 _direction; // direction

    __device__ __host__ ray(){};
    __device__ __host__ ray(const vec3& origin, const vec3& direction)
      : _origin(origin), _direction(direction)
    {
    }
    __device__ __host__ vec3 origin() const
    {
        return _origin;
    }
    __device__ __host__ vec3 direction() const
    {
        return _direction;
    }
    __device__ __host__ vec3 point_at_parameter(float t) const
    {
        return _origin + (_direction * t);
    }
};
} // namespace path_tracer
} // namespace ppt