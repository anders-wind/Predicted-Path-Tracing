#pragma once

#include "vec3.cuh"
namespace ppt::path_tracer
{
struct ray
{
	vec3 _origin;	// origin
	vec3 _direction; // direction

	__device__ ray(){};
	__device__ ray(const vec3 &origin, const vec3 &direction) : _origin(origin), _direction(direction)
	{
	}
	__device__ vec3 origin() const { return _origin; }
	__device__ vec3 direction() const { return _direction; }
	__device__ vec3 point_at_parameter(float t) const { return _origin + (_direction * t); }
};
} // namespace ppt::path_tracer