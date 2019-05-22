#pragma once
#include <random>
#include "vec3.cuh"
#include "ray.cuh"

namespace ppt::path_tracer
{

struct camera
{
	vec3 _origin;
	vec3 _lower_left_corner;
	vec3 _horizontal;
	vec3 _vertical;

	__host__ __device__ camera()
	{
		_lower_left_corner = vec3(-2.0, -1.0, -1.0);
		_horizontal = vec3(4.0, 0.0, 0.0);
		_vertical = vec3(0.0, 2.0, 0.0);
		_origin = vec3(0.0, 0.0, 0.0);
	}

	__device__ ray get_ray(float u, float v)
	{
		return ray(_origin, _lower_left_corner + (_horizontal * u) + (_vertical * v) - _origin);
	}
};
} // namespace ppt::path_tracer
