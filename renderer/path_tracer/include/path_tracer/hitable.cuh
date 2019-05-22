#pragma once
#include "ray.cuh"

namespace ppt::path_tracer
{
// forward decleration
struct material;

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

struct hitable
{
	__device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &out) const = 0;
};
} // namespace ppt::path_tracer
