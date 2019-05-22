#pragma once
#include "vec3.cuh"
#include "ray.cuh"
#include "hitable.cuh"
#include "random_helpers.cuh"

namespace ppt::path_tracer
{
struct material
{
	__device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const = 0;
};

struct lambertian : public material
{
	rgb albedo;
	__device__ lambertian(const rgb &a) : albedo(a)
	{
	}

	__device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override
	{
		vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state) / 1.2f;
		scattered = ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}
};

struct metal : public material
{
	rgb albedo;
	float fuzz;

	__device__ metal(const vec3 &a, float f) : albedo(a)
	{
		if (f < 1)
		{
			fuzz = f;
		}
		else
		{
			fuzz = 1;
		}
	}

	__device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override
	{
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + random_in_unit_sphere(local_rand_state) * fuzz);
		attenuation = albedo;
		return dot(scattered.direction(), rec.normal) > 0;
	}
};

struct dielectric : public material
{
	float _ref_idx;
	__device__ dielectric(float ri) : _ref_idx(ri)
	{
	}

	__device__ inline float schlick(float cosine) const
	{
		float r0 = (1 - _ref_idx) / (1 + _ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}

	__device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override
	{
		vec3 outward_normal, refracted;
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		attenuation = vec3(1.0, 1.0, 1.0);
		float reflect_prob, ni_over_nt, cosine;

		if (dot(r_in.direction(), rec.normal) > 0.0f)
		{
			outward_normal = -rec.normal;
			ni_over_nt = _ref_idx;
			cosine = _ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
		}
		else
		{
			outward_normal = rec.normal;
			ni_over_nt = 1.0f / _ref_idx;
			cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
		}

		if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
		{
			reflect_prob = schlick(cosine);
		}
		else
		{
			reflect_prob = 1.0f;
		}

		if (curand_uniform(local_rand_state) < reflect_prob)
		{
			scattered = ray(rec.p, reflected);
		}
		else
		{
			scattered = ray(rec.p, refracted);
		}
		return true;
	}
};
} // namespace ppt::path_tracer
