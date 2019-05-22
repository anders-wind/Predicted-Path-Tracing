#pragma once
#include <random>
#include <curand_kernel.h>
#include "vec3.cuh"

namespace ppt::shared
{
// std::random_device rd;	// Will be used to obtain a seed for the random number engine
//std::mt19937 gen(42);		// Standard mersenne_twister_engine seeded with rd()
//std::uniform_real_distribution<float> dis(0.0, 1.0);

__device__ const float gen = 0.5;
__device__ float dis(float value)
{
	return 0.5f;
}
//__device__ vec3 random_in_unit_sphere() {
//	vec3 p;
//	do {
//		p = vec3(dis(gen), dis(gen), dis(gen))*2.0 - vec3(1, 1, 1);
//	} while (p.squared_length() >= 1.0);
//	return p;
//}

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state)
{
	vec3 p;
	do
	{
		const auto rand_vec = vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
		p = (rand_vec * 2.0f) - vec3(1.0f, 1.0f, 1.0f);
	} while (p.squared_length() >= 1.0f);
	return p;
}
} // namespace ppt::shared