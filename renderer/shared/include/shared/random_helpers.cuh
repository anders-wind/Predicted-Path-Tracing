#pragma once
#include "vecs/vec3.cuh"
#include <curand_kernel.h>
#include <random>

namespace ppt
{
namespace shared
{

#define RANDVEC3(local_rand_state) \
    vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

#define RANDVEC2(local_rand_state) \
    vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0.0f)

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state)
{
    vec3 p;
    do
    {
        p = RANDVEC3(local_rand_state) * 2.0f - 1;
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 random_in_unit_disk(curandState* local_rand_state)
{
    vec3 p;
    do
    {
        p = RANDVEC2(local_rand_state) * 2.0f - vec3(1.0f, 1.0f, 0.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

} // namespace shared
} // namespace ppt