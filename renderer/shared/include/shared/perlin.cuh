#pragma once
#include "random_helpers.cuh"
#include "vecs/vec3.cuh"
#include <curand_kernel.h>

namespace ppt
{
namespace shared
{

__device__ static float* ranfloat;
__device__ static int* perm_x;
__device__ static int* perm_y;
__device__ static int* perm_z;

class perlin
{
    public:
    __device__ float noise(const vec3& p) const
    {
        // float u = p[0] - floor(p[0]);
        // float v = p[1] - floor(p[1]);
        // float w = p[2] - floor(p[2]);
        int i = int(4 * p[0]) & 255;
        int j = int(4 * p[1]) & 255;
        int k = int(4 * p[2]) & 255;
        return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
    }
};


__device__ float* perlin_generate(curandState* curand_state)
{
    constexpr int size = 256;
    float* p = new float[size];
    for (auto i = 0; i < size; i++)
    {
        p[i] = curand_uniform(curand_state);
    }
    return p;
}

__device__ void permute(int* p, int n, curandState* curand_state)
{
    for (auto i = n - 1; i >= 0; i--)
    {
        int target = int(curand_uniform(curand_state) * (i + 1));
        int tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;
    }
}

__device__ int* perlin_generate_perm(curandState* curand_state)
{
    constexpr int size = 256;
    int* p = new int[size];
    for (int i = 0; i < size; i++)
    {
        p[i] = i;
    }
    permute(p, size, curand_state);
    return p;
}


} // namespace shared
} // namespace ppt