#pragma once
#include "random_helpers.cuh"
#include "vecs/vec3.cuh"
#include <curand_kernel.h>

namespace ppt
{
namespace shared
{
__device__ inline float trilinear_interp(float c[2][2][2], float u, float v, float w)
{
    float accum = 0.0f;
    for (auto i = 0; i < 2; i++)
    {
        for (auto j = 0; j < 2; j++)
        {
            for (auto k = 0; k < 2; k++)
            {
                accum += (i * u + (1.0f - i) * (1.0f - u)) * //
                         (j * v + (1.0f - j) * (1.0f - v)) * //
                         (k * w + (1.0f - k) * (1.0f - w)) * //
                         c[i][j][k];
            }
        }
    }
    return accum;
}

__device__ static float* ranfloat;
__device__ static int* perm_x;
__device__ static int* perm_y;
__device__ static int* perm_z;

class perlin
{
    public:
    __device__ float noise(const vec3& p) const
    {
        float u = p[0] - floor(p[0]);
        float v = p[1] - floor(p[1]);
        float w = p[2] - floor(p[2]);
        u = u * u * (3 - 2 * u);
        v = v * v * (3 - 2 * v);
        w = w * w * (3 - 2 * w);
        int i = int(4 * p[0]) & 255;
        int j = int(4 * p[1]) & 255;
        int k = int(4 * p[2]) & 255;
        float c[2][2][2];
        for (auto di = 0; di < 2; di++)
        {
            const auto i_p = perm_x[(i + di) & 255];
            for (auto dj = 0; dj < 2; dj++)
            {
                const auto j_p = perm_y[(j + dj) & 255];
                for (auto dk = 0; dk < 2; dk++)
                {
                    const auto k_p = perm_z[(k + dk) & 255];
                    c[di][dj][dk] = ranfloat[i_p ^ j_p ^ k_p];
                }
            }
        }
        return trilinear_interp(c, u, v, w);
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