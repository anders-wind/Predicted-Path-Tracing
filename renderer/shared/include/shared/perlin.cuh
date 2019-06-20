#pragma once
#include "random_helpers.cuh"
#include "vecs/vec3.cuh"
#include <curand_kernel.h>

namespace ppt
{
namespace shared
{
__device__ inline float trilinear_interp(vec3 c[2][2][2], float u, float v, float w)
{
    const auto uu = u * u * (3 - 2 * u);
    const auto vv = v * v * (3 - 2 * v);
    const auto ww = w * w * (3 - 2 * w);
    float accum = 0.0f;
    for (auto i = 0; i < 2; i++)
    {
        for (auto j = 0; j < 2; j++)
        {
            for (auto k = 0; k < 2; k++)
            {
                auto weight = vec3(u - i, v - j, w - k);
                accum += (i * uu + (1.0f - i) * (1.0f - uu)) * //
                         (j * vv + (1.0f - j) * (1.0f - vv)) * //
                         (k * ww + (1.0f - k) * (1.0f - ww)) * //
                         vec3::dot(c[i][j][k], weight);
            }
        }
    }
    return accum;
}

__device__ static vec3* ranvec;
__device__ static int* perm_x;
__device__ static int* perm_y;
__device__ static int* perm_z;

class perlin
{
    public:
    __device__ float noise(const vec3& p) const
    {
        int i = floor(p[0]);
        int j = floor(p[1]);
        int k = floor(p[2]);
        float u = p[0] - i;
        float v = p[1] - j;
        float w = p[2] - k;
        vec3 c[2][2][2];
        for (auto di = 0; di < 2; di++)
        {
            const auto i_p = perm_x[(i + di) & 255];
            for (auto dj = 0; dj < 2; dj++)
            {
                const auto j_p = perm_y[(j + dj) & 255];
                for (auto dk = 0; dk < 2; dk++)
                {
                    const auto k_p = perm_z[(k + dk) & 255];
                    c[di][dj][dk] = ranvec[i_p ^ j_p ^ k_p];
                }
            }
        }
        return trilinear_interp(c, u, v, w);
    }

    __device__ float turb(const vec3& p, int depth = 7) const
    {
        auto accum = 0.0f;
        auto temp_p = p;
        float weight = 1.0;
        for (auto i = 0; i < depth; i++)
        {
            accum += weight * noise(p);
            weight *= 0.5;
            temp_p *= 2;
        }
        return std::fabs(accum);
    }
};


__device__ vec3* perlin_generate(curandState* curand_state)
{
    constexpr int size = 256;
    auto p = new vec3[size];
    for (auto i = 0; i < size; i++)
    {
        p[i] = vec3::unit_vector(RANDVEC3(curand_state) * 2 - 1);
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
    auto p = new int[size];
    for (int i = 0; i < size; i++)
    {
        p[i] = i;
    }
    permute(p, size, curand_state);
    return p;
}


} // namespace shared
} // namespace ppt