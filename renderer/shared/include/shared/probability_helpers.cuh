#pragma once
#include <math.h>
#include <vector>

namespace ppt
{
namespace shared
{

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

__device__ __host__ inline float calc_mean(const float* const values, int n)
{
    if (n == 0)
    {
        return 0;
    }

    float sum = 0.0f;
    for (auto i = 0; i < n; i++)
    {
        sum += values[i];
    }
    const float mean = sum / float(n);
    return mean;
}


__device__ __host__ inline float calc_variance(const float* const values, int n)
{
    if (n == 0)
    {
        return 0;
    }

    const auto mean = calc_mean(values, n);
    auto vari = 0.0f;
    for (auto i = 0; i < n; i++)
    {
        const auto val = (values[i] - mean);
        vari += val * val;
    }
    return vari / float(n - 1);
}

__device__ __host__ inline float calc_standard_dev(const float* const values, int n)
{
    return std::sqrt(calc_variance(values, n));
}

__device__ __host__ inline float online_mean(float prev_mean, float new_val, size_t n)
{
    auto newM = prev_mean + (new_val - prev_mean) / n;
    return n > 0 ? newM : 0.0;
}

__device__ __host__ inline float online_variance_sum(float prev_vari, float prev_mean, float new_val, size_t n)
{
    const auto new_mean = online_mean(prev_mean, new_val, n);
    auto newS = prev_vari + (new_val - prev_mean) * (new_val - new_mean);
    return n > 1 ? newS : 0.0;
}

__device__ __host__ inline float online_variance(float prev_vari, float prev_mean, float new_val, size_t n)
{
    return n > 1 ? online_variance_sum(prev_vari, prev_mean, new_val, n) / (n - 1) : 0.0;
}

} // namespace shared
} // namespace ppt