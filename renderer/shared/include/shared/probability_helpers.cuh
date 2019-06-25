#pragma once
#include "cuda_helpers.cuh"
#include "shared/cuda_helpers.cuh"
#include "shared/probability_helpers.cuh"
#include <builtin_types.h>
#include <math.h>
#include <vector>

namespace ppt
{
namespace shared
{
#define RM(row, col, w) row* w + col
#define CM(row, col, h) col* h + row

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

__device__ __host__ inline float
online_variance_sum(float prev_vari_sum, float prev_mean, float new_mean, float new_val, size_t n)
{
    auto newS = prev_vari_sum + (new_val - prev_mean) * (new_val - new_mean);
    return n > 1 ? newS : 0.0;
}

__device__ __host__ inline float
online_variance(float prev_vari_sum, float prev_mean, float new_mean, float new_val, size_t n)
{
    return n > 1 ? online_variance_sum(prev_vari_sum, prev_mean, new_mean, new_val, n) / (n - 1) : 0.0;
}

__device__ __host__ inline float calc_mean_online_rollout(const float* const values, int n)
{
    auto mean = 0.0f;
    for (auto i = 0; i < n; i++)
    {
        mean = ppt::shared::online_mean(mean, values[i], i + 1);
    }
    return n > 0 ? mean : 0.0;
}

__device__ __host__ inline float calc_variance_online_rollout(const float* const values, int n)
{
    auto mean = 0.0f;
    auto vari = 0.0f;
    for (auto i = 0; i < n; i++)
    {
        auto new_mean = ppt::shared::online_mean(mean, values[i], i + 1);
        vari = ppt::shared::online_variance_sum(vari, mean, new_mean, values[i], i + 1);
        mean = new_mean;
    }
    return n > 1 ? vari / (n - 1) : 0.0;
}


} // namespace shared
} // namespace ppt