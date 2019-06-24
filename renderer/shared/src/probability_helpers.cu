#include "shared/cuda_helpers.cuh"
#include "shared/probability_helpers.cuh"
#include <builtin_types.h>
#include <cuda.h>
#include <device_launch_parameters.h>
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

__global__ void calc_variance_online_1d(float* variance_sum,
                                        float* variance,
                                        float* means,
                                        float* values,
                                        unsigned int* sample_count,
                                        unsigned int max_x)
{
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < max_x)
    {
        const auto value = values[idx];
        const auto samples = sample_count[idx];
        const auto prev_mean = means[idx];
        const auto prev_vari_sum = variance_sum[idx];
        float new_mean = online_mean(prev_mean, value, samples);
        float vari = online_variance(prev_vari_sum, prev_mean, new_mean, value, samples);

        variance_sum[idx] = vari;
        variance[idx] = vari / (samples - 1);
        means[idx] = new_mean;
    }
}

__global__ void calc_variance_online_2d(float* variance_sum,
                                        float* variance,
                                        float* means,
                                        float* values,
                                        unsigned int* sample_count,
                                        unsigned int max_x,
                                        unsigned int max_y)
{
    const auto row = threadIdx.x + blockIdx.x * blockDim.x;
    const auto col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    const auto idx = RM(row, col, max_x);
    const auto value = values[idx];
    const auto samples = sample_count[idx];
    const auto prev_mean = means[idx];
    const auto prev_vari_sum = variance_sum[idx];
    float new_mean = online_mean(prev_mean, value, samples);
    float vari = online_variance(prev_vari_sum, prev_mean, new_mean, value, samples);

    variance_sum[idx] = vari;
    variance[idx] = vari / (samples - 1);
    means[idx] = new_mean;
}


probability_stat_collection::probability_stat_collection(size_t height, size_t width, const dim3& block_dim, const dim3& thread_dim)
  : height(height), width(width), block_dim(block_dim), thread_dim(thread_dim)
{
}

probability_stat_collection::~probability_stat_collection()
{
}

void probability_stat_collection::update_variance(float* d_values)
{
    calc_variance_online_2d<<<thread_dim, block_dim>>>(
        d_variance_sum, d_variance, d_means, d_values, d_sample_counts, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}


} // namespace shared
} // namespace ppt