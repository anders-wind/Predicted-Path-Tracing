#pragma once
#include "cuda_helpers.cuh"
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

__device__ __host__ inline float calc_mean(const float* const values, int n);

__device__ __host__ inline float calc_variance(const float* const values, int n);

__device__ __host__ inline float calc_standard_dev(const float* const values, int n);

__device__ __host__ inline float online_mean(float prev_mean, float new_val, size_t n);

__device__ __host__ inline float
online_variance_sum(float prev_vari_sum, float prev_mean, float new_mean, float new_val, size_t n);

__device__ __host__ inline float
online_variance(float prev_vari_sum, float prev_mean, float new_mean, float new_val, size_t n);

__device__ __host__ inline float calc_mean_online_rollout(const float* const values, int n);

__device__ __host__ inline float calc_variance_online_rollout(const float* const values, int n);


class probability_stat_collection
{
    private:
    // device arrays
    float* d_variance_sum;
    float* d_means;
    float* d_variance;
    unsigned int* d_sample_counts;

    const unsigned int height;
    const unsigned int width;
    const dim3 block_dim;
    const dim3 thread_dim;

    public:
    probability_stat_collection(size_t height, size_t width, const dim3& block_dim, const dim3& thread_dim);

    ~probability_stat_collection();

    probability_stat_collection(const probability_stat_collection& other) = delete;
    probability_stat_collection operator=(const probability_stat_collection& other) = delete;
    probability_stat_collection(probability_stat_collection&& other) = delete;
    probability_stat_collection operator=(probability_stat_collection&& other) = delete;

    void update_variance(float* d_values);
};


} // namespace shared
} // namespace ppt