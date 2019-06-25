
#include "shared/matrix_probability_stats.cuh"
#include "shared/probability_helpers.cuh"
#include "shared/vecs/vec3.cuh"
#include <algorithm>
#include <builtin_types.h>
#include <shared/cuda_reduction.cuh>

namespace ppt
{
namespace shared
{

#define RM(row, col, w) row* w + col
#define CM(row, col, h) col* h + row


__global__ void calc_variance_online_2d(float* variance_sum,
                                        float* variance,
                                        float* means,
                                        const float* const values,
                                        const unsigned int* const sample_count,
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

__global__ void calc_variance_online_2d(vec3* variance_sum,
                                        vec3* variance,
                                        vec3* means,
                                        const vec3* const values,
                                        const unsigned int* const sample_count,
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

    vec3 new_mean(0.0f);
    vec3 new_vari(0.0f);
    for (auto i = 0; i < 3; i++)
    {
        new_mean[i] = online_mean(prev_mean[i], value[i], samples);
        new_vari[i] = online_variance(prev_vari_sum[i], prev_mean[i], new_mean[i], value[i], samples);
    }

    variance_sum[idx] = new_vari;
    variance[idx] = new_vari / (samples - 1);
    means[idx] = new_mean;
}

__global__ void
variance_block_sum(const vec3* const variance, float* variance_block, unsigned int max_x, unsigned int max_y)
{
    extern __shared__ vec3 sdata[];
    const auto row = threadIdx.x + blockIdx.x * blockDim.x;
    const auto col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    const auto idx = RM(row, col, max_x);
    const auto local_idx = RM(threadIdx.x, threadIdx.y, blockDim.x);
    const auto block_idx = RM(blockIdx.x, blockIdx.y, gridDim.x);
    const auto block_size = blockDim.x * blockDim.y;

    sdata[local_idx] = variance[idx];
    __syncthreads();

    for (auto s = block_size / 2; s > 32; s >>= 1)
    {
        if (local_idx < s)
        {
            sdata[local_idx] += sdata[local_idx + s];
        }
        __syncthreads();
    }

    if (local_idx < 32)
    {
        sdata[local_idx] += sdata[local_idx + 32];
        __syncthreads();
        sdata[local_idx] += sdata[local_idx + 16];
        __syncthreads();
        sdata[local_idx] += sdata[local_idx + 8];
        __syncthreads();
        sdata[local_idx] += sdata[local_idx + 4];
        __syncthreads();
        sdata[local_idx] += sdata[local_idx + 2];
        __syncthreads();
        sdata[local_idx] += sdata[local_idx + 1];
    }
    __syncthreads();
    if (local_idx == 0)
    {
        variance_block[block_idx] = sdata[0].sum();
        // printf("test %d, %f \n", block_idx, sdata[local_idx][0]);
    }
}


template <>
matrix_probability_stats<float>::matrix_probability_stats(size_t height, size_t width, const dim3& block_dim, const dim3& thread_dim)
  : height(height), width(width), block_dim(block_dim), thread_dim(thread_dim)
{
    if (thread_dim.x * thread_dim.y < 64)
    {
        std::cerr << "Thread x *y must be more than 64 for now, i.e 8x8" << std::endl;
        exit(1);
    }
    checkCudaErrors(cudaMalloc((void**)&d_means, width * height * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_variance_sum, width * height * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_variance, width * height * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_variance_blocks, block_dim.x * block_dim.y * sizeof(float)));

    checkCudaErrors(cudaMemset((void**)&d_means, 0, width * height));
    checkCudaErrors(cudaMemset((void**)&d_variance_sum, 0, width * height));
    checkCudaErrors(cudaMemset((void**)&d_variance, 0, width * height));
    checkCudaErrors(cudaMemset((void**)&d_variance_blocks, 0, block_dim.x * block_dim.y));
}

template <>
matrix_probability_stats<vec3>::matrix_probability_stats(size_t height, size_t width, const dim3& block_dim, const dim3& thread_dim)
  : height(height), width(width), block_dim(block_dim), thread_dim(thread_dim)
{
    checkCudaErrors(cudaMalloc((void**)&d_means, width * height * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_variance_sum, width * height * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_variance, width * height * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_variance_blocks, block_dim.x * block_dim.y * sizeof(float)));

    checkCudaErrors(cudaMemset(d_means, 0, width * height));
    checkCudaErrors(cudaMemset(d_variance_sum, 0, width * height));
    checkCudaErrors(cudaMemset(d_variance, 0, width * height));
    checkCudaErrors(cudaMemset(d_variance_blocks, 0, block_dim.x * block_dim.y));
}

template <>
matrix_probability_stats<vec3>::~matrix_probability_stats()
{
    checkCudaErrors(cudaFree(d_variance_sum));
    checkCudaErrors(cudaFree(d_means));
    checkCudaErrors(cudaFree(d_variance));
    checkCudaErrors(cudaFree(d_variance_blocks));
}
template <>
matrix_probability_stats<float>::~matrix_probability_stats()
{
    checkCudaErrors(cudaFree(d_variance_sum));
    checkCudaErrors(cudaFree(d_means));
    checkCudaErrors(cudaFree(d_variance));
    checkCudaErrors(cudaFree(d_variance_blocks));
}

template <>
void matrix_probability_stats<float>::update_variance(const float* const d_values,
                                                      const unsigned int* const d_sample_count)
{
    calc_variance_online_2d<<<block_dim, thread_dim>>>(
        d_variance_sum, d_variance, d_means, d_values, d_sample_count, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

template <>
void matrix_probability_stats<vec3>::update_variance(const vec3* const d_values, const unsigned int* const d_sample_count)
{
    calc_variance_online_2d<<<block_dim, thread_dim>>>(
        d_variance_sum, d_variance, d_means, d_values, d_sample_count, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

template <>
std::vector<float> matrix_probability_stats<float>::get_variance() const
{
    auto h_variance = std::vector<float>(width * height);
    h_variance.resize(width * height);
    auto bytes = sizeof(float) * width * height;
    checkCudaErrors(cudaMemcpy(&h_variance[0], d_variance, bytes, cudaMemcpyDeviceToHost));
    return h_variance;
}

template <>
std::vector<vec3> matrix_probability_stats<vec3>::get_variance() const
{
    auto h_variance = std::vector<vec3>(width * height);
    h_variance.resize(width * height);
    auto bytes = sizeof(vec3) * width * height;
    checkCudaErrors(cudaMemcpy(&h_variance[0], d_variance, bytes, cudaMemcpyDeviceToHost));
    return h_variance;
}

template <>
std::vector<float> matrix_probability_stats<vec3>::get_variance_blocks() const
{
    auto h_variance = std::vector<float>(block_dim.x * block_dim.y);
    h_variance.resize(block_dim.x * block_dim.y);
    auto bytes = sizeof(float) * block_dim.x * block_dim.y;
    checkCudaErrors(cudaMemcpy(&h_variance[0], d_variance_blocks, bytes, cudaMemcpyDeviceToHost));
    return h_variance;
}


template <>
float matrix_probability_stats<vec3>::get_variance_mean() const
{
    variance_block_sum<<<block_dim, thread_dim, thread_dim.x * thread_dim.y * sizeof(vec3)>>>(
        d_variance, d_variance_blocks, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto vari = get_variance_blocks();

    auto sum = 0.0f;
    for (auto i = 0; i < vari.size(); i++)
    {
        sum += vari[i];
    }
    return sum / float(height * width * 3);
}


} // namespace shared
} // namespace ppt
