
#include "shared/matrix_probability_stats.cuh"
#include "shared/probability_helpers.cuh"
#include "shared/vecs/vec3.cuh"
#include <cuda.h>
#include <device_launch_parameters.h>

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


template <>
matrix_probability_stats<float>::matrix_probability_stats(size_t height, size_t width, const dim3& block_dim, const dim3& thread_dim)
  : height(height), width(width), block_dim(block_dim), thread_dim(thread_dim)
{
    checkCudaErrors(cudaMalloc((void**)&d_means, width * height * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_variance_sum, width * height * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_variance, width * height * sizeof(float)));

    cudaMemset((void**)&d_means, 0, width * height);
    cudaMemset((void**)&d_variance_sum, 0, width * height);
    cudaMemset((void**)&d_variance, 0, width * height);
}

template <>
matrix_probability_stats<vec3>::matrix_probability_stats(size_t height, size_t width, const dim3& block_dim, const dim3& thread_dim)
  : height(height), width(width), block_dim(block_dim), thread_dim(thread_dim)
{
    checkCudaErrors(cudaMalloc((void**)&d_means, width * height * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_variance_sum, width * height * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_variance, width * height * sizeof(vec3)));

    checkCudaErrors(cudaMemset(d_means, 0, width * height));
    checkCudaErrors(cudaMemset(d_variance_sum, 0, width * height));
    checkCudaErrors(cudaMemset(d_variance, 0, width * height));
}

template <> matrix_probability_stats<vec3>::~matrix_probability_stats()
{
    checkCudaErrors(cudaFree(d_variance_sum));
    checkCudaErrors(cudaFree(d_means));
    checkCudaErrors(cudaFree(d_variance));
}
template <> matrix_probability_stats<float>::~matrix_probability_stats()
{
    checkCudaErrors(cudaFree(d_variance_sum));
    checkCudaErrors(cudaFree(d_means));
    checkCudaErrors(cudaFree(d_variance));
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

template <> std::vector<float> matrix_probability_stats<float>::get_variance() const
{
    auto h_variance = std::vector<float>(width * height);
    h_variance.resize(width * height);
    auto bytes = sizeof(float) * width * height;
    checkCudaErrors(cudaMemcpy(&h_variance[0], d_variance, bytes, cudaMemcpyDeviceToHost));
    return h_variance;
}

template <> std::vector<vec3> matrix_probability_stats<vec3>::get_variance() const
{
    auto h_variance = std::vector<vec3>(width * height);
    h_variance.resize(width * height);
    auto bytes = sizeof(vec3) * width * height;
    checkCudaErrors(cudaMemcpy(&h_variance[0], d_variance, bytes, cudaMemcpyDeviceToHost));
    return h_variance;
}

template <> float matrix_probability_stats<vec3>::get_variance_sum() const
{
    const auto vari = get_variance();
    auto sum = 0.0f;
    for (auto i = 0; i < vari.size(); i++)
    {
        sum += vari[i].sum();
    }
    return sum;
}


} // namespace shared
} // namespace ppt
