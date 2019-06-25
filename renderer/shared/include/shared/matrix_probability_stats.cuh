#pragma once
#include <builtin_types.h>

namespace ppt
{
namespace shared
{

class matrix_probability_stats
{
    private:
    // device arrays
    float* d_variance_sum;
    float* d_means;
    float* d_variance;

    const unsigned int height;
    const unsigned int width;
    const dim3 block_dim;
    const dim3 thread_dim;

    public:
    matrix_probability_stats(size_t height, size_t width, const dim3& block_dim, const dim3& thread_dim);

    ~matrix_probability_stats();

    matrix_probability_stats(const matrix_probability_stats& other) = delete;
    matrix_probability_stats operator=(const matrix_probability_stats& other) = delete;
    matrix_probability_stats(matrix_probability_stats&& other) = delete;
    matrix_probability_stats operator=(matrix_probability_stats&& other) = delete;

    void update_variance(const float* const d_values, const unsigned int* const d_sample_count);
};
} // namespace shared
} // namespace ppt