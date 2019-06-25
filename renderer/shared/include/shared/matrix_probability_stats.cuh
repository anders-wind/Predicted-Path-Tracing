#pragma once
#include <builtin_types.h>
#include <vector>

namespace ppt
{
namespace shared
{

/**
 * T supported types are float and vec3
 */
template <typename T>
class matrix_probability_stats
{
    private:
    // device arrays
    T* d_variance_sum;
    T* d_means;
    T* d_variance;
    float* d_variance_blocks;

    const unsigned int height;
    const unsigned int width;
    const dim3 block_dim;
    const dim3 thread_dim;

    public:
    matrix_probability_stats(size_t height, size_t width, const dim3& block_dim, const dim3& thread_dim);

    ~matrix_probability_stats();

    matrix_probability_stats(const matrix_probability_stats& other) = delete;
    matrix_probability_stats& operator=(const matrix_probability_stats& other) = delete;
    matrix_probability_stats(matrix_probability_stats&& other) = delete;
    matrix_probability_stats& operator=(matrix_probability_stats&& other) = delete;

    void update_variance(const T* const d_values, const unsigned int* const d_sample_count);
    void calc_variance_blocks();
    /**
     * Host memory, variance vector
     */
    std::vector<T> get_variance() const;
    std::vector<float> get_variance_blocks() const;

    float get_variance_mean() const;
};

} // namespace shared
} // namespace ppt