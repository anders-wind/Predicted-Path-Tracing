#pragma once
#include <builtin_types.h>
#include <vector>

namespace ppt
{
namespace shared
{
template <typename T> class matrix_probability_stats
{
    private:
    // device arrays
    T* d_variance_sum;
    T* d_means;
    T* d_variance;

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

    void update_variance(const T* const d_values, const unsigned int* const d_sample_count);

    std::vector<T> get_variance() const;
};

} // namespace shared
} // namespace ppt