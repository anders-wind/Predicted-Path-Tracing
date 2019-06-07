#pragma once
#include <experimental/optional>
#include <math.h>
#include <random>
#include <vector>

namespace ppt
{
namespace shared
{

class sample_service
{
    private:
    std::random_device rd;
    std::mt19937 generator;

    public:
    sample_service() : generator(rd())
    {
        generator.seed(42);
    }

    /**
     * Sample service which returns random numbers in powers
     * Example base 10: 1-9, 10-90, 100-900, and then samplesum
     */
    std::vector<int> get_samples_in_pow(int number_of_samples, int total_sum, int pow_base = 10)
    {
        auto result = std::vector<int>(number_of_samples);
        auto sample_sum = 0;

        auto prev = 1;
        for (auto i = 0; i < number_of_samples; i++)
        {
            auto is_last = i == (number_of_samples - 1);

            auto next = prev * pow_base;
            auto distribution = std::uniform_int_distribution<int>(prev, next - prev); // prev to ensure sum stays below next exponent.

            int sample = !is_last ? distribution(generator) : std::max(total_sum - sample_sum, 0);
            result[i] = sample;

            sample_sum += sample;
            prev = next;
        }
        return result;
    }
}; // namespace shared

} // namespace shared
} // namespace ppt