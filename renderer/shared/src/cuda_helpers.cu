#include <shared/cuda_helpers.cuh>

namespace ppt
{
namespace shared
{

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";

        std::cerr << "this here: " << cudaGetErrorString(result) << std::endl;
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

} // namespace shared
} // namespace ppt