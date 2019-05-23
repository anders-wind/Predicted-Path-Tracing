#pragma once

#include "cuda_runtime.h"
#include <builtin_types.h>
#include <iostream>

namespace ppt
{
namespace shared
{

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
}
} // namespace ppt

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)