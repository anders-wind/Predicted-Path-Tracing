#pragma once
#include "vecs/vec3.cuh"
#include <builtin_types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace ppt
{
namespace shared
{

template <unsigned int blockSize, typename T>
__device__ void warp_reduce(volatile T* sdata, unsigned int tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int shared_data[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    shared_data[tid] = 0;
    while (i < n)
    {
        shared_data[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            shared_data[tid] += shared_data[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            shared_data[tid] += shared_data[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            shared_data[tid] += shared_data[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
        warp_reduce<blockSize>(shared_data, tid);
    if (tid == 0)
        g_odata[blockIdx.x] = shared_data[0];
}

} // namespace shared
} // namespace ppt
