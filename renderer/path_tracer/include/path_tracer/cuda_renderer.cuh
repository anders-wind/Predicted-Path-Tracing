#pragma once
// <Auto-Generated>
// This is here so CodeMaid doesn't reorganize this document
// </Auto-Generated>

#include "camera.cuh"
#include "hitable.cuh"
#include "hitable_list.cuh"
#include "material.cuh"
#include "ray.cuh"
#include "render.cuh"
#include "sphere.cuh"
#include <cuda.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <limits>
#include <random>
#include <shared/cuda_helpers.cuh>
#include <shared/random_helpers.cuh>
#include <shared/render_datapoint.cuh>
#include <shared/scoped_timer.cuh>
#include <shared/vec3.cuh>
#include <time.h>
#include <vector>

/**
 *
 * Goto the botton for the definition of the cuda_renderer class
 *
 */

namespace ppt
{
namespace path_tracer
{
#define RM(row, col, w) row* w + col
#define CM(row, col, h) col* h + row

#define RM3(row, col, w) 3 * row* w + 3 * col
#define CM3(row, col, h) 3 * col* h + 3 * row
constexpr auto FLOAT_MAX = 1000000000.0f;


namespace cuda_methods
{

__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < 10; i++)
    {
        hit_record rec;
        if (!(*world)->hit(cur_ray, 0.001f, FLOAT_MAX, rec))
        {
            break;
        }

        ray scattered;
        vec3 attenuation;
        if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
        {
            cur_attenuation *= attenuation;
            cur_ray = scattered;
        }
    }
    vec3 unit_direction = unit_vector(cur_ray.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    vec3 c = vec3(1.0, 1.0, 1.0) * (1.0f - t) + vec3(0.5, 0.7, 1.0) * t;
    return c * cur_attenuation;
}

__global__ void
render_image(vec3* image_matrix, int max_x, int max_y, int samples, camera** camera, hitable** world, curandState* rand_state)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    curandState local_rand_state = rand_state[pixel_index];

    vec3 pix = image_matrix[pixel_index];
    for (int s = 0; s < samples; s++)
    {
        float u = float(col + curand_normal(&local_rand_state)) / float(max_x);
        float v = float(max_y - row + curand_normal(&local_rand_state)) / float(max_y);
        ray r = (*camera)->get_ray(u, v);
        pix += color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    image_matrix[pixel_index] = pix;
}

__global__ void normalize(vec3* image_matrix, int max_x, int max_y, int samples)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    image_matrix[pixel_index] = (image_matrix[pixel_index] / float(samples)).v_sqrt();
}

__global__ void normalize_out(vec3* image_matrix, vec3* out_image_matrix, int max_x, int max_y, int samples)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    out_image_matrix[pixel_index] = (image_matrix[pixel_index] / float(samples)).v_sqrt();
}


__global__ void render_init(int max_x, int max_y, int offset, curandState* rand_state)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(row, col, offset, &rand_state[pixel_index]);
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new sphere(vec3(-1, 0.0, -1), 0.5, new dielectric(1.5f));
        d_list[4] = new sphere(vec3(-1, 0.0, -1), -0.45, new dielectric(1.5f));
        *d_world = new hitable_list(d_list, 5);
        *d_camera = camera_factory().make_16_9_camera();
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera)
{
    for (int i = 0; i < 5; i++)
    {
        delete ((sphere*)d_list[i])->_material;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}
} // namespace cuda_methods

class cuda_renderer
{
    curandState* d_rand_state;
    hitable** d_list;
    hitable** d_world;
    camera** d_camera;

    public:
    int num_threads_x = 32;
    int num_threads_y = 32;
    dim3 blocks;
    dim3 threads;
    const size_t w;
    const size_t h;

    cuda_renderer(int w, int h)
      : blocks(h / num_threads_y + 1, w / num_threads_x + 1), threads(num_threads_x, num_threads_y), w(w), h(h)
    {
        const auto timer = shared::scoped_timer("cuda_renderer");

        checkCudaErrors(cudaMalloc((void**)&d_rand_state, w * h * sizeof(curandState)));
        checkCudaErrors(cudaMalloc((void**)&d_list, 5 * sizeof(hitable*)));
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
        checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // init random
        cuda_methods::render_init<<<blocks, threads>>>(w, h, 0, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        cuda_methods::create_world<<<1, 1>>>(d_list, d_world, d_camera);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~cuda_renderer()
    {
        checkCudaErrors(cudaDeviceSynchronize());
        cuda_methods::free_world<<<1, 1>>>(d_list, d_world, d_camera);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaFree(d_list));
        checkCudaErrors(cudaFree(d_world));
    }

    public: // Methods
    render ray_trace(int samples)
    {
        const auto timer = shared::scoped_timer("ray_trace");

        auto ray_traced_image = render(w, h);
        auto* image_matrix = ray_traced_image.get_image_matrix();

        cuda_methods::render_image<<<blocks, threads>>>(image_matrix, w, h, samples, d_camera, d_world, d_rand_state);
        // cuda_methods::render_image<<<blocks, threads>>>(image_matrix, w, h, samples / 2, d_camera, d_world, d_rand_state);

        checkCudaErrors(cudaDeviceSynchronize());
        cuda_methods::normalize<<<blocks, threads>>>(image_matrix, w, h, samples);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        return ray_traced_image;
    }

    shared::render_datapoint ray_trace_datapoint(int samples[4])
    {
        const auto timer = shared::scoped_timer("ray_trace_datapoint");
        auto result = shared::render_datapoint(w, h);

        auto ray_traced_image = render(w, h);
        auto* image_matrix = ray_traced_image.get_image_matrix();

        auto out_ray_traced_image = render(w, h);
        auto* out_image_matrix = out_ray_traced_image.get_image_matrix();
        auto sample_sum = 0;
        for (auto i = 0; i < 4; i++)
        {
            const auto timer_intern = shared::scoped_timer("   _iteration");

            int sample = samples[i];
            sample_sum += sample;

            cuda_methods::render_image<<<blocks, threads>>>(image_matrix, w, h, sample, d_camera, d_world, d_rand_state);
            checkCudaErrors(cudaDeviceSynchronize());

            cuda_methods::normalize_out<<<blocks, threads>>>(image_matrix, out_image_matrix, w, h, sample_sum);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            if (i == 3)
            {
                result.target = out_ray_traced_image.get_vector3_representation();
            }
            else
            {
                result.renders[i] = out_ray_traced_image.get_vector5_representation();
            }
        }
        return result;
    }
};

} // namespace path_tracer
} // namespace ppt