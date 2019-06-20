#include "path_tracer/cuda_renderer.cuh"
#include "path_tracer/material.cuh"
#include "path_tracer/objects/aabb.cuh"
#include "path_tracer/objects/box.cuh"
#include "path_tracer/objects/bvh_node.cuh"
#include "path_tracer/objects/flip_normals.cuh"
#include "path_tracer/objects/hitable_list.cuh"
#include "path_tracer/objects/plane.cuh"
#include "path_tracer/objects/rect.cuh"
#include "path_tracer/objects/rotate.cuh"
#include "path_tracer/objects/sphere.cuh"
#include "path_tracer/objects/translate.cuh"
#include "path_tracer/ray.cuh"
#include <cuda.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <shared/perlin.cuh>
#include <shared/random_helpers.cuh>
#include <shared/vecs/vec3.cuh>
#include <shared/vecs/vec8.cuh>

namespace ppt
{
namespace path_tracer
{
namespace cuda_methods
{


__device__ vec3 color_rec(const ray& r, hitable** world, curandState* local_rand_state, float min_depth, float max_depth, int depth)
{
    vec3 emitted(0.00001f); // ambience
    vec3 attenuation;
    hit_record rec;
    ray scattered;
    if ((*world)->hit(r, min_depth, max_depth, rec))
    {
        emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
        if (depth >= 0 && rec.mat_ptr->scatter(r, rec, attenuation, scattered, local_rand_state))
        {
            return emitted +
                   attenuation * color_rec(scattered, world, local_rand_state, min_depth, max_depth, depth - 1);
        }
    }
    return emitted;
}

__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state, float min_depth, float max_depth)
{
    ray cur_ray = r;
    hit_record rec;
    vec3 cur_attenuation = vec3(1.0f);

    for (int i = 0; i < 5; i++)
    {
        if (!(*world)->hit(cur_ray, min_depth, max_depth, rec))
        {
            break;
        }

        ray scattered;
        vec3 attenuation;
        vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
        if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
        {
            cur_attenuation *= attenuation;
            cur_ray = scattered;
        }
    }
    return cur_attenuation;
}

__global__ void render_image(vec3* image_matrix,
                             int* samples,
                             int max_x,
                             int max_y,
                             int samples_for_pixel,
                             int sample_decrease_factor,
                             camera* cam,
                             hitable** world,
                             curandState* rand_state)
{
    int row = (threadIdx.x + blockIdx.x * blockDim.x) *
              (((sample_decrease_factor) * (1.0f + curand_uniform(rand_state))));
    int col = (threadIdx.y + blockIdx.y * blockDim.y) *
              (((sample_decrease_factor) * (1.0f + curand_uniform(rand_state))));
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    curandState local_rand_state = rand_state[pixel_index];

    vec3 pix = vec3(image_matrix[pixel_index].e);
    const camera local_camera(*cam);
    for (int s = 0; s < samples_for_pixel; s++)
    {
        float u = float(col + curand_normal(&local_rand_state)) / float(max_x);
        float v = float(max_y - row + curand_normal(&local_rand_state)) / float(max_y);
        ray r = local_camera.get_ray(u, v, &local_rand_state);
        pix += color_rec(r, world, &local_rand_state, local_camera._min_depth, local_camera._max_depth, 5);
    }

    samples[pixel_index] += samples_for_pixel;
    rand_state[pixel_index] = local_rand_state;
    image_matrix[pixel_index] = pix;
}

__device__ hit_record depth_map(hitable** world, camera* cam, int col, int row, int max_x, int max_y, curandState* rand_state)
{
    hit_record rec;
    float u = float(col) / float(max_x);
    float v = float(max_y - row) / float(max_y);
    const camera local_camera(*cam);

    ray r = local_camera.get_ray(u, v, rand_state);
    (*world)->hit(r, local_camera._min_depth, local_camera._max_depth, rec);
    return rec;
}

__global__ void post_process(vec3* image_matrix,
                             vec8* out_image_matrix,
                             hitable** world,
                             camera* camera,
                             int* samples,
                             int max_x,
                             int max_y,
                             curandState* rand_state)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    curandState local_rand_state = rand_state[pixel_index];

    auto in_pixel = image_matrix[pixel_index];
    auto samples_for_pixel = samples[pixel_index];
    auto norm_rgb = (vec3::clamp(vec3(in_pixel.e) / float(samples_for_pixel))).v_sqrt();

    auto sample_precision = __logf(samples_for_pixel) /
                            16.0f; // 2^16=65536 is our (arbitrarily) choosen max number of samples
    auto hit = depth_map(world, camera, col, row, max_x, max_y, &local_rand_state);
    auto depth = sqrtf(fabs(hit.t)) / sqrtf(camera->_max_depth);

    out_image_matrix[pixel_index] =
        vec8(norm_rgb, sample_precision, depth, hit.normal[0], hit.normal[1], hit.normal[2]);
}

__global__ void post_process_fast(vec3* image_matrix, vec8* out_image_matrix, int* samples, int max_x, int max_y)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    auto samples_for_pixel = samples[pixel_index];

    auto in_pixel = image_matrix[pixel_index];
    auto norm_rgb = (vec3::clamp(vec3(in_pixel.e) / float(samples_for_pixel))).v_sqrt();

    out_image_matrix[pixel_index] = vec8(norm_rgb, 0, 0, 0, 0, 0);
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

    auto local_rand_state = rand_state[0];
    if (row == 0 && col == 0)
    {
        shared::ranvec = shared::perlin_generate(&local_rand_state);
        shared::perm_x = shared::perlin_generate_perm(&local_rand_state);
        shared::perm_y = shared::perlin_generate_perm(&local_rand_state);
        shared::perm_z = shared::perlin_generate_perm(&local_rand_state);
    }
}

__global__ void reset_image(vec3* color_matrix, int* samples, int max_x, int max_y)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    color_matrix[pixel_index] = vec3(0, 0, 0);
    samples[pixel_index] = 0;
}

__global__ void
create_cornell_box(hitable** d_list, hitable** d_world, camera* d_camera, int hitables_size, curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        int i = 0;
        material* red = new lambertian(new constant_texture(vec3(0.65f, 0.05f, 0.05f)));
        material* metal = new lambertian(new constant_texture(vec3(0.73f)));
        material* white = new lambertian(new constant_texture(vec3(0.73f)));
        material* green = new lambertian(new constant_texture(vec3(0.12f, 0.45f, 0.15f)));
        material* light = new diffuse_light(new constant_texture(vec3(15.0f)));

        // room
        d_list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
        d_list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
        d_list[i++] = new xz_rect(213, 343, 227, 332, 554, light);
        d_list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
        d_list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
        d_list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));

        // boxes
        d_list[i++] =
            new translate(new rotate_y(new box(vec3(0.0f), vec3(165.0f), white), -18), vec3(130, 0, 65));
        d_list[i++] = new translate(new rotate_y(new box(vec3(0.0f), vec3(165, 330, 165), white), 15),
                                    vec3(265, 0, 295));

        *d_world = new bvh_node(d_list, hitables_size);
        // *d_world = new hitable_list(d_list, hitables_size);
        const auto look_from = vec3(278, 278, -800);
        const auto look_at = vec3(278, 278, 0);
        const auto focus_dist = 10;
        const auto aperture = 0.01;
        const auto fov = 40;
        *d_camera = camera_factory().make_16_9_camera(look_from, look_at, fov, aperture, focus_dist);
    }
}

__global__ void
create_small_world(hitable** d_list, hitable** d_world, camera* d_camera, int hitables_size, curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_list[0] = new sphere(vec3(0, -100.5, 0), 100, new lambertian(new noise_texture(10.0f)));
        // d_list[0] = new plane(vec3(0, -0.5, 0), vec3(0, 1, 0), new lambertian(new noise_texture(1.0f)));
        d_list[1] = new sphere(vec3(0, 2, 0), 2, new lambertian(new constant_texture(vec3(0.1, 0.2, 0.5))));
        d_list[2] = new sphere(vec3(0, 7, 0), 2, new diffuse_light(new constant_texture(vec3(4))));
        d_list[3] = new sphere(vec3(-1, 0.0, -1), 0.5, new dielectric(1.5f));
        d_list[4] = new sphere(vec3(-1, 0.0, -1), -0.45, new dielectric(1.5f));
        d_list[5] = new xy_rect(3, 5, 1, 3, -2, new diffuse_light(new constant_texture(9.0f)));


        *d_world = new bvh_node(d_list, hitables_size);
        // *d_world = new hitable_list(d_list, hitables_size);
        const auto look_from = vec3(13, 4, 8);
        const auto look_at = vec3(1, 2, 0);
        const auto focus_dist = 13;
        const auto fov = 45;
        const auto aperture = 0.1;
        *d_camera = camera_factory().make_16_9_camera(look_from, look_at, fov, aperture, focus_dist);
    }
}

__global__ void
create_world(hitable** d_list, hitable** d_world, camera* d_camera, int hitables_size, curandState* rand_state)
{
    // Based on the final image of raytracinginaweekend
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        auto local_rand_state = rand_state[0];
        auto checker = new checker_texture(new constant_texture(vec3(0.2, 0.3, 0.1)),
                                           new constant_texture(vec3(0.9, 0.9, 0.9)));
        auto noise = new noise_texture(10.0f);
        d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(noise));

        int i = 1;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float choose_mat = curand_uniform(&local_rand_state);
                vec3 center(a + 0.9 * curand_uniform(&local_rand_state), 0.2, b + 0.9 * curand_uniform(&local_rand_state));
                if ((center - vec3(4, 0.2, 0)).length() > 0.9)
                {
                    if (choose_mat < 0.8)
                    { // diffuse
                        d_list[i++] =
                            new sphere(center,
                                       0.2,
                                       new lambertian(new constant_texture(vec3(
                                           RANDVEC3(&local_rand_state) * RANDVEC3(&local_rand_state)))));
                    }
                    else if (choose_mat < 0.95)
                    { // metal
                        d_list[i++] =
                            new sphere(center,
                                       0.2,
                                       new metal(new constant_texture((RANDVEC3(&local_rand_state) + 1) * 0.5),
                                                 0.5 * curand_uniform(&local_rand_state)));
                    }
                    else
                    { // glass
                        d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                    }
                }
            }
        }

        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(checker));
        d_list[i++] =
            new sphere(vec3(4, 1, 0), 1.0, new metal(new constant_texture(vec3(0.7, 0.6, 0.5)), 0.0));

        *d_world = new bvh_node(d_list, hitables_size);
        //*d_world = new hitable_list(d_list, hitables_size);
        const auto look_from = vec3(13, 2, 3);
        const auto look_at = vec3(0, 0, 0);
        const auto focus_dist = 10.0f;
        *d_camera = camera_factory().make_16_9_camera(look_from, look_at, 20, 0.1, focus_dist);
    }
}

__global__ void create_random_world(hitable** d_list,
                                    hitable** d_world,
                                    camera* d_camera,
                                    int hitables_size,
                                    int reflection,
                                    int refraction,
                                    curandState* curand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        auto local_rand = &curand_state[0];
        auto number_of_reflection = 0;
        auto number_of_refraction = 0;
        for (auto i = 0; i < hitables_size; i++)
        {
            if (i == 0)
            {
                auto noise = new noise_texture(10.0f);
                d_list[i] = new plane(vec3(0, -3, 0), vec3(0, 1, 0), new lambertian(noise));
                continue;
            }

            material* mat;
            if (number_of_reflection < reflection)
            {
                mat = new metal(new constant_texture(RANDVEC3(local_rand)), curand_uniform(local_rand));
                number_of_reflection++;
            }
            else if (number_of_refraction < refraction)
            {
                mat = new dielectric(0.001f + fabs(curand_uniform(local_rand) - 0.001f));
                number_of_refraction++;
            }
            else
            {
                mat = new lambertian(new constant_texture(RANDVEC3(local_rand)));
            }
            d_list[i] = new sphere(vec3((curand_uniform(local_rand) - 0.5) * 14,
                                        (curand_uniform(local_rand) - 0.5) * 8,
                                        curand_uniform(local_rand) * -3 - 2),
                                   curand_uniform(local_rand) * curand_uniform(local_rand) * 1.5 + 0.3,
                                   mat);
        }

        *d_world = new bvh_node(d_list, hitables_size);
        //*d_world = new hitable_list(d_list, hitables_size);
        const auto look_from = vec3(0, 0, 0);
        const auto look_at = vec3(0, 0, -4);
        const auto focus_dist = (look_from - look_at).length();
        *d_camera = camera_factory().make_16_9_camera(look_from, look_at, 110, 0.2, focus_dist);
    }
}

__global__ void free_world(hitable** d_world, camera* d_camera)
{
    delete d_world;
    delete d_camera;
}
} // namespace cuda_methods


cuda_renderer::cuda_renderer(int w, int h, std::shared_ptr<shared::sample_service> sampler)
  : blocks(h / num_threads_y + 1, w / num_threads_x + 1)
  , threads(num_threads_x, num_threads_y)
  , w(w)
  , h(h)
  , _sampler(sampler)
{
    if (!_sampler)
    {
        throw std::runtime_error("cuda_renderer::ctor - sampler was not initialized");
    }

    const auto timer = shared::scoped_timer("cuda_renderer");

    int hitables_size = 8; // 485 for large, 8 for cornell
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, w * h * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&d_list, hitables_size * sizeof(hitable*)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera)));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // init random
    cuda_methods::render_init<<<blocks, threads>>>(w, h, 0, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cuda_methods::create_cornell_box<<<1, 1>>>(d_list, d_world, d_camera, hitables_size, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

cuda_renderer::~cuda_renderer()
{
    checkCudaErrors(cudaDeviceSynchronize());
    cuda_methods::free_world<<<1, 1>>>(d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
}

std::shared_ptr<render> cuda_renderer::ray_trace(int samples, int sample_sum, bool fast) const
{
    const auto timer = shared::scoped_timer("ray_trace");

    auto ray_traced_image = std::make_shared<render>(w, h);
    ray_trace(samples, sample_sum, *ray_traced_image, fast);
    return ray_traced_image;
}

void cuda_renderer::ray_trace(int samples, int sample_sum, render& ray_traced_image, bool fast) const
{
    {
        // const auto timer = shared::scoped_timer("ray_tracing");
        constexpr int reduce_factor = 3;
        const dim3 blocks_decrease =
            dim3(h / (reduce_factor * num_threads_y) + 1, w / (reduce_factor * num_threads_x) + 1);

        cuda_methods::render_image<<<blocks_decrease, threads>>>(ray_traced_image.get_color_matrix(),
                                                                 ray_traced_image.get_sample_matrix(),
                                                                 w,
                                                                 h,
                                                                 samples,
                                                                 reduce_factor,
                                                                 d_camera,
                                                                 d_world,
                                                                 d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    {
        // const auto timer = shared::scoped_timer("post process");

        if (fast)
        {
            cuda_methods::post_process_fast<<<blocks, threads>>>(ray_traced_image.get_color_matrix(),
                                                                 ray_traced_image.get_image_matrix(),
                                                                 ray_traced_image.get_sample_matrix(),
                                                                 w,
                                                                 h);
        }
        else
        {
            cuda_methods::post_process<<<blocks, threads>>>(ray_traced_image.get_color_matrix(),
                                                            ray_traced_image.get_image_matrix(),
                                                            d_world,
                                                            d_camera,
                                                            ray_traced_image.get_sample_matrix(),
                                                            w,
                                                            h,
                                                            d_rand_state);
        }
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
}


std::vector<path_tracer::render_datapoint> cuda_renderer::ray_trace_datapoints(size_t number_of_images)
{
    const auto timer = shared::scoped_timer("ray_trace_datapoint");
    auto ray_traced_image = render(w, h);
    auto results = std::vector<path_tracer::render_datapoint>();
    results.reserve(number_of_images);

    for (auto i = 0; i < number_of_images; i++)
    {
        std::cout << "Rendering " << (i + 1) << "/" << number_of_images << std::endl;

        update_world();
        results.push_back(ray_trace_datapoint(ray_traced_image));
        reset_image(ray_traced_image);
    }

    return results;
}

path_tracer::render_datapoint cuda_renderer::ray_trace_datapoint() const
{
    const auto timer = shared::scoped_timer("ray_trace_datapoint");
    auto ray_traced_image = render(w, h);
    auto result = ray_trace_datapoint(ray_traced_image);

    return result;
}

path_tracer::render_datapoint cuda_renderer::ray_trace_datapoint(render& ray_traced_image) const
{
    auto result = path_tracer::render_datapoint(w, h);
    auto sample_population = _sampler->get_samples_in_pow(4, 1000, 8);
    auto sample_sum = 0;
    for (auto i = 0; i < 4; i++)
    {
        auto samples = sample_population[i];
        const auto timer_intern = shared::scoped_timer(" _samples: " + std::to_string(samples));

        sample_sum += samples;

        ray_trace(samples, sample_sum, ray_traced_image, false);

        result.set_result(ray_traced_image, i);
    }
    return result;
}


void cuda_renderer::update_world()
{
    checkCudaErrors(cudaDeviceSynchronize());
    cuda_methods::free_world<<<1, 1>>>(d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    cuda_methods::create_random_world<<<1, 1>>>(d_list, d_world, d_camera, 13, 2, 2, d_rand_state);
    checkCudaErrors(cudaGetLastError());
}

void cuda_renderer::reset_image(render& ray_traced_image) const
{
    cuda_methods::reset_image<<<blocks, threads>>>(ray_traced_image.get_color_matrix(),
                                                   ray_traced_image.get_sample_matrix(),
                                                   w,
                                                   h);
    checkCudaErrors(cudaDeviceSynchronize());
}

} // namespace path_tracer
} // namespace ppt
