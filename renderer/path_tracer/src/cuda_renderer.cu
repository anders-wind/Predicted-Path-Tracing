#include "path_tracer/cuda_renderer.cuh"
#include <cuda.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
namespace ppt
{
namespace path_tracer
{
namespace cuda_methods
{


__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state, float min_depth, float max_depth)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
    hit_record rec;

    for (int i = 0; i < 10; i++)
    {
        if (!(*world)->hit(cur_ray, min_depth, max_depth, rec))
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
render_image(vec3* image_matrix, int max_x, int max_y, int samples, camera* cam, hitable** world, curandState* rand_state)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    curandState local_rand_state = rand_state[pixel_index];

    vec3 pix = vec3(image_matrix[pixel_index].e);
    const camera local_camera(*cam);
    for (int s = 0; s < samples; s++)
    {
        float u = float(col + curand_normal(&local_rand_state)) / float(max_x);
        float v = float(max_y - row + curand_normal(&local_rand_state)) / float(max_y);
        ray r = local_camera.get_ray(u, v, &local_rand_state);
        pix += color(r, world, &local_rand_state, local_camera._min_depth, local_camera._max_depth);
    }

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
                             int max_x,
                             int max_y,
                             int samples,
                             curandState* rand_state)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    curandState local_rand_state = rand_state[pixel_index];

    auto in_pixel = image_matrix[pixel_index];
    auto norm_rgb = (vec3(in_pixel.e) / float(samples)).v_sqrt();

    auto sample_precision = __logf(samples) / 16.0f; // 2^16=65536 is our (arbitrarily) choosen max number of samples
    auto hit = depth_map(world, camera, col, row, max_x, max_y, &local_rand_state);
    auto depth = sqrtf(fabs(hit.t)) / sqrtf(camera->_max_depth);

    out_image_matrix[pixel_index] =
        vec8(norm_rgb, sample_precision, depth, hit.normal[0], hit.normal[1], hit.normal[2]);
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

__global__ void reset_image(vec3* color_matrix, int max_x, int max_y)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if ((col >= max_x) || (row >= max_y))
        return;

    int pixel_index = RM(row, col, max_x);
    color_matrix[pixel_index] = vec3(0, 0, 0);
}

__global__ void
create_small_world(hitable** d_list, hitable** d_world, camera* d_camera, int hitables_size, curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_list[0] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.6, 0.8, 0.6)));
        // d_list[0] = new plane(vec3(0, -0.5, 0), vec3(0, 1, 0), new lambertian(vec3(0.5, 0.4, 0.3)));
        d_list[1] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
        d_list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new sphere(vec3(-1, 0.0, -1), 0.5, new dielectric(1.5f));
        d_list[4] = new sphere(vec3(-1, 0.0, -1), -0.45, new dielectric(1.5f));

        *d_world = new bvh_node(d_list, hitables_size);
        // *d_world = new hitable_list(d_list, hitables_size);
        const auto look_from = vec3(-13, 1, 8);
        const auto look_at = vec3(0, 0, -1);
        const auto focus_dist = 16;
        const auto fov = 8;
        const auto aperture = 0.5;
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
        d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));

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
                        d_list[i++] = new sphere(
                            center,
                            0.2,
                            new lambertian(
                                vec3(curand_uniform(&local_rand_state) * curand_uniform(&local_rand_state),
                                     curand_uniform(&local_rand_state) * curand_uniform(&local_rand_state),
                                     curand_uniform(&local_rand_state) * curand_uniform(&local_rand_state))));
                    }
                    else if (choose_mat < 0.95)
                    { // metal
                        d_list[i++] =
                            new sphere(center,
                                       0.2,
                                       new metal(vec3(0.5 * (1 + curand_uniform(&local_rand_state)),
                                                      0.5 * (1 + curand_uniform(&local_rand_state)),
                                                      0.5 * (1 + curand_uniform(&local_rand_state))),
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
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

        *d_world = new bvh_node(d_list, hitables_size);
        // *d_world = new hitable_list(d_list, hitables_size);
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
                d_list[i] = new plane(vec3(0, -3, 0), vec3(0, 1, 0), new lambertian(RANDVEC3(local_rand)));
                continue;
            }

            material* mat;
            if (number_of_reflection < reflection)
            {
                mat = new metal(RANDVEC3(local_rand), curand_uniform(local_rand));
                number_of_reflection++;
            }
            else if (number_of_refraction < refraction)
            {
                mat = new dielectric(0.001f + fabs(curand_uniform(local_rand) - 0.001f));
                number_of_refraction++;
            }
            else
            {
                mat = new lambertian(RANDVEC3(local_rand));
            }
            d_list[i] = new sphere(vec3((curand_uniform(local_rand) - 0.5) * 14,
                                        (curand_uniform(local_rand) - 0.5) * 8,
                                        curand_uniform(local_rand) * -3 - 2),
                                   curand_uniform(local_rand) * curand_uniform(local_rand) * 1.5 + 0.3,
                                   mat);
        }

        *d_world = new bvh_node(d_list, hitables_size);
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

    int hitables_size = 5; // 485 for large
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
    cuda_methods::create_small_world<<<1, 1>>>(d_list, d_world, d_camera, hitables_size, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

cuda_renderer::~cuda_renderer()
{
    checkCudaErrors(cudaDeviceSynchronize());
    cuda_methods::free_world<<<1, 1>>>(d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
}

std::shared_ptr<render> cuda_renderer::ray_trace(int samples, int sample_sum) const
{
    const auto timer = shared::scoped_timer("ray_trace");

    auto ray_traced_image = std::make_shared<render>(w, h);
    ray_trace(samples, sample_sum, *ray_traced_image);
    return ray_traced_image;
}

void cuda_renderer::ray_trace(int samples, int sample_sum, render& ray_traced_image) const
{
    {
        // const auto timer = shared::scoped_timer("ray_tracing");

        cuda_methods::render_image<<<blocks, threads>>>(
            ray_traced_image.get_color_matrix(), w, h, samples, d_camera, d_world, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    {
        // const auto timer = shared::scoped_timer("post process");

        cuda_methods::post_process<<<blocks, threads>>>(ray_traced_image.get_color_matrix(),
                                                        ray_traced_image.get_image_matrix(),
                                                        d_world,
                                                        d_camera,
                                                        w,
                                                        h,
                                                        sample_sum,
                                                        d_rand_state);
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

        ray_trace(samples, sample_sum, ray_traced_image);

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
    cuda_methods::reset_image<<<blocks, threads>>>(ray_traced_image.get_color_matrix(), w, h);
    checkCudaErrors(cudaDeviceSynchronize());
}

} // namespace path_tracer
} // namespace ppt
