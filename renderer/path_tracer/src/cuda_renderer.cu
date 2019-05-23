#include <path_tracer/cuda_renderer.cuh>

namespace ppt
{
namespace path_tracer
{
namespace cuda_renderer
{

void write_ppm_image(std::vector<rgb> colors, int w, int h, std::string filename)
{
    std::ofstream myfile;
    myfile.open(filename + ".ppm");
    myfile << "P3\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            auto color = colors[RM(i, j, w)];
            myfile << color.r() * 255.99 << " " << color.g() * 255.99 << " " << color.b() * 255.99 << std::endl;
        }
    }
    myfile.close();
}

std::vector<rgb> cuda_ray_render(int w, int h, int samples)
{

    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, w * h * sizeof(curandState)));
    hitable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 5 * sizeof(hitable*)));
    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    size_t fb_size = w * h * sizeof(vec3);
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, stop;
    start = clock();

    int tx = 8, ty = 8;
    dim3 blocks(h / ty + 1, w / tx + 1);
    dim3 threads(tx, ty);

    render_init<<<blocks, threads>>>(w, h, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, w, h, samples, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    auto colors = std::vector<rgb>(w * h);
    // auto world = std::make_shared<hitable_list>();
    // world->add_hitable(std::make_shared<sphere>(vec3(0, 0, -1), 0.5f,
    // std::make_shared<lambertian>(vec3(0.8f, 0.3f, 0.3f)))); world->add_hitable(std::make_shared<sphere>(vec3(0,
    // -100.5, -1), 100.0f, std::make_shared<lambertian>(vec3(0.8f, 0.8f, 0.0f)))); world->add_hitable(std::make_shared<sphere>(vec3(1,
    // 0, -1), 0.5f, std::make_shared<metal>(vec3(0.8f, 0.6f, 0.2f), 0.3f))); world->add_hitable(std::make_shared<sphere>(vec3(-1,
    // 0, -1), 0.5f, std::make_shared<dielectric>(1.5f))); world->add_hitable(std::make_shared<sphere>(vec3(-1,
    // 0, -1), -0.45f, std::make_shared<dielectric>(1.5f)));

    std::cout << "P3\n" << h << " " << w << "\n255\n";
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            const size_t pixel_index = RM(i, j, w);
            colors[pixel_index] = fb[pixel_index];
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    return colors;
}

} // namespace cuda_renderer
} // namespace path_tracer
} // namespace ppt