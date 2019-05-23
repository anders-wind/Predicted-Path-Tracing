#include <path_tracer/cuda_renderer.cuh>
#include <shared/scoped_timer.cuh>

namespace ppt
{
namespace path_tracer
{
namespace cuda_renderer
{

void write_ppm_image(std::vector<rgb> colors, int w, int h, std::string filename)
{
    const auto timer = ppt::shared::ScopedTimer("write_ppm_image");

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
    const auto timer = ppt::shared::ScopedTimer("cuda_ray_render");

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

    size_t render_image_bytes = w * h * sizeof(vec3);
    vec3* d_image_matrix;
    vec3* h_image_matrix = new vec3[w * h];
    checkCudaErrors(cudaMallocManaged((void**)&d_image_matrix, render_image_bytes));

    int tx = 2, ty = 2;
    dim3 blocks(h / ty + 1, w / tx + 1);
    dim3 threads(tx, ty);

    render_init<<<blocks, threads>>>(w, h, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(d_image_matrix, w, h, samples, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(h_image_matrix, d_image_matrix, render_image_bytes, cudaMemcpyDeviceToHost);

    auto colors = std::vector<rgb>(w * h);

    std::cout << "P3\n" << h << " " << w << "\n255\n";

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            const size_t pixel_index = RM(i, j, w);
            colors[pixel_index] = h_image_matrix[pixel_index];
        }
    }


    // free up the memory
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_image_matrix));

    return colors;
}

} // namespace cuda_renderer
} // namespace path_tracer
} // namespace ppt