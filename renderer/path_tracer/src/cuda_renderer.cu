#include <fstream>
#include <iostream>
#include <path_tracer/cuda_renderer.cuh>
#include <sstream>

using namespace ppt::shared;

namespace ppt
{
namespace path_tracer
{
namespace cuda_renderer
{

void write_ppm_image(std::vector<rgb> colors, int w, int h, std::string filename)
{
    const auto timer = scoped_timer("write_ppm_image");

    std::ofstream myfile;
    std::stringstream ss;
    ss << "P3\n" << w << " " << h << "\n255\n";

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            ss << colors[RM(i, j, w)] << std::endl;
        }
    }

    myfile.open(filename + ".ppm");
    myfile << ss.str();
    myfile.close();
}

} // namespace cuda_renderer
} // namespace path_tracer
} // namespace ppt