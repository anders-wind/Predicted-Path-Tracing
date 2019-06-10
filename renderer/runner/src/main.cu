#include <GL/glut.h>
#include <dataset_creator/dataset_repository.hpp>
#include <path_tracer/cuda_renderer.cuh>
#include <shared/sample_service.cuh>
#include <string>

void display_me(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POLYGON);
    glVertex3f(0.5, 0.0, 0.5);
    glVertex3f(0.5, 0.0, 0.0);
    glVertex3f(0.0, 0.5, 0.0);
    glVertex3f(0.0, 0.0, 0.5);
    glEnd();
    glFlush();
}

void init_window(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(400, 300);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Hello world!");
    glutDisplayFunc(display_me);
    glutMainLoop();
}

int main(int argc, char** argv)
{
    // init_window(argc, argv);
    // settings
    int w = 640;
    int h = 360;
    size_t number_of_images = 2;
    std::string filename = "trial";

    // services
    const auto sampler = std::make_shared<ppt::shared::sample_service>();
    auto repository = ppt::dataset_creator::dataset_repository(
        std::string(getenv("HOME")) + "/Documents/datasets/ppt/640x360_run07");

    auto renderer = ppt::path_tracer::cuda_renderer(w, h, sampler);

    // ray trace
    auto render_datapoints = renderer.ray_trace_datapoints(number_of_images);
    repository.save_datapoints(render_datapoints, filename);
    repository.save_ppms(render_datapoints, filename);
}