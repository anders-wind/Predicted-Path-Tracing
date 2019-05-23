#include <GL/glut.h>
#include <iostream>
#include <path_tracer/cuda_renderer.cuh>
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
    init_window(argc, argv);
    using namespace ppt::shared;
    using namespace ppt::path_tracer;
    int w = 1280;
    int h = 720;
    int s = 32;
    try
    {
        auto colors = cuda_renderer::cuda_ray_render(w, h, s);
        cuda_renderer::write_ppm_image(colors, w, h, "render");
    }
    catch (...)
    {
        std::cout << "failed" << std::endl;
    }
}