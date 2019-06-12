
#pragma once

#include <GLFW/glfw3.h>

namespace ppt
{
namespace app
{

GLFWwindow* init_window()
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        exit(-1);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "PPT", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(-1);
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);
    return window;
}

} // namespace app
} // namespace ppt