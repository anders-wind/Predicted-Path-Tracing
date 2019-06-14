
#pragma once

#include <GLFW/glfw3.h>

namespace ppt
{
namespace app
{

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

GLFWwindow* init_window(unsigned int w = 640, unsigned int h = 360, const std::string& name = "PPT")
{
    GLFWwindow* window;
    glfwSetErrorCallback(glfw_error_callback);


    /* Initialize the library */
    if (!glfwInit())
        exit(-1);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(w, h, name.c_str(), NULL, NULL);
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