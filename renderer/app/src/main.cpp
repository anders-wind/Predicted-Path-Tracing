#include "app/glew_helpers.hpp"
#include "app/glfw_helpers.hpp"
#include "app/opengl_helpers.hpp"
#include "app/shader_helpers.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <string>

namespace ppt
{
namespace app
{


void main_loop(GLFWwindow* window)
{
    // a triangle
    constexpr int element_size = 2;
    constexpr int number_of_elements = 4;
    constexpr int number_of_indices = 6;
    float positions[element_size * number_of_elements] = {
        -1.0f, -1.0f, // 0
        1.0f,  -1.0f, // 1
        1.0f,  1.0f, // 2
        -1.0f, 1.0f // 3
    };
    unsigned int indices[number_of_indices] = {
        0,
        1,
        2,
        // second triangle
        0,
        2,
        3,
    };

    // vertex buffer
    unsigned int buffer; // this is the handle for the buffer
    GL_CALL(glGenBuffers(1, &buffer)); // create the buffer and write the id
    GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, buffer)); // Select the buffer as active.
    GL_CALL(glBufferData(GL_ARRAY_BUFFER, element_size * (number_of_elements) * sizeof(float), positions, GL_STATIC_DRAW)); // static for update once, dynamic for updated many time
    GL_CALL(glEnableVertexAttribArray(0));
    GL_CALL(glVertexAttribPointer(0, element_size, GL_FLOAT, GL_FALSE, element_size * sizeof(float), (const void*)0));
    // glBindBuffer(GL_ARRAY_BUFFER, 0); // bind no buffer.

    // index buffer
    unsigned int ibo; // this is the handle for the buffer
    GL_CALL(glGenBuffers(1, &ibo)); // create the buffer and write the id
    GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)); // Select the buffer as active.
    GL_CALL(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                         number_of_indices * sizeof(unsigned int),
                         indices,
                         GL_STATIC_DRAW)); // static for update once, dynamic for updated many time
    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // bind no buffer.


    auto shader_programs = parse_shader("app/res/shaders/basic.shader");
    auto shader = create_shader(shader_programs.vertex_source, shader_programs.fragment_source);


    GL_CALL(glUseProgram(shader));

    GL_CALL(int u_color_location = glGetUniformLocation(shader, "u_color"));
    ASSERT(u_color_location != -1);
    /* Loop until the user closes the window */
    float r = 0.0f;
    float increment = 0.05f;

    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

        GL_CALL(glUniform4f(u_color_location, r, 0.3f, 0.8f, 1.0f));
        GL_CALL(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr));

        if (r > 1.0f || r < 0.0f)
        {
            increment *= -1;
        }

        r += increment;
        /* Swap front and back buffers */
        GL_CALL(glfwSwapBuffers(window));

        /* Poll for and process events */
        GL_CALL(glfwPollEvents());
    }
    GL_CALL(glDeleteProgram(shader));
}


void shutdown()
{
    glfwTerminate();
}

} // namespace app
} // namespace ppt

int main(void)
{
    GLFWwindow* window = ppt::app::init_window();

    ppt::app::glew_init();

    ppt::app::main_loop(window);

    ppt::app::shutdown();

    return 0;
}