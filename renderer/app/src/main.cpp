#include "app/helpers/glew_helpers.hpp"
#include "app/helpers/glfw_helpers.hpp"
#include "app/helpers/opengl_helpers.hpp"
#include "app/helpers/shader_helpers.hpp"
#include "app/opengl_primitives/index_buffer.hpp"
#include "app/opengl_primitives/renderer.hpp"
#include "app/opengl_primitives/shader.hpp"
#include "app/opengl_primitives/texture.hpp"
#include "app/opengl_primitives/vertex_array.hpp"
#include "app/opengl_primitives/vertex_buffer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace ppt
{
namespace app
{


void main_loop(GLFWwindow* window)
{
    // a triangle
    constexpr int pos_size = 2;
    constexpr int tex_size = 2;
    constexpr int element_size = pos_size + tex_size;
    constexpr int number_of_elements = 4;
    constexpr int number_of_indices = 6;
    float positions[element_size * number_of_elements] = {
        -1.0f, -1.0f, 0.0f, 0.0f, // 0
        1.0f,  -1.0f, 1.0f, 0.0f, // 1
        1.0f,  1.0f,  1.0f, 1.0f, // 2
        -1.0f, 1.0f,  0.0f, 1.0f // 3
    };
    unsigned int indices[number_of_indices] = {
        0,
        1,
        2,
        // second triangle
        2,
        3,
        0,
    };

    GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    GL_CALL(glEnable(GL_BLEND));
    // vertex array

    auto va = vertex_array();
    auto vb = vertex_buffer(positions, element_size * number_of_elements * sizeof(float));
    auto ib = index_buffer(indices, number_of_indices);
    auto layout = vertex_buffer_layout();
    auto basic_shader = shader("app/res/shaders/basic.shader");
    auto tex = texture("app/res/textures/dummy_texture02.png");
    auto re = renderer();

    layout.push<float>(pos_size); // first pos.x, pos.y
    layout.push<float>(tex_size); // then tex.x, tex.y
    va.add_buffer(vb, layout);

    basic_shader.bind();
    const auto tex_slot = 0;
    tex.bind(tex_slot);
    basic_shader.set_uniform1i("u_texture", tex_slot);

    va.unbind();
    vb.unbind();
    ib.unbind();
    basic_shader.unbind();
    tex.unbind();


    /* Loop until the user closes the window */
    float red = 0.0f;
    float increment = 0.05f;
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        re.clear();
        // color
        // Draw
        basic_shader.bind();

        re.draw(va, ib, basic_shader);


        if (red > 1.0f || red < 0.0f)
        {
            increment *= -1;
        }
        red += increment;

        /* Swap front and back buffers */
        GL_CALL(glfwSwapBuffers(window));

        /* Poll for and process events */
        GL_CALL(glfwPollEvents());
    }
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