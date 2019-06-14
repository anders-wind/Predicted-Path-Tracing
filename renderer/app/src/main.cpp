#include "app/helpers/glew_helpers.hpp"
#include "app/helpers/glfw_helpers.hpp"
#include "app/helpers/imgui_helpers.hpp"
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

struct gui_state
{
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    bool show_demo_window = false;
};

void imgui_window(gui_state& state)
{
    if (state.show_demo_window)
        ImGui::ShowDemoWindow(&state.show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {
        static float f = 0.0f;
        static int counter = 0;
        ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text."); // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &state.show_demo_window); // Edit bools storing our window open/close state

        ImGui::SliderFloat("float", &f, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&state.clear_color); // Edit 3 floats representing a color

        if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::End();
    }
}

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

    GL_CALL(glEnable(GL_BLEND));
    GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    // vertex array

    auto va = vertex_array();
    auto vb = vertex_buffer(positions, element_size * number_of_elements * sizeof(float));
    auto ib = index_buffer(indices, number_of_indices);
    auto layout = vertex_buffer_layout();
    auto basic_shader = shader("app/res/shaders/basic.shader");
    auto tex = texture("app/res/textures/test01_target.png");
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

    auto state = gui_state();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {

        /* Render here */
        re.clear();
        basic_shader.bind();
        tex.bind(tex_slot);


        // Draw
        re.draw(va, ib, basic_shader);

        // GUI elements
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        imgui_window(state);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        /* Swap front and back buffers */
        GL_CALL(glfwSwapBuffers(window));

        /* Poll for and process events */
        GL_CALL(glfwPollEvents());
    }
}


void shutdown()
{
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
}

} // namespace app
} // namespace ppt

int main(void)
{
    GLFWwindow* window = ppt::app::init_window();

    ppt::app::glew_init();
    ppt::app::imgui_init(window);

    ppt::app::main_loop(window);

    ppt::app::shutdown();

    return 0;
}