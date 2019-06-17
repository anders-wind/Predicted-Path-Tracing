#include "app/gui/gui_controller.hpp"
#include "app/gui/gui_state.hpp"
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
#include <future>
#include <iostream>
#include <memory>
#include <path_tracer/cuda_renderer.cuh>
#include <shared/sample_service.cuh>
#include <shared/scoped_thread.cuh>
#include <sstream>
#include <string>

namespace ppt
{
namespace app
{


void main_loop(GLFWwindow* window, std::shared_ptr<path_tracer::cuda_renderer> path_tracer)
{
    using namespace gui;
    auto re = renderer();
    auto state = std::make_shared<gui_state>();
    constexpr auto inc = 1;
    state->sample_sum += inc;
    auto rendering = path_tracer->ray_trace(inc, state->sample_sum);
    const auto gui = gui_controller(state, path_tracer, rendering);

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

    // vertex array

    auto va = vertex_array();
    auto vb = vertex_buffer(positions, element_size * number_of_elements * sizeof(float));
    auto ib = index_buffer(indices, number_of_indices);
    auto layout = vertex_buffer_layout();
    auto basic_shader = shader("app/res/shaders/basic.shader");
    auto tex = texture(rendering->get_byte_representation(), path_tracer->w, path_tracer->h);

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

    auto render_thread_handle = true;
    auto t = shared::scoped_thread(std::thread([state, path_tracer, rendering, &render_thread_handle] {
        while (render_thread_handle)
        {
            auto lock = rendering->get_scoped_lock();
            state->sample_sum += inc;
            path_tracer->ray_trace(inc, state->sample_sum, *rendering);
            std::this_thread::yield();
        }
    }));

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        re.clear();
        basic_shader.bind();
        tex.bind(tex_slot);

        // Draw
        re.draw(va, ib, basic_shader);
        gui.draw();
        tex.update_local_buffer(rendering->get_byte_representation());


        /* Swap front and back buffers */
        GL_CALL(glfwSwapBuffers(window));

        /* Poll for and process events */
        GL_CALL(glfwPollEvents());
    }

    render_thread_handle = false;
}

void setup(GLFWwindow* const window)
{
    ppt::app::gl_init();
    ppt::app::glew_init();
    ppt::app::imgui::init(window);
}


void shutdown()
{
    ppt::app::imgui::shutdown();
    glfwTerminate();
}

} // namespace app
} // namespace ppt

int main(void)
{
    int w = 1280;
    int h = 720;

    const auto sampler = std::make_shared<ppt::shared::sample_service>();
    auto path_tracer = std::make_shared<ppt::path_tracer::cuda_renderer>(w, h, sampler);

    GLFWwindow* window = ppt::app::init_window(w, h + 20);

    ppt::app::setup(window);

    ppt::app::main_loop(window, path_tracer);

    ppt::app::shutdown();

    return 0;
}