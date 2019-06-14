#pragma once
#include "app/helpers/imgui_helpers.hpp"
#include "gui_state.hpp"
#include <memory>
#include <path_tracer/cuda_renderer.cuh>
#include <path_tracer/render.cuh>

namespace ppt
{
namespace app
{
namespace gui
{
struct gui_controller
{
    private:
    std::shared_ptr<gui_state> state;
    std::shared_ptr<path_tracer::cuda_renderer> renderer;
    std::shared_ptr<path_tracer::render> render;

    public:
    gui_controller(std::shared_ptr<gui_state> state,
                   std::shared_ptr<path_tracer::cuda_renderer> renderer,
                   std::shared_ptr<path_tracer::render> render)
      : state(state), renderer(renderer), render(render)
    {
    }

    void draw()
    {
        imgui::start_frame();
        basic_window();
        imgui::end_frame();
    }

    private:
    void add_reset_button()
    {
        if (ImGui::Button("Reset")) // Buttons return true when clicked (most widgets return true when edited/activated)
        {
            state->sample_sum = 0;
            renderer->reset_image(*render);
        }
        ImGui::SameLine();
        ImGui::Text("sample_sum = %d", state->sample_sum);
    }

    void add_fps()
    {
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
    }

    void add_update_world_button()
    {
        if (ImGui::Button("Update World")) // Buttons return true when clicked (most widgets return true when edited/activated)
        {
            state->sample_sum = 0;
            renderer->reset_image(*render);
            renderer->update_world();
        }
    }

    void basic_window()
    {
        if (state->show_demo_window)
            ImGui::ShowDemoWindow(&state->show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            ImGui::Begin("PPT controller"); // Create a window called "Hello, world!" and append into it.

            add_fps();

            ImGui::Checkbox("Toggle Demo Window", &state->show_demo_window); // Edit bools storing our window open/close state

            add_update_world_button();
            add_reset_button();

            ImGui::End();
        }
    }
};
} // namespace gui
} // namespace app
} // namespace ppt