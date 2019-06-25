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


class gui_controller
{
    private:
    const std::shared_ptr<gui_state> state;
    const std::shared_ptr<path_tracer::cuda_renderer> path_tracer;
    const std::shared_ptr<path_tracer::render> render;

    public:
    gui_controller(const std::shared_ptr<gui_state>& state,
                   const std::shared_ptr<path_tracer::cuda_renderer>& path_tracer,
                   const std::shared_ptr<path_tracer::render>& render)
      : state(state), path_tracer(path_tracer), render(render)
    {
    }

    void draw() const
    {
        imgui::start_frame();
        basic_window();
        imgui::end_frame();
    }

    private:
    void basic_window() const
    {
        if (state->show_demo_window)
            ImGui::ShowDemoWindow(&state->show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            ImGui::Begin("PPT controller"); // Create a window called "Hello, world!" and append into it.

            add_fps();
            add_variance();

            ImGui::Checkbox("Toggle Demo Window", &state->show_demo_window); // Edit bools storing our window open/close state

            add_update_world_button();
            add_reset_button();

            ImGui::End();
        }
    }

    void add_reset_button() const
    {
        if (ImGui::Button("Reset")) // Buttons return true when clicked (most widgets return true when edited/activated)
        {
            auto lock = render->get_scoped_lock();

            state->sample_sum = 0;
            path_tracer->reset_image(*render);
        }
        ImGui::SameLine();
        ImGui::Text("sample_sum = %d", state->sample_sum);
    }

    void add_fps() const
    {
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
    }

    void add_variance() const
    {
        ImGui::Text("Variance sum %.3f", path_tracer->get_variance_mean());
    }

    void add_update_world_button() const
    {
        if (ImGui::Button("Update World")) // Buttons return true when clicked (most widgets return true when edited/activated)
        {
            auto lock = render->get_scoped_lock();

            state->sample_sum = 0;
            path_tracer->reset_image(*render);
            path_tracer->update_world();
        }
    }
};
} // namespace gui
} // namespace app
} // namespace ppt