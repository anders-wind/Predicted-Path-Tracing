#pragma once
#include "app/helpers/imgui_helpers.hpp"
#include "gui_state.hpp"
#include <memory>

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

    public:
    gui_controller(std::shared_ptr<gui_state> state) : state(state)
    {
    }

    void draw()
    {
        imgui::start_frame();
        basic_window();
        imgui::end_frame();
    }

    private:
    void basic_window()
    {
        if (state->show_demo_window)
            ImGui::ShowDemoWindow(&state->show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            static float f = 0.0f;
            ImGui::Begin("PPT controller"); // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text."); // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &state->show_demo_window); // Edit bools storing our window open/close state

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&state->clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
            {
                // counter++;
            }
            ImGui::SameLine();
            ImGui::Text("sample_sum = %d", state->sample_sum);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::End();
        }
    }
};
} // namespace gui
} // namespace app
} // namespace ppt