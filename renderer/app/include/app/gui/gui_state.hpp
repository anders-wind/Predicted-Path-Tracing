#pragma once
#include "app/helpers/imgui_helpers.hpp"

namespace ppt
{
namespace app
{
namespace gui
{
struct gui_state
{
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    bool show_demo_window = false;
    int sample_sum = 0;
};
} // namespace gui
} // namespace app
} // namespace ppt