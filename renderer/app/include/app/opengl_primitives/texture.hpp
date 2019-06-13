#pragma once
#include "app/helpers/opengl_helpers.hpp"
#include <string>

namespace ppt
{
namespace app
{

class texture
{
    private:
    unsigned int m_renderer_id;
    std::string m_file_path;
    unsigned char* m_local_buffer;
    int m_width;
    int m_height;
    int m_bpp; // bits per pixel;

    texture(const std::string& path)
      : m_renderer_id(0), m_file_path(path), m_local_buffer(nullptr), m_width(0), m_height(0), m_bpp(0)
    {
        GL_CALL(glGenTextures(1, &m_renderer_id));

        GL_CALL(glBindTexture(GL_TEXTURE_2D, m_renderer_id));
    }

    void bind(unsigned int slot = 0) const
    {
        GL_CALL(glBindTexture(slot, m_renderer_id));
    }

    void unbind() const
    {
        GL_CALL(glBindTexture(0, 0));
    }

    int get_width() const
    {
        return m_width;
    }
    int get_height() const
    {
        return m_height;
    }
    int get_bpp() const
    {
        return m_bpp;
    }
};

} // namespace app
} // namespace ppt