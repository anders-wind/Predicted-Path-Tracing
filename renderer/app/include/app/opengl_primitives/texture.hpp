#pragma once
#include "app/helpers/opengl_helpers.hpp"
#include <stb/stb_image.h>
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
    int m_bpp; // bytes per pixel;

    public:
    texture(const std::string& path)
      : m_renderer_id(0), m_file_path(path), m_local_buffer(nullptr), m_width(0), m_height(0), m_bpp(0)
    {
        stbi_set_flip_vertically_on_load(1);
        m_local_buffer = stbi_load(path.c_str(), &m_width, &m_height, &m_bpp, 4);

        GL_CALL(glGenTextures(1, &m_renderer_id));

        GL_CALL(glBindTexture(GL_TEXTURE_2D, m_renderer_id));

        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

        GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_local_buffer));
        unbind();

        if (m_local_buffer) // free it again
        {
            stbi_image_free(m_local_buffer);
        }
    }

    void bind(unsigned int slot = 0) const
    {
        GL_CALL(glActiveTexture(GL_TEXTURE0 + slot));
        GL_CALL(glBindTexture(GL_TEXTURE_2D, m_renderer_id));
    }

    void unbind() const
    {
        GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));
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