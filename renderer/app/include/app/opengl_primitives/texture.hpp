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
    unsigned char* m_local_buffer;
    int m_width;
    int m_height;
    int m_bpp; // bytes per pixel;

    public:
    texture(const std::string& path)
      : m_renderer_id(0), m_local_buffer(nullptr), m_width(0), m_height(0), m_bpp(0)
    {
        stbi_set_flip_vertically_on_load(true);
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

    texture(const std::vector<std::vector<std::array<unsigned char, 4>>>& image)
      : m_renderer_id(0), m_local_buffer(nullptr), m_width(0), m_height(0), m_bpp(0)
    {
        GL_CALL(glGenTextures(1, &m_renderer_id));

        m_height = image.size();
        m_width = image[0].size();
        m_bpp = 4;
        m_local_buffer = new unsigned char[m_height * m_width * m_bpp];

        update_local_buffer(image);

        GL_CALL(glBindTexture(GL_TEXTURE_2D, m_renderer_id));

        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

        GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_local_buffer));
        unbind();
    }

    void init_local_buffer()
    {
    }

    void update_local_buffer(const std::vector<std::vector<std::array<unsigned char, 4>>>& image)
    {
        for (auto i = 0; i < m_height; i++)
        {
            auto row = m_height - i - 1;
            for (auto j = 0; j < m_width; j++)
            {
                m_local_buffer[row * m_width * m_bpp + j * m_bpp + 0] = image[i][j][0];
                m_local_buffer[row * m_width * m_bpp + j * m_bpp + 1] = image[i][j][1];
                m_local_buffer[row * m_width * m_bpp + j * m_bpp + 2] = image[i][j][2];
                m_local_buffer[row * m_width * m_bpp + j * m_bpp + 3] = image[i][j][3];
            }
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