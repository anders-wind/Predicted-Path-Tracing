#pragma once
#include "opengl_helpers.hpp"

namespace ppt
{
namespace app
{
class vertex_buffer
{

    private:
    unsigned int m_renderer_id;

    public:
    /**
     * Size is size in bytes (number_of_items*sizeof(typeof data))
     */
    vertex_buffer(const void* data, unsigned int size)
    {
        GL_CALL(glGenBuffers(1, &m_renderer_id)); // create the buffer and write the id
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, m_renderer_id)); // Select the buffer as active.
        GL_CALL(glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW)); // static for update once, dynamic for updated many time
    }

    ~vertex_buffer()
    {
        GL_CALL(glDeleteBuffers(GL_ARRAY_BUFFER, &m_renderer_id));
    }

    void bind() const
    {
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, m_renderer_id)); // Select the buffer as active.
    }

    void unbind() const
    {
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0)); // Select the buffer as active.
    }
};
} // namespace app
} // namespace ppt