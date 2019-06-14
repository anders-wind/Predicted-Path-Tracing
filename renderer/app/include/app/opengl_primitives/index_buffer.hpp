#pragma once
#include <app/helpers/opengl_helpers.hpp>

namespace ppt
{
namespace app
{
class index_buffer
{

    private:
    unsigned int m_renderer_id;
    unsigned int m_count;

    public:
    /**
     * Size is size in bytes (number_of_items*sizeof(typeof data))
     */
    index_buffer(const unsigned int* data, unsigned int count) : m_count(count)
    {
        ASSERT(sizeof(unsigned int) == sizeof(GLuint));
        auto size = count * sizeof(unsigned int);

        GL_CALL(glGenBuffers(1, &m_renderer_id)); // create the buffer and write the id
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_renderer_id)); // Select the buffer as active.
        GL_CALL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW)); // static for update once, dynamic for updated many time
    }

    ~index_buffer()
    {
        GL_CALL(glDeleteBuffers(1, &m_renderer_id));
    }

    void bind() const
    {
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_renderer_id)); // Select the buffer as active.
    }

    void unbind() const
    {
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)); // Select the buffer as active.
    }

    inline unsigned int get_count() const
    {
        return m_count;
    }
};
} // namespace app
} // namespace ppt