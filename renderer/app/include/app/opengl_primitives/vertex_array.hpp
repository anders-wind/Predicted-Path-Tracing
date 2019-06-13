#pragma once
#include "vertex_buffer.hpp"
#include "vertex_buffer_layout.hpp"
#include <app/helpers/opengl_helpers.hpp>


namespace ppt
{
namespace app
{
class vertex_array
{
    private:
    unsigned int m_renderer_id;

    public:
    vertex_array()
    {
        GL_CALL(glGenVertexArrays(1, &m_renderer_id));
    }

    ~vertex_array()
    {
        GL_CALL(glDeleteVertexArrays(1, &m_renderer_id));
    }

    void bind() const
    {
        GL_CALL(glBindVertexArray(m_renderer_id));
    }

    void unbind() const
    {
        GL_CALL(glBindVertexArray(m_renderer_id));
    }

    void add_buffer(const vertex_buffer& vb, const vertex_buffer_layout& layout)
    {
        bind();
        vb.bind();
        const auto& elements = layout.get_elements();
        intptr_t offset = 0;
        for (auto i = 0; i < elements.size(); i++)
        {
            const auto& element = elements[i];
            GL_CALL(glEnableVertexAttribArray(i));
            GL_CALL(glVertexAttribPointer(
                i, element.count, element.type, element.normalized, layout.get_stride(), (const void*)offset));
            offset += element.count * vertex_buffer_element::get_size_of_type(element.type);
        }
    }
};
} // namespace app
} // namespace ppt