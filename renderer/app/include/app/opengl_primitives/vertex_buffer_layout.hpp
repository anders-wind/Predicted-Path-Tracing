#pragma once
#include <app/helpers/opengl_helpers.hpp>
#include <vector>

namespace ppt
{
namespace app
{

struct vertex_buffer_element
{
    unsigned int type;
    unsigned int count;
    unsigned char normalized;

    static unsigned int get_size_of_type(unsigned int type)
    {
        switch (type)
        {
        case GL_FLOAT:
            return sizeof(float);
        case GL_INT:
            return sizeof(int);
        case GL_UNSIGNED_INT:
            return sizeof(unsigned int);
        case GL_UNSIGNED_BYTE:
            return sizeof(unsigned char);
        default:
            ASSERT(false);
        }
        return 0;
    }
};

class vertex_buffer_layout
{
    private:
    std::vector<vertex_buffer_element> m_elements;
    unsigned int m_stride;

    public:
    vertex_buffer_layout() : m_stride(0)
    {
    }

    template <typename T> void push(unsigned int count);


    inline const std::vector<vertex_buffer_element>& get_elements() const
    {
        return m_elements;
    }

    inline unsigned int get_stride() const
    {
        return m_stride;
    }
};

template <> void vertex_buffer_layout::push<float>(unsigned int count)
{
    m_elements.push_back({ GL_FLOAT, count, GL_FALSE });
    m_stride += vertex_buffer_element::get_size_of_type(GL_FLOAT) * count;
}

template <> void vertex_buffer_layout::push<int>(unsigned int count)
{
    m_elements.push_back({ GL_INT, count, GL_FALSE });
    m_stride += vertex_buffer_element::get_size_of_type(GL_INT) * count;
}

template <> void vertex_buffer_layout::push<unsigned int>(unsigned int count)
{
    m_elements.push_back({ GL_UNSIGNED_INT, count, GL_FALSE });
    m_stride += vertex_buffer_element::get_size_of_type(GL_UNSIGNED_INT) * count;
}

template <> void vertex_buffer_layout::push<unsigned char>(unsigned int count)
{
    m_elements.push_back({ GL_UNSIGNED_BYTE, count, GL_TRUE });
    m_stride += vertex_buffer_element::get_size_of_type(GL_UNSIGNED_BYTE) * count;
}

} // namespace app
} // namespace ppt