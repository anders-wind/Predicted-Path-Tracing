#pragma once
#include "index_buffer.hpp"
#include "shader.hpp"
#include "vertex_array.hpp"
#include <GL/glew.h>


namespace ppt
{
namespace app
{

class renderer
{
    private:
    public:
    void clear() const
    {
        GL_CALL(glClear(GL_COLOR_BUFFER_BIT));
    }

    void draw(const vertex_array& va, const index_buffer& ib, const shader& sh) const
    {
        // Bind
        sh.bind();
        va.bind();
        ib.bind();
        GL_CALL(glDrawElements(GL_TRIANGLES, ib.get_count(), GL_UNSIGNED_INT, nullptr));
    }
};

} // namespace app
} // namespace ppt