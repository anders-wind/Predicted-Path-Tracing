
#pragma once
#include <GL/glew.h>
#include <iostream>

namespace ppt
{
namespace app
{

static void gl_clear_errors()
{
    while (glGetError() != GL_NO_ERROR)
    {
    }
}

static bool gl_check_errors(const char* function, const char* file, int line)
{
    auto is_ok = true;
    while (GLenum error = glGetError())
    {
        std::cerr << "[OpenGl error] (" << error << ") " << function << " " << file << ":" << line << std::endl;
        is_ok = false;
    }
    return is_ok;
}

void gl_error_callback()
{
}


#define ASSERT(x) \
    if (!(x))     \
        raise(SIGTRAP);

#define GL_CALL(x)     \
    gl_clear_errors(); \
    x;                 \
    ASSERT(gl_check_errors(#x, __FILE__, __LINE__));

} // namespace app
} // namespace ppt