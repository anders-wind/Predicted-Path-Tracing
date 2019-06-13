#pragma once
#include <GL/glew.h>
#include <iostream>

namespace ppt
{
namespace app
{
void glew_init()
{
    // Init Glew
    if (glewInit() != GLEW_OK || !glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        exit(1);
    }
}
} // namespace app
} // namespace ppt