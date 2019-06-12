
#pragma once
#include <GL/glew.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace ppt
{
namespace app
{
struct shader_program_source
{
    std::string vertex_source;
    std::string fragment_source;
};

/**
 * Parser for our own shader format (based on TheCherno opengl tutorial)
 * The file is seperated into a vertex shader starting with #shader vertex
 *  and a fragment shader starting with #shader fragment
 *
 */
static shader_program_source parse_shader(const std::string& file_path)
{
    std::ifstream stream(file_path);
    if (stream.fail())
    {
        std::cerr << "file did not exist" << std::endl;
        exit(1);
    }
    std::stringstream ss[2];

    enum class shader_type
    {
        none = -1,
        vertex = 0,
        fragment = 1,
    };

    auto mode = shader_type::none;

    std::string line;
    while (getline(stream, line))
    {
        if (line.find("#shader") != std::string::npos)
        {
            if (line.find("vertex") != std::string::npos)
            {
                mode = shader_type::vertex;
            }
            else if (line.find("fragment") != std::string::npos)
            {
                mode = shader_type::fragment;
            }
        }
        else
        {
            ss[(int)mode] << line << std::endl;
        }
    }
    return { ss[0].str(), ss[1].str() };
}


static unsigned int compile_shader(const std::string& source, unsigned int type)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    // error handeling
    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char)); // the cherno for getting stack mem of variable length
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "Failed to compile" << (type == GL_VERTEX_SHADER ? " vertex" : "fragment")
                  << " shader" << std::endl;
        std::cout << message << std::endl;
        exit(1);
    }
    return id;
}

static unsigned int create_shader(const std::string& vertex_shader, const std::string& fragment_shader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = compile_shader(vertex_shader, GL_VERTEX_SHADER);
    unsigned int fs = compile_shader(fragment_shader, GL_FRAGMENT_SHADER);

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

} // namespace app
} // namespace ppt