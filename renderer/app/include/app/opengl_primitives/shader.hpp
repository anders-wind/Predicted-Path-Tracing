#pragma once
#include <app/helpers/opengl_helpers.hpp>
#include <app/helpers/shader_helpers.hpp>
#include <string>
#include <unordered_map>


namespace ppt
{
namespace app
{
class shader
{
    private:
    unsigned int m_renderer_id;
    const std::string file_path;
    std::unordered_map<std::string, int> m_uniform_location_cache;

    public:
    shader(const std::string& file_path) : m_renderer_id(0), file_path(file_path)
    {
        auto shader_programs = parse_shader(file_path);
        m_renderer_id = create_shader(shader_programs.vertex_source, shader_programs.fragment_source);
    }

    ~shader()
    {
        GL_CALL(glDeleteProgram(m_renderer_id));
    }

    void bind() const
    {
        GL_CALL(glUseProgram(m_renderer_id));
    }

    void unbind() const
    {
        GL_CALL(glUseProgram(0));
    }

    // set uniforms
    void set_uniform1i(const std::string& name, int v0)
    {
        const auto location = get_uniform_location(name);
        GL_CALL(glUniform1i(location, v0));
    }

    void set_uniform1f(const std::string& name, float v0)
    {
        const auto location = get_uniform_location(name);
        GL_CALL(glUniform1f(location, v0));
    }

    void set_uniform4f(const std::string& name, float v0, float v1, float v2, float v3)
    {
        const auto location = get_uniform_location(name);
        GL_CALL(glUniform4f(location, v0, v1, v2, v3));
    }

    private:
    unsigned int get_uniform_location(const std::string& name)
    {
        if (m_uniform_location_cache.find(name) != m_uniform_location_cache.end())
        {
            return m_uniform_location_cache[name];
        }

        GL_CALL(int location = glGetUniformLocation(m_renderer_id, name.c_str()));
        if (location == -1)
        {
            std::cerr << "Warning: uniform (" << name << ") does not exist" << std::endl;
        }
        m_uniform_location_cache[name] = location;
        return location;
    }
};
} // namespace app
} // namespace ppt