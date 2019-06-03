#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

namespace ppt
{
namespace shared
{

struct vec3
{
    float e[3];
    __host__ __device__ vec3(){};
    __host__ __device__ vec3(float e0, float e1, float e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }

    __host__ __device__ explicit vec3(float o[5])
    {
        e[0] = o[0];
        e[1] = o[1];
        e[2] = o[2];
    }

    __host__ __device__ inline float x() const
    {
        return e[0];
    }
    __host__ __device__ inline void x(float v)
    {
        e[0] = v;
    }
    __host__ __device__ inline float y()
    {
        return e[1];
    }
    __host__ __device__ inline void y(float v)
    {
        e[1] = v;
    }
    __host__ __device__ inline float z()
    {
        return e[2];
    }
    __host__ __device__ inline void z(float v)
    {
        e[2] = v;
    }

    // unary operators
    __host__ __device__ inline const vec3& operator+() const
    {
        return *this;
    }
    __host__ __device__ inline vec3 operator-() const
    {
        return vec3(-e[0], -e[1], -e[2]);
    }
    __host__ __device__ inline float operator[](int i) const
    {
        return e[i];
    }
    __host__ __device__ inline float& operator[](int i)
    {
        return e[i];
    }

    // binary operators
    __host__ __device__ inline vec3& operator+=(const vec3& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator-=(const vec3& v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator*=(const vec3& v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator/=(const vec3& v)
    {
        e[0] /= v.e[0];
        e[1] /= v.e[1];
        e[2] /= v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ inline vec3& operator/=(const float t)
    {
        e[0] /= t;
        e[1] /= t;
        e[2] /= t;
        return *this;
    }

    __host__ __device__ inline vec3& operator-=(const float t)
    {
        e[0] -= t;
        e[1] -= t;
        e[2] -= t;
        return *this;
    }

    __host__ __device__ inline vec3& operator+=(const float t)
    {
        e[0] += t;
        e[1] += t;
        e[2] += t;
        return *this;
    }

    __host__ __device__ inline float length() const
    {
        return sqrt(squared_length());
    }

    __host__ __device__ inline vec3& v_square()
    {
        e[0] *= e[0];
        e[1] *= e[1];
        e[2] *= e[2];
        return *this;
    }

    __host__ __device__ inline vec3& v_sqrt()
    {
        e[0] = sqrt(e[0]);
        e[1] = sqrt(e[1]);
        e[2] = sqrt(e[2]);
        return *this;
    }

    __host__ __device__ inline float squared_length() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __host__ __device__ inline vec3& make_unit_vector()
    {
        float k = 1.0f / length();
        e[0] *= k;
        e[1] *= k;
        e[2] *= k;
        return *this;
    }
};

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2)
{
    return vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2)
{
    return vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2)
{
    return vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2)
{
    return vec3(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2)
{
    return vec3(v1[1] * v2[2] - v1[2] * v2[1], v1[0] * v2[2] - v1[2] * v2[0], v1[0] * v2[1] - v1[1] * v2[0]);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const float t)
{
    return vec3(v1[0] + t, v1[1] + t, v1[2] + t);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const float t)
{
    return vec3(v1[0] - t, v1[1] - t, v1[2] - t);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const float t)
{
    return vec3(v1[0] * t, v1[1] * t, v1[2] * t);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const float t)
{
    return vec3(v1[0] / t, v1[1] / t, v1[2] / t);
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
    return v - n * (2 * dot(v, n));
}

__host__ __device__ inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
{
    vec3 uv = unit_vector(v);
    auto dt = dot(uv, n);
    auto discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0.0f)
    {
        refracted = (uv - n * dt) * ni_over_nt - n * sqrt(discriminant);
        return true;
    }
    return false;
}

inline std::istream& operator>>(std::istream& is, vec3& t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& is, vec3& t)
{
    is << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return is;
}

// Specific versions

struct rgb : public vec3
{
    __host__ __device__ rgb(){};
    __host__ __device__ rgb(float e0, float e1, float e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ rgb(const vec3& v)
    {
        e[0] = v[0];
        e[1] = v[1];
        e[2] = v[2];
    }
    __host__ __device__ inline float r() const
    {
        return e[0];
    }
    __host__ __device__ inline void r(float v)
    {
        e[0] = v;
    }
    __host__ __device__ inline float g()
    {
        return e[1];
    }
    __host__ __device__ inline void g(float v)
    {
        e[1] = v;
    }
    __host__ __device__ inline float b()
    {
        return e[2];
    }
    __host__ __device__ inline void b(float v)
    {
        e[2] = v;
    }
};

} // namespace shared
} // namespace ppt