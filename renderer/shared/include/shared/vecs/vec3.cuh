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
    __host__ __device__ vec3(float v)
    {
        e[0] = v;
        e[1] = v;
        e[2] = v;
    }
    __host__ __device__ vec3(float e0, float e1, float e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }

    __host__ __device__ explicit vec3(float o[8])
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
        return sqrtf(squared_length());
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
        e[0] = sqrtf(e[0]);
        e[1] = sqrtf(e[1]);
        e[2] = sqrtf(e[2]);
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

    __host__ __device__ inline float sum() const
    {
        return (e[0] + e[1] + e[2]);
    }

    __host__ __device__ inline float average() const
    {
        return sum() / 3.0f;
    }


    // STATICS

    __host__ __device__ inline static vec3 min(const vec3& v1, const vec3& v2)
    {
        return vec3(std::fmin(v1[0], v2[0]), std::fmin(v1[1], v2[1]), std::fmin(v1[2], v2[2]));
    }


    __host__ __device__ inline static vec3 max(const vec3& v1, const vec3& v2)
    {
        return vec3(std::fmax(v1[0], v2[0]), std::fmax(v1[1], v2[1]), std::fmax(v1[2], v2[2]));
    }

    __host__ __device__ inline static vec3 clamp_max(const vec3& v, float max = 1.0)
    {
        return vec3::min(vec3(max), v);
    }

    __host__ __device__ inline static vec3 clamp_min(const vec3& v, float min = 0.0)
    {
        return vec3::max(vec3(min), v);
    }

    __host__ __device__ inline static vec3 clamp(const vec3& v, float min = 0.0, float max = 1.0)
    {
        return clamp_min(clamp_max(v, max), min);
    }

    __host__ __device__ inline static vec3 unit_vector(const vec3& v)
    {
        return v / v.length();
    }

    __host__ __device__ inline static vec3 cross(const vec3& v1, const vec3& v2)
    {
        return vec3(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
    }

    __host__ __device__ inline static float dot(const vec3& v1, const vec3& v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    __host__ __device__ inline static vec3 reflect(const vec3& v, const vec3& n)
    {
        return v - n * (2 * vec3::dot(v, n));
    }

    __host__ __device__ inline static bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
    {
        vec3 uv = vec3::unit_vector(v);
        auto dt = vec3::dot(uv, n);
        auto discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
        if (discriminant > 0.0f)
        {
            refracted = (uv - n * dt) * ni_over_nt - n * sqrt(discriminant);
            return true;
        }
        return false;
    }

    __host__ __device__ inline vec3 operator+(const vec3& v2) const
    {
        return vec3(e[0] + v2[0], e[1] + v2[1], e[2] + v2[2]);
    }

    __host__ __device__ inline vec3 operator-(const vec3& v2) const
    {
        return vec3(e[0] - v2[0], e[1] - v2[1], e[2] - v2[2]);
    }

    __host__ __device__ inline vec3 operator*(const vec3& v2) const
    {
        return vec3(e[0] * v2[0], e[1] * v2[1], e[2] * v2[2]);
    }

    __host__ __device__ inline vec3 operator/(const vec3& v2) const
    {
        return vec3(e[0] / v2[0], e[1] / v2[1], e[2] / v2[2]);
    }

    __host__ __device__ inline vec3 operator+(const float t) const
    {
        return vec3(e[0] + t, e[1] + t, e[2] + t);
    }

    __host__ __device__ inline vec3 operator-(const float t) const
    {
        return vec3(e[0] - t, e[1] - t, e[2] - t);
    }

    __host__ __device__ inline vec3 operator*(const float t) const
    {
        return vec3(e[0] * t, e[1] * t, e[2] * t);
    }

    __host__ __device__ inline vec3 operator/(const float t) const
    {
        return vec3(e[0] / t, e[1] / t, e[2] / t);
    }
};

__host__ __device__ inline vec3 operator+(const float t, const vec3& v)
{
    return v + t;
}

__host__ __device__ inline vec3 operator*(const float t, const vec3& v)
{
    return v * t;
}


// __host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2)
// {
//     return v1.operator+(v2);
// }

// __host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2)
// {
//     return v1.operator-(v2);
// }

// __host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2)
// {
//     return v1.operator*(v2);
// }

// __host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2)
// {
//     return v1.operator/(v2);
// }

// __host__ __device__ inline vec3 operator+(const vec3& v1, const float t)
// {
//     return v1.operator+(t);
// }

// __host__ __device__ inline vec3 operator-(const vec3& v1, const float t)
// {
//     return v1.operator-(t);
// }

// __host__ __device__ inline vec3 operator*(const vec3& v1, const float t)
// {
//     return v1.operator*(t);
// }

// __host__ __device__ inline vec3 operator/(const vec3& v1, const float t)
// {
//     return v1.operator/(t);
// }


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


} // namespace shared
} // namespace ppt