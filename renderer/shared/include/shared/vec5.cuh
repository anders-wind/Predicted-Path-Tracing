#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

namespace ppt
{
namespace shared
{

struct vec5
{
    float e[5];
    __host__ __device__ vec5(){};
    __host__ __device__ vec5(float e0, float e1, float e2, float e3, float e4)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
        e[3] = e3;
        e[4] = e4;
    }

    __host__ __device__ vec5(const vec3& v)
    {
        e[0] = v[0];
        e[1] = v[1];
        e[2] = v[2];
        e[3] = 0;
        e[4] = 0;
    }

    // unary operators
    __host__ __device__ inline const vec5& operator+() const
    {
        return *this;
    }
    __host__ __device__ inline vec5 operator-() const
    {
        return vec5(-e[0], -e[1], -e[2], -e[3], -e[4]);
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
    __host__ __device__ inline vec5& operator+=(const vec5& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        e[3] += v.e[3];
        e[4] += v.e[4];
        return *this;
    }

    __host__ __device__ inline vec5& operator-=(const vec5& v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        e[3] -= v.e[3];
        e[4] -= v.e[4];
        return *this;
    }

    __host__ __device__ inline vec5& operator*=(const vec5& v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        e[3] *= v.e[3];
        e[4] *= v.e[4];
        return *this;
    }

    __host__ __device__ inline vec5& operator/=(const vec5& v)
    {
        e[0] /= v.e[0];
        e[1] /= v.e[1];
        e[2] /= v.e[2];
        e[3] /= v.e[3];
        e[4] /= v.e[4];
        return *this;
    }

    __host__ __device__ inline vec5& operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        e[3] *= t;
        e[4] *= t;
        return *this;
    }

    __host__ __device__ inline vec5& operator/=(const float t)
    {
        e[0] /= t;
        e[1] /= t;
        e[2] /= t;
        e[3] /= t;
        e[4] /= t;
        return *this;
    }

    __host__ __device__ inline vec5& operator-=(const float t)
    {
        e[0] -= t;
        e[1] -= t;
        e[2] -= t;
        e[3] -= t;
        e[4] -= t;
        return *this;
    }

    __host__ __device__ inline vec5& operator+=(const float t)
    {
        e[0] += t;
        e[1] += t;
        e[2] += t;
        e[3] += t;
        e[4] += t;
        return *this;
    }

    __host__ __device__ inline float length() const
    {
        return sqrt(squared_length());
    }

    __host__ __device__ inline vec5& v_square()
    {
        e[0] *= e[0];
        e[1] *= e[1];
        e[2] *= e[2];
        e[3] *= e[3];
        e[4] *= e[4];
        return *this;
    }

    __host__ __device__ inline vec5& v_sqrt()
    {
        e[0] = sqrt(e[0]);
        e[1] = sqrt(e[1]);
        e[2] = sqrt(e[2]);
        e[3] = sqrt(e[3]);
        e[4] = sqrt(e[4]);
        return *this;
    }

    __host__ __device__ inline float squared_length() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3] + e[4] * e[4];
    }

    __host__ __device__ inline vec5& make_unit_vector()
    {
        float k = 1.0f / length();
        e[0] *= k;
        e[1] *= k;
        e[2] *= k;
        return *this;
    }
};

__host__ __device__ inline vec5 operator+(const vec5& v1, const vec5& v2)
{
    return vec5(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2], v1[3] + v2[3], v1[4] + v2[4]);
}

__host__ __device__ inline vec5 operator-(const vec5& v1, const vec5& v2)
{
    return vec5(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2], v1[3] - v2[3], v1[4] - v2[4]);
}

__host__ __device__ inline vec5 operator*(const vec5& v1, const vec5& v2)
{
    return vec5(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2], v1[3] * v2[3], v1[4] * v2[4]);
}

__host__ __device__ inline vec5 operator/(const vec5& v1, const vec5& v2)
{
    return vec5(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2], v1[3] / v2[3], v1[4] / v2[4]);
}

__host__ __device__ inline vec5 operator+(const vec5& v1, const float t)
{
    return vec5(v1[0] + t, v1[1] + t, v1[2] + t, v1[3] + t, v1[4] + t);
}

__host__ __device__ inline vec5 operator-(const vec5& v1, const float t)
{
    return vec5(v1[0] - t, v1[1] - t, v1[2] - t, v1[3] - t, v1[4] - t);
}

__host__ __device__ inline vec5 operator*(const vec5& v1, const float t)
{
    return vec5(v1[0] * t, v1[1] * t, v1[2] * t, v1[3] * t, v1[4] * t);
}

__host__ __device__ inline vec5 operator/(const vec5& v1, const float t)
{
    return vec5(v1[0] / t, v1[1] / t, v1[2] / t, v1[3] / t, v1[4] / t);
}

__host__ __device__ inline vec5 unit_vector(vec5 v)
{
    return v / v.length();
}

inline std::istream& operator>>(std::istream& is, vec5& t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2] >> t.e[3] >> t.e[4];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, vec5& t)
{
    os << t.e[0] << " " << t.e[1] << " " << t.e[2] << " " << t.e[3] << " " << t.e[4];
    return os;
}

} // namespace shared
} // namespace ppt