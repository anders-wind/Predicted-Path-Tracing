#pragma once
#include "vec3.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

namespace ppt
{
namespace shared
{

struct vec8
{
    float e[8];
    __host__ __device__ vec8(){};

    __host__ __device__ vec8(float v)
    {
        e[0] = v;
        e[1] = v;
        e[2] = v;
        e[3] = v;
        e[4] = v;
        e[5] = v;
        e[6] = v;
        e[7] = v;
    }

    __host__ __device__ vec8(float e0, float e1, float e2, float e3, float e4, float e5, float e6, float e7)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
        e[3] = e3;
        e[4] = e4;
        e[5] = e5;
        e[6] = e6;
        e[7] = e7;
    }

    __host__ __device__ explicit vec8(vec3 o)
    {
        e[0] = o[0];
        e[1] = o[1];
        e[2] = o[2];
        e[3] = 0;
        e[4] = 0;
        e[5] = 0;
        e[6] = 0;
        e[7] = 0;
    }

    __host__ __device__ explicit vec8(vec5 o)
    {
        e[0] = o[0];
        e[1] = o[1];
        e[2] = o[2];
        e[3] = o[3];
        e[4] = o[4];
        e[5] = 0;
        e[6] = 0;
        e[7] = 0;
    }

    __host__ __device__ explicit vec8(vec3 o, float e3, float e4, float e5, float e6, float e7)
    {
        e[0] = o[0];
        e[1] = o[1];
        e[2] = o[2];
        e[3] = e3;
        e[4] = e4;
        e[5] = e5;
        e[6] = e6;
        e[7] = e7;
    }

    __host__ __device__ explicit vec8(float o[8])
    {
        e[0] = o[0];
        e[1] = o[1];
        e[2] = o[2];
        e[3] = o[3];
        e[4] = o[4];
        e[5] = o[5];
        e[6] = o[6];
        e[7] = o[7];
    }

    // unary operators
    __host__ __device__ inline const vec8& operator+() const
    {
        return *this;
    }

    __host__ __device__ inline vec8 operator-() const
    {
        return vec8(-e[0], -e[1], -e[2], -e[3], -e[4], -e[5], -e[6], -e[7]);
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
    __host__ __device__ inline vec8& operator+=(const vec8& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        e[3] += v.e[3];
        e[4] += v.e[4];
        e[5] += v.e[5];
        e[6] += v.e[6];
        e[7] += v.e[7];
        return *this;
    }

    __host__ __device__ inline vec8& operator-=(const vec8& v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        e[3] -= v.e[3];
        e[4] -= v.e[4];
        e[5] -= v.e[5];
        e[6] -= v.e[6];
        e[7] -= v.e[7];
        return *this;
    }

    __host__ __device__ inline vec8& operator*=(const vec8& v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        e[3] *= v.e[3];
        e[4] *= v.e[4];
        e[5] *= v.e[5];
        e[6] *= v.e[6];
        e[7] *= v.e[7];
        return *this;
    }

    __host__ __device__ inline vec8& operator/=(const vec8& v)
    {
        e[0] /= v.e[0];
        e[1] /= v.e[1];
        e[2] /= v.e[2];
        e[3] /= v.e[3];
        e[4] /= v.e[4];
        e[5] /= v.e[5];
        e[6] /= v.e[6];
        e[7] /= v.e[7];
        return *this;
    }

    __host__ __device__ inline vec8& operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        e[3] *= t;
        e[4] *= t;
        e[5] *= t;
        e[6] *= t;
        e[7] *= t;
        return *this;
    }

    __host__ __device__ inline vec8& operator/=(const float t)
    {
        e[0] /= t;
        e[1] /= t;
        e[2] /= t;
        e[3] /= t;
        e[4] /= t;
        e[5] /= t;
        e[6] /= t;
        e[7] /= t;
        return *this;
    }

    __host__ __device__ inline vec8& operator-=(const float t)
    {
        e[0] -= t;
        e[1] -= t;
        e[2] -= t;
        e[3] -= t;
        e[4] -= t;
        e[5] -= t;
        e[6] -= t;
        e[7] -= t;
        return *this;
    }

    __host__ __device__ inline vec8& operator+=(const float t)
    {
        e[0] += t;
        e[1] += t;
        e[2] += t;
        e[3] += t;
        e[4] += t;
        e[5] += t;
        e[6] += t;
        e[7] += t;
        return *this;
    }

    __host__ __device__ inline float length() const
    {
        return sqrt(squared_length());
    }

    __host__ __device__ inline vec8& v_square()
    {
        e[0] *= e[0];
        e[1] *= e[1];
        e[2] *= e[2];
        e[3] *= e[3];
        e[4] *= e[4];
        e[5] *= e[5];
        e[6] *= e[6];
        e[7] *= e[7];
        return *this;
    }

    __host__ __device__ inline vec8& v_sqrt()
    {
        e[0] = sqrt(e[0]);
        e[1] = sqrt(e[1]);
        e[2] = sqrt(e[2]);
        e[3] = sqrt(e[3]);
        e[4] = sqrt(e[4]);
        e[5] = sqrt(e[5]);
        e[6] = sqrt(e[6]);
        e[7] = sqrt(e[7]);
        return *this;
    }

    __host__ __device__ inline float squared_length() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3] + e[4] * e[4] + e[5] * e[5] +
               e[6] * e[6] + e[7] * e[7];
    }

    __host__ __device__ inline vec8& make_unit_vector()
    {
        float k = 1.0f / length();
        e[0] *= k;
        e[1] *= k;
        e[2] *= k;
        e[3] *= k;
        e[4] *= k;
        e[5] *= k;
        e[6] *= k;
        e[7] *= k;
        return *this;
    }
};

__host__ __device__ inline vec8 operator+(const vec8& v1, const vec8& v2)
{
    return vec8(v1[0] + v2[0],
                v1[1] + v2[1],
                v1[2] + v2[2],
                v1[3] + v2[3],
                v1[4] + v2[4],
                v1[5] + v2[5],
                v1[6] + v2[6],
                v1[7] + v2[7]);
}

__host__ __device__ inline vec8 operator-(const vec8& v1, const vec8& v2)
{
    return vec8(v1[0] - v2[0],
                v1[1] - v2[1],
                v1[2] - v2[2],
                v1[3] - v2[3],
                v1[4] - v2[4],
                v1[5] - v2[5],
                v1[6] - v2[6],
                v1[7] - v2[7]);
}

__host__ __device__ inline vec8 operator*(const vec8& v1, const vec8& v2)
{
    return vec8(v1[0] * v2[0],
                v1[1] * v2[1],
                v1[2] * v2[2],
                v1[3] * v2[3],
                v1[4] * v2[4],
                v1[5] * v2[5],
                v1[6] * v2[6],
                v1[7] * v2[7]);
}

__host__ __device__ inline vec8 operator/(const vec8& v1, const vec8& v2)
{
    return vec8(v1[0] / v2[0],
                v1[1] / v2[1],
                v1[2] / v2[2],
                v1[3] / v2[3],
                v1[4] / v2[4],
                v1[5] / v2[5],
                v1[6] / v2[6],
                v1[7] / v2[7]);
}

__host__ __device__ inline vec8 operator+(const vec8& v1, const float t)
{
    return vec8(v1[0] + t, v1[1] + t, v1[2] + t, v1[3] + t, v1[4] + t, v1[5] + t, v1[6] + t, v1[7] + t);
}

__host__ __device__ inline vec8 operator-(const vec8& v1, const float t)
{
    return vec8(v1[0] - t, v1[1] - t, v1[2] - t, v1[3] - t, v1[4] - t, v1[5] - t, v1[6] - t, v1[7] - t);
}

__host__ __device__ inline vec8 operator*(const vec8& v1, const float t)
{
    return vec8(v1[0] * t, v1[1] * t, v1[2] * t, v1[3] * t, v1[4] * t, v1[5] * t, v1[6] * t, v1[7] * t);
}

__host__ __device__ inline vec8 operator/(const vec8& v1, const float t)
{
    return vec8(v1[0] / t, v1[1] / t, v1[2] / t, v1[3] / t, v1[4] / t, v1[5] / t, v1[6] / t, v1[7] / t);
}

__host__ __device__ inline vec8 unit_vector(vec8 v)
{
    return v / v.length();
}


} // namespace shared
} // namespace ppt