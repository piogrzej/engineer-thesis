#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

typedef float2 point2;
typedef float  floatingPoint;
#define make_point2(a,b) make_float2(a,b);


class d_Rect
{
public:
	point2 topLeft;
	point2 bottomRight;
	__host__ __device__ d_Rect(point2 topLeft, point2 bottmRight) : topLeft(topLeft),bottomRight(bottomRight){};
	__host__ __device__ d_Rect(floatingPoint xTL, floatingPoint yTL, floatingPoint xBR, floatingPoint yBR);
	__host__ __device__ d_Rect(){};
	__host__ __device__ ~d_Rect(){};
	__host__ __device__ __forceinline__ bool contains(point2 p) const
	{
	    if ((p.x >= topLeft.x) && (p.y >= topLeft.y) && (p.x <= bottomRight.x) && (p.y <= bottomRight.y))
	    {
	        return true;
	    }
	    else return false;
	};
	__host__ __device__ __forceinline__ bool contains(d_Rect const& rect)
	{
        return  topLeft.x <= rect.topLeft.x &&
        topLeft.y <= rect.topLeft.y &&
        bottomRight.x >= rect.bottomRight.x &&
        bottomRight.y >= rect.bottomRight.y;
	};
	__host__ __device__ floatingPoint getWidth() const;
	__host__ __device__ floatingPoint getHeigth() const;
	__host__ __device__ floatingPoint getPerimeter() const;
	__host__ __device__ point2 getPointFromNindex(int index, int Nsample);
	__host__ __device__ d_Rect createGaussianSurfaceX(floatingPoint factorX) const {return createGaussianSurface(factorX, 1);}
	__host__ __device__ d_Rect createGaussianSurfaceY(floatingPoint factorY) const {return createGaussianSurface(1, factorY);}
	__host__ __device__ d_Rect createGaussianSurface(floatingPoint factorX, floatingPoint factorY) const;
	__host__ __device__ bool rectsCollision(d_Rect const& r2) const;
	__host__ __device__ bool operator==(const d_Rect& r2) const;
	__host__ __device__ bool operator!=(const d_Rect& r2) const { return !(*this == r2); }
};
