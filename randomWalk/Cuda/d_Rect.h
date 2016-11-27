#ifndef D_RECT_H
#define D_RECT_H

#include <cuda.h>
#include <cuda_runtime.h>

typedef double2 point2;
typedef double  floatingPoint;
#define make_point2(a,b) make_double2(a,b);


class d_Rect
{
public:
	point2 topLeft;
	point2 bottomRight;
	__host__ __device__ d_Rect(point2 tL, point2 bR) : topLeft(tL),bottomRight(bR){};
	__host__ __device__ d_Rect(floatingPoint xTL, floatingPoint yTL, floatingPoint xBR, floatingPoint yBR);
	__host__ __device__ d_Rect(){ bottomRight = topLeft = make_point2(-1, -1);  } // incorrect rect
	__host__ __device__ ~d_Rect(){};
	__device__ __forceinline__ bool contains(point2 p) const
	{
	    if ((p.x >= topLeft.x) &&
	        (p.y >= topLeft.y) &&
	        (p.x <= bottomRight.x) &&
	        (p.y <= bottomRight.y))
	    {
	        return true;
	    }
	    else
	        return false;
	};
	__host__ __device__ __forceinline__ bool contains(d_Rect const& rect)
	{
        return  topLeft.x <= rect.topLeft.x &&
        topLeft.y <= rect.topLeft.y &&
        bottomRight.x >= rect.bottomRight.x &&
        bottomRight.y >= rect.bottomRight.y;
	};
	__device__ floatingPoint getWidth() const;
	__device__ floatingPoint getHeigth() const;
	__device__ floatingPoint getPerimeter() const;
	__device__ point2 getPointFromNindex(int index, int Nsample);
	__device__ d_Rect createGaussianSurfaceX(floatingPoint factorX) const {return createGaussianSurface(factorX, 1);}
	__device__ d_Rect createGaussianSurfaceY(floatingPoint factorY) const {return createGaussianSurface(1, factorY);}
	__device__ d_Rect createGaussianSurface(floatingPoint factorX, floatingPoint factorY) const;
	__device__ bool rectsCollision(d_Rect const& r2) const;
	__device__ bool operator==(const d_Rect& r2) const;
	__device__ bool operator!=(const d_Rect& r2) const { return !(*this == r2); }
};
#endif
