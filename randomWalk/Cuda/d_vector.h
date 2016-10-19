#pragma once

#include <vector_functions.h>

#define VECTOR_INITIAL_SIZE 10;

typedef float2 point2;
typedef float  floatingPoint;
#define make_point2(a,b) make_float2(a,b)


struct RectCuda
{
  float2 topLeft;
  float2 bottmRight;
	__host__ __device__ RectCuda(float2 topLeft, float2 bottmRight) :
	    topLeft(topLeft),
	    bottmRight(bottmRight)	{};
	__host__ __device__ RectCuda(){};
	__host__ __device__ RectCuda(float x0,float x1,
	                             float y0,float y1)
	{
		topLeft.x=x0;
		topLeft.y=y0;
		bottmRight.x =x1;
		bottmRight.y=y1;
	}

	__host__ __device__ ~RectCuda(){};
	__host__ __device__ __forceinline__ bool contains(RectCuda const& rect)
	{
           return  topLeft.x <= rect.topLeft.x &&
		   topLeft.y <= rect.topLeft.y &&
		   bottmRight.x >= rect.bottmRight.x &&
		   bottmRight.y >= rect.bottmRight.y;
	}
};

template<class V>
class d_vector
{
private:
	int* valuesSize;
	int* size;
	V* values;
public:
	__host__ __device__  d_vector();
	__host__ __device__ ~d_vector();
	__device__ void add(V r);
	__device__ void rem(const int index);
	__device__ V get(const int index);
	__device__ V operator[](const int index);
	__device__ int length();
};
