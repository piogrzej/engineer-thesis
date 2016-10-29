#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_functions.h>			//PO CO TO INCLUDOWALEM?
#include <stdio.h>

#define VECTOR_INITIAL_SIZE 10;

template<class V>
class d_vector
{
private:
	int* valuesSize;
	int* size;
	V* values;
public:
	__host__ __device__  d_vector()
	{
		int tmp = sizeof(V)*VECTOR_INITIAL_SIZE;
		cudaMalloc(&(this->values),tmp);
		cudaMalloc(&(this->size),sizeof(int));
		cudaMalloc(&(this->valuesSize),sizeof(int));
		*(this->size) = 0;
		*(this->valuesSize) = VECTOR_INITIAL_SIZE;
	}
	__host__ __device__ ~d_vector()
	{
		cudaFree((this->values));
		cudaFree((this->size));
		cudaFree((this->valuesSize));
	}
	__device__ void add(V r)
	{
		if(*size==(*valuesSize-1)){
			V* tmp;
			int sizeOf=(*valuesSize)*sizeof(V);
			cudaMalloc(&tmp,sizeOf);
			for(int i=0; i<*size;++i)
				tmp[i]=values[i];
			cudaFree(values);
			*(this->valuesSize)+=VECTOR_INITIAL_SIZE;
			sizeOf=(*valuesSize)*sizeof(V);
			cudaMalloc(&values,sizeOf);
			for(int i=0; i<*size;++i)
				values[i]=tmp[i];
			cudaFree(tmp);
		}
		values[*size] = r;
		++(*(size));
	}
	__device__ void rem(const int index)
	{
	    if(index == ((*valuesSize)-1))
		    --(*size);
	    else if(index < ((*valuesSize)-1))
	    {
	      for(int i=index; i<(*size-1);++i)
	      {
		      values[i]=values[i+1];
	      }
	      --(*size);
	    }
	}
	__device__ V get(const int index)
	{
		if(index < *(this->size))
			return this->values[index];
		else
		{
			printf("Index out of range!\n");
			//TODO return "error" msg
		}
	}
	__device__ V operator[](const int index)
	{
		return this->get(index);
	}

	__device__ int length()
	{
		return *(this->size);
	}
};
