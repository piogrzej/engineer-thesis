#include<cuda.h>
#include<stdio.h>
#include <iostream>
#include"d_vector.h"

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

//__device__ int fact(int f)
//{
//  if(f == 0)
//    return 1;
//  else
//    return f * fact(f - 1);
//}
//
//__global__ void fillVector(vector* a){
//    a->values[blockIdx.x]=blockIdx.x;
//}
//
//__global__ void printVector(vector* a){
//    printf("%i \n",a->values[blockIdx.x]);
//}
//
////Example of recursion and calling kernels form kernels
//__global__ void factVector(vector* a){
//	a->values[blockIdx.x]=fact(a->values[blockIdx.x]);
//	//printVector<<<1,1>>>(a);
//}

template<typename V> __host__ __device__  d_vector<V>::d_vector(){
	int tmp = sizeof(V)*VECTOR_INITIAL_SIZE;
	cudaMalloc(&(this->values),tmp);
	cudaMalloc(&(this->size),sizeof(int));
	cudaMalloc(&(this->valuesSize),sizeof(int));
	*(this->size) = 0;
	*(this->valuesSize) = VECTOR_INITIAL_SIZE;
}

template<typename V> __host__ __device__  d_vector<V>::~d_vector(){
	cudaFree((this->values));
	cudaFree((this->size));
	cudaFree((this->valuesSize));
}

template<typename V> __device__ void d_vector<V>::add(V r){
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


template<typename V> __device__ void d_vector<V>::rem(const int index)
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

template<typename V> __device__ V d_vector<V>::operator[](const int index){
	return this->get(index);
}

template<typename V> __device__ int d_vector<V>::length(){
	return *(this->size);
}


template<typename V> __device__ V d_vector<V>::get(int index){
	if(index < *(this->size))
		return this->values[index];
	else
	{
		printf("Index out of range!\n");
		//TODO return "error" msg
	}
}
/*
__global__ void testVector(){
	d_vector<rect> *d_v = new d_vector<rect>();
	for(int i=0; i<100;++i)
		d_v->add(rect(point(0,0),point(i,i)));
	printf("%d\n",d_v->length());
	d_v->rem(10);
	printf("%d\n",d_v->length());
	for(int i=0; i<100;++i){
//		rect r = d_v->get(i);
		rect r = (*d_v)[i];
		printf("%f;%f %f;%f\n",r.topLeft.x,r.topLeft.y,r.bottmRight.x,r.bottmRight.y);
	}

	delete(d_v);
}

int main() {

	auto t1 = Clock::now();
	testVector<<<1,1>>>();
	cudaDeviceSynchronize();
	auto t2 = Clock::now();
		std::cout << "Execution time: "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
				<< " ms" << std::endl;

    return 0;
}*/
