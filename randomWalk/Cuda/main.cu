#include <stdio.h>	//printf
#include <chrono>	//timer
#include <iostream>	//cout
#include "mainkernels.h"

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

//__global__ void testVector(){
//	d_vector<d_Rect> *d_v = new d_vector<d_Rect>();
//	point2 tl;
//	tl.x=0;
//	tl.y=0;
//	point2 br;
//	for(int i=0; i<100;++i)
//	{
//		br.x=i;
//		br.y=i;
//		d_v->add(d_Rect(tl,br));
//	}
//	printf("%d\n",d_v->length());
//	d_v->rem(10);
//	printf("%d\n",d_v->length());
//	for(int i=0; i<99;++i){
////		RectCuda r = d_v->get(i);
//		d_Rect r = (*d_v)[i];
//		printf("%f;%f %f;%f\n",r.topLeft.x,r.topLeft.y,r.bottomRight.x,r.bottomRight.y);
//	}
//
//	delete(d_v);
//}

int main() {

//	auto t1 = Clock::now();
//	//testVector<<<1,1>>>();
//	cudaDeviceSynchronize();
//	auto t2 = Clock::now();
//	std::cout << "Execution time: "
//			<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
//			<< " ms" << std::endl;
//
    return 0;
}
