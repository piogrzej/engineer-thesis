#define VECTOR_INITIAL_SIZE 10;

typedef float floatingPoint;

struct point
{
	floatingPoint x;
	floatingPoint y;
	__host__ __device__ point(){
	};
	__host__ __device__ ~point(){
	};
	__host__ __device__ point(floatingPoint v1, floatingPoint v2)
	{
		this->x=v1;
		this->y=v2;
	}
};

struct rect
{
	point topLeft;
	point bottmRight;
	__host__ __device__ rect(){};
	__host__ __device__ ~rect(){};
	__host__ __device__ rect(point tl, point br){
		topLeft.x=tl.x;
		topLeft.y=tl.y;
		bottmRight.x =br.x;
		bottmRight.y=br.y;
	}
};

template<class V>
class d_vector{
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
