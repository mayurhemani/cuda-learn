#include <cstdio>

__global__ void mykern() {
	int idx = threadIdx.x;
	
	__shared__ int array[128];

	array[idx] = threadIdx.x;
	__syncthreads();

	if (idx < 127) {
		int temp = array[idx + 1];
		__syncthreads();
		array[idx] = temp;
		__syncthreads();
	}

	printf("%d %d\n", array[idx], threadIdx.x);
}


int main() {
	
	mykern<<<1, 128>>>();
	
	cudaDeviceSynchronize();

	return 0;
		
}
