#include <stdio.h>


__global__ void hello() {
	printf("Hello world! I\'m a thread in block %d\n", blockIdx.x);
}


int main(int argc, char** argv) {
	hello<<<16, 1>>>();

	// this statement will make the printfs() to flush to stdout
	cudaDeviceSynchronize();

	return 0;
}
