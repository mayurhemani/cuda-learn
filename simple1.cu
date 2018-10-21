#include <iostream>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <vector>
#include <functional>

__global__ void add(int* in, int* out) {
	int idx = threadIdx.x;
	out[idx] = in[idx] * in[idx];
}


int main(int argc, char* argv[]) {

	std::vector<int> dd(512);
	std::iota(std::begin(dd), std::end(dd), 1);

	auto sz = 512 * sizeof(int);
	int* din;
	int* dout;

	cudaMalloc( (void**)&din, sz );
	cudaMalloc( (void**)&dout, sz );

	cudaMemcpy(din, &(dd.data()[0]), sz, cudaMemcpyHostToDevice);
	add<<<1, 512>>>(din, dout);
	
	std::vector<int> oo(512, 0);
	cudaMemcpy(&(oo.data()[0]), dout, sz, cudaMemcpyDeviceToHost);

	cudaFree( din );
	cudaFree( dout );

	std::copy(std::begin(oo), std::end(oo), std::ostream_iterator<int>(std::cout, " "));
	std::cout << "\n";

	return 0;
}
