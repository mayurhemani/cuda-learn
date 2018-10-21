#include <iostream>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <vector>
#include <functional>
#include <cstdio>
#include "opencv/cv.h"
#include "opencv/highgui.h"

__global__ void gray(unsigned long* in, unsigned char* out) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char r = in[idx] & 0xFF;
	unsigned char g = (in[idx] >> 8) & 0xFF;
	unsigned char b = (in[idx] >> 16) & 0xFF;
	out[idx] = (unsigned char)( 0.3 * r + 0.59 * g + 0.11 * b );
}
#define cudaCheckErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
					msg, cudaGetErrorString(__err), \
					__FILE__, __LINE__); \
			fprintf(stderr, "*** FAILED - ABORTING\n"); \
			exit(1); \
		} \
	} while (0)


void serial_gray(std::vector<unsigned long> const& in, size_t w, size_t h) {
	cv::Mat out(h, w, CV_8UC1);
	for (size_t i = 0; i < h; ++i) {
		for (size_t j = 0; j < w; ++j) {
			auto p = i * w + j;
			out.at<unsigned char>(i, j) = (unsigned char)( 0.3 * (in[p] & 0xFF) + 0.59 * ((in[p] >> 8)&0xFF) + 0.11 * ((in[p] >> 16)&0xFF) );
		}
	}
	cv::imwrite("gray.jpg", out);
}

int main(int argc, char* argv[]) {

	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " inputimage outputimage\n";
		return 0;
	}

	cv::Mat img = cv::imread(argv[1]);

	std::vector<unsigned long> vinimg(img.rows*img.cols, 0);
	for (int r = 0; r < img.rows; ++r)
		for (int c = 0; c < img.cols; ++c) {
			cv::Vec3b v = img.at<cv::Vec3b>(r, c);
			const auto idx = r * img.cols + c;
			vinimg[idx] |= (v[0] << 16);
			vinimg[idx] |= (v[1] << 8);
			vinimg[idx] |= v[2];
		}
	unsigned long* inimg = vinimg.data();

	//serial_gray(vinimg, img.cols, img.rows);

	unsigned long* din;
	unsigned char* dout;
	
	cv::Mat outimg(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
	unsigned char* outdata = outimg.data;
	auto sz = outimg.rows * outimg.cols;

	cudaMalloc( (void**)&din, sz * sizeof(unsigned long) );
    cudaCheckErrors("cudamalloc fail");
	cudaMalloc( (void**)&dout, sz * sizeof(unsigned char) );
    cudaCheckErrors("cudamalloc fail");

	cudaMemcpy(din, inimg, sz*sizeof(unsigned long), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudamemcpy fail");

	auto const nthreads = 512;
	auto const blocksz = 1 + sz / nthreads;
	gray<<<dim3(blocksz, 1, 1), dim3(nthreads, 1, 1)>>>(din, dout);
	cudaCheckErrors("kernel failed");
	
	cudaMemcpy(outdata, dout, sz, cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudamemcpy fail");

	cudaFree( din );
	cudaFree( dout );
	cudaCheckErrors("cudafree fail");

	cv::imwrite(argv[2], outimg);

	return 0;
}
