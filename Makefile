all: simple1 grayscale


simple1: simple1.cu
	nvcc simple1.cu -o simple1 -std=c++11

grayscale: grayscale1.cu
	nvcc grayscale1.cu -o grayscale -lopencv_core -lopencv_imgproc -lopencv_highgui
