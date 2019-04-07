NVCC=/usr/local/cuda/bin/nvcc
CCFLAGS= -std=c++11

all: simple1 grayscale threadorder


simple1: simple1.cu
	$(NVCC) simple1.cu -o simple1 $(CCFLAGS) -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ -D_FORCE_INLINES

grayscale: grayscale.cu
	$(NVCC) grayscale.cu -o grayscale $(CCFLAGS) -lopencv_core -lopencv_imgproc -lopencv_highgui

threadorder: order.cu
	$(NVCC) order.cu -o order $(CCFLAGS)

