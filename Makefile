_DEPS = rgb.h write_image.h structs.h cpu_trace.h tsc_x86.h
_OBJ = main.c.o write_image.c.o rgb.c.o cpu_trace.c.o
_CUDA_OBJ = gpu_trace.cu.o


# CUDA Compiler and Flags
CUDA_PATH = /opt/cuda-10.2
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

NVCC_FLAGS += -m64 -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets


IDIR =include
CC=g++
CFLAGS=-g -Wall -D_REENTRANT -std=c++0x -pthread
INCLUDE = -I$(IDIR) -I$(CUDA_INC_PATH)
LIBS=-lm -lpng -L$(CUDA_LIB_PATH) -lcudart -lcufft

DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

ODIR = obj
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

CUDA_OBJ = $(patsubst %,$(ODIR)/%,$(_CUDA_OBJ))





raytracer: $(OBJ) $(ODIR)/cuda.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) -o $@ $(INCLUDE) $^ $(LIBS)

$(ODIR)/cuda.o: $(CUDA_OBJ)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^

$(ODIR)/%.c.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $<



.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core
