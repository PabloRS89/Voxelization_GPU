#!/bin/bash

#g++ -fpic -g -c -Wall hello.cu
nvcc -arch=sm_30 -m64 -o libr3d.a -lcufft -lcudart -lcublas -shared -Xcompiler -fPIC r3d.cu v3d.cu
#make a shared library, not a static library (thanks cat plus plus) voxelCuda.cu
#g++ -shared -o hello.so hello.o  
