#!/bin/bash

#g++ -fpic -g -c -Wall hello.cu
nvcc -m64 -o voxel.so --shared -Xcompiler -fPIC voxelCuda.cu
#make a shared library, not a static library (thanks cat plus plus)
#g++ -shared -o hello.so hello.o
