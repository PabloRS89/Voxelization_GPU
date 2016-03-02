#!/bin/bash

#g++ -fpic -g -c -Wall hello.cu
nvcc -m64 -o libr3d.so --shared -Xcompiler -fPIC r3d.cu v3d.cu
rm *.o
#make a shared library, not a static library (thanks cat plus plus) voxelCuda.cu
#g++ -shared -o hello.so hello.o
