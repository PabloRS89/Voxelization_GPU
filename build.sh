#!/bin/bash

#g++ -fpic -g -c -Wall hello.cu
nvcc -m64 -o libr3d.so --shared -Xcompiler -fPIC voxelCuda.cu r3d.c
#make a shared library, not a static library (thanks cat plus plus)
#g++ -shared -o hello.so hello.o
