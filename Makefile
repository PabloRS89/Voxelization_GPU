######################################################
#
#	Makefile
#
#	libr3d.a
#
#	See Readme.md for usage
#
######################################################

CUDA      	= /usr/local/cuda-6.5
NVCC 		= nvcc
NVCCFLAGS 	= -arch=sm_30 -Xcompiler -fPIC --compiler-options -Wall
LIBS      	= -lcuda -lcudart -lcufft -lcublas -lm
SRC 		= r3d.cu
DEPS 		= r3d.h Makefile
OBJ 		= $(SRC:.cu=.o)

libr3d.so: r3d.h $(SRC)
	$(NVCC) -shared -o libr3d.so $(SRC) $(NVCCFLAGS) $(LIBS)

clean:
	\rm -f *.o