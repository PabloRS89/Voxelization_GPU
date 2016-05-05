######################################################
#
#	Makefile
#
#	libr3d.a
#
#	See Readme.md for usage
#
######################################################

######## User options #########

# Use single-precision computations
#OPT += -DSINGLE_PRECISION

###############################

CUDA      	= /usr/local/cuda-6.5
NVCC 		= nvcc 
GCC 		= gcc
NVCCFLAGS 	= -arch=sm_30 -Xcompiler -fPIC
CFLAGS 		= -Wall
#LDFLAGS   	= -L$(CUDA)/lib64/ -L.
LIBS      	= -lcuda -lcudart -lcufft -lcublas
SRC 		= r3d.cu v3d.cu
DEPS 		= r3d.h v3d.h Makefile
OBJ 		= $(SRC:.cu=.o)


libr3d.so:	r3d.h  v3d.h $(SRC)
	$(NVCC) -shared -o libr3d.so $(SRC) $(NVCCFLAGS) --compiler-options -Wall $(LIBS)

clean:
	\rm -f *.o

#all: libr3d.a

#libr3d.so: $(OBJ)
#	ar -rs $@ $^

#%.o: %.cu $(DEPS)
#	$(NVCC) -c -o $@ $< $(NVCCFLAGS) $(LDFLAGS) $(LIBS)

#clean:
#	rm -rf *.o *.so 