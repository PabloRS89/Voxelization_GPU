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
NVCCFLAGS 	= -I. -arch sm_30 --compiler-options "-O3 -fPIC -malign-double -m64", --compiler-bindir=/usr/bin/g++-4.8
LDFLAGS   	= -L$(CUDA)/lib64/
LIBS      	= -lcuda -lcudart
SRC 		= r3d.cu v3d.cu
DEPS 		= r3d.h v3d.h Makefile
OBJ 		= $(SRC:.cu=.o)

all: libr3d.so

libr3d.so: $(OBJ)
	ar -rcs $@ $^

%.o: %.cu $(DEPS)
	$(NVCC) --shared -c -o $@ $< $(NVCCFLAGS) $(LDFLAGS) $(OPT) $(LIBS)

clean:
	rm -rf *.o *.so 
