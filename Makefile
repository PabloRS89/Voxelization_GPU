# Locations of libraries
CUDA      = /usr/local/cuda-6.5
GL_LIB    = /usr/X11R6/lib64/
GL_INC    = /usr/X11R6/include/

# Compilers
CXX       = g++
CXXFLAGS  = -O2 -fPIC -malign-double -m64
NVCC      = nvcc
NVCCFLAGS = -I$(GL_INC) --compiler-options "-O2 -fPIC" --compiler-bindir=/usr/bin/g++-4.8

# Linker flags
LDFLAGS   = -L$(CUDA)/lib64/ -L. -L${GL_LIB}
LIBS      = -lcuda -lcudart -lGL

NBODYCODE = hello.cu

libhello.so:	Makefile hello.h $(NBODYCODE)
	$(NVCC) --shared $(NBODYCODE) -o libhello.so $(NVCCFLAGS) $(LDFLAGS) $(LIBS)

clean:
	\rm -f *.o *.so
