######################################################
#
#	Makefile
#
#	for r3d robustness unit testing
#
#	usage: `make`
#
######################################################

R3D_HOME = ..
UTEST_HOME = .
R3D_SRC = r3d_unit_tests.c

CC = gcc
CFLAGS = -Wall -I.
R3D_SRC += $(UTEST_HOME)/utest.c test_helpers.c
INC += -I$(R3D_HOME) -I$(UTEST_HOME)
LIB += -L$(R3D_HOME)
LDFLAGS += -lr3d -lm
R3D_OBJ = $(R3D_SRC:.c=.o)

all: r3d r3d_unit_tests

r3d: 
	make -C $(R3D_HOME)

r3d_unit_tests: $(R3D_OBJ)
	$(CC) $(LIB) $(R3D_OBJ) -o $@ $(LDFLAGS) $(CFLAGS)


%.o: %.c $(DEPS)
	$(CC) $(INC) -c -o $@ $< $(CFLAGS) $(OPT)

clean:
	rm -rf r3d_unit_tests $(R3D_OBJ)
