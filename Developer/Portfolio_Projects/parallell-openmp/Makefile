CC = clang
CFLAGS = -O1 -Wall -Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/20.1.0/include
LIBS = -lm -lomp -L/opt/homebrew/Cellar/libomp/20.1.0/lib

ifdef GPROF
CFLAGS += -pg
endif

SRCS = main.c tests.c sequential.c parallel.c

test: $(SRCS) utils.h
	$(CC) $(CFLAGS) $(SRCS) -o test $(LIBS)

.PHONY: clean

clean:
	rm -f *.o test gmon.out *~