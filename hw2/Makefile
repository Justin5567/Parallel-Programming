CC = gcc
CXX = clang++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread -ffast-math
hw2b: CC = mpicc 
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp 
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
