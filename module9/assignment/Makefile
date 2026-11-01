all: assignment.cu
	mkdir -p build
	nvcc assignment.cu -std=c++20 -arch=native -I../../common/inc -o build/assignment.exe -lcurand -lnppc -lnppial -lnppif -lnppist
