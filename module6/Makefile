all: assignment.cu
	mkdir -p build
	nvcc assignment.cu -std=c++20 -arch=native -I../common/inc -lcufft -o build/assignment.exe
