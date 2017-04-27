CC=g++
NC=nvcc
.PHONY: test clean

memtest:
	$(NC) cuda_code/cuda.cu -c cuda.o
	$(CC) cuda.o -L/usr/local/cuda-8.0/lib64 -lcudart -o  CUDA_TEST
golbaltest:
	$(NC) cuda_code/global.cu -c global.o
	$(CC) global.o -L/usr/local/cuda-8.0/lib64 -lcudart -o  GLO_TEST
spacetest:
	$(NC) cuda_code/space.cu -c space.o
	$(CC) space.o -L/usr/local/cuda-8.0/lib64 -lcudart -o  SPACE_TEST
device_pro:
	$(NC) cuda_code/device.cu -c device.o
	$(CC) device.o -L/usr/local/cuda-8.0/lib64 -lcudart -o  DEV_PRO
kernel:
	$(NC) cuda_code/kernel.cu -c kernel.o
	$(CC) kernel.o -L/usr/local/cuda-8.0/lib64 -lcudart -o  KER
cutess:
	$(NC) cute/cute.cu -c cute.o
	$(CC) cute.o -I/usr/include/opencv2 -lopencv_imgproc -lopencv_features2d -lopencv_core -lopencv_highgui -L/usr/local/cuda-8.0/lib64 -lcudart -o  CUTESS
clean:
	rm -rf *.*~ *.o 
	rm -rf GLO_TEST CUDA_TEST SPACE_TEST DEV_PRO KER CUTESS
cpu_v:
	$(CC) cpu_version/cpu_version_yuv2rgb.cpp  -c 
	$(CC) cpu_version_yuv2rgb.o -I/usr/include/opencv2 -lopencv_imgproc -lopencv_features2d -lopencv_core -lopencv_highgui -o CPU_T
