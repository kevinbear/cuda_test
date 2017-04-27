#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define DATA_SIZE 1048576

int data[DATA_SIZE];

bool InitCUDA(){

	int count;


	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}


	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}

	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);
	return true;
}

void GenerateNumbers(int *number, int size){
	for (int i = 0; i < size; i++) {
		number[i] = rand() % 10;
	}
}

__global__ static void sumOfSquares(int *num, int* result, clock_t* time) {

	int sum = 0;
	int i;
	clock_t start = clock();
	for (i = 0; i < DATA_SIZE; i++) {
		sum += num[i] * num[i];
	}
	*result = sum;
	*time = clock() - start;
}

int main() {
	if (!InitCUDA()) {
		return 0;
	}
	int nDevices;
	printf("CUDA initialized.\n");

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device %d has compute capability %d.%d.\n", i, prop.major, prop.minor);
		printf("NO.1\n");
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		//printf("  Memory Clock Rate (KHz): %d\n",
		//	prop.memoryClockRate);
		//printf("  Memory Bus Width (bits): %d\n",
		//	prop.memoryBusWidth);
		//printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
		//	2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		//printf("Maximum number of threads per block %d\n", prop.maxThreadsPerBlock);
		printf(" clockRate%d %f\n", prop.clockRate, ((float)prop.clockRate)/1000000);
		printf(" totalGlobalMem    %lu\n", prop.totalGlobalMem);
		printf(" maxThreadsPerBlock  %d\n", prop.maxThreadsPerBlock);
	}
	GenerateNumbers(data, DATA_SIZE);

	int* gpudata, *result;
	clock_t* time;

	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int));
	cudaMalloc((void**)&time, sizeof(clock_t));


	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	int sum;
	clock_t time_used;
	clock_t realtime=clock();
	sumOfSquares << <1, 1, 0 >> >(gpudata, result, time);


	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	time_used = clock() - realtime;






	clock_t cpuclock = clock();
	int sum1 = 0;
	for (int i = 0; i < DATA_SIZE; i++) {
		sum1 += data[i] * data[i];
	}	
	printf("NO.2\n");
	printf("cuda sum: %d, CPU sum: %d\n", sum,sum1);
	clock_t cpuend = clock();
	//printf("CPU sum: %d\n", sum);
	printf("NO.3\n");
	printf("GPUtime %f\n", (float)time_used / CLOCKS_PER_SEC);
	printf("clocktime %f\n", (float)(cpuend - cpuclock) / CLOCKS_PER_SEC);
	system("PAUSE");
	return 0;
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	//addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
