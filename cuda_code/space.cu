#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <memory.h>
/*-- grid -> block -> thread --*/
#define size 12
#define thread_ 3
#define block_ 4 
#define times thread_*block_
/*=============================*/
/*--struct index newtype--*/
struct ind
{
	int block;
	int thread;
};
typedef ind INDEX;
/*========================*/
/*--paralle kernel function--*/
__global__ void share_memory(INDEX* tem_dev)
{
	int dev_block=blockIdx.x;
	int dev_thread=threadIdx.x;
	int thread_of_block_num=blockDim.x;
	int thread_location_of_array=dev_block*thread_of_block_num+dev_thread;
	/*--threads write own block & threadIdx--*/
	tem_dev[thread_location_of_array].block=dev_block;
	tem_dev[thread_location_of_array].thread=dev_thread;
}
/*==========================*/
int main()
{	
	/*--set host & device--*/
	INDEX* host_temp=NULL;
	INDEX *device;
	cudaError_t error;
	/*--set device memory & host memory--*/
	host_temp=(INDEX*)malloc(sizeof(INDEX)*size);
	error=cudaMalloc((void**) &(device),sizeof(INDEX)*size);
	printf("cudaMalloc():%s\n",cudaGetErrorString(error));
	/*--call device kernel--*/
	dim3 blocksize(640,480,4);
	dim3 threadnum(3,3);
	share_memory<<< blocksize, threadnum>>>(device);
	
	cudaMemcpy(host_temp,device,sizeof(INDEX)*size,cudaMemcpyDeviceToHost);
	
	for(int i=0;i<times;i++)
		printf("host_temp[%d]={block:%d,thread:%d}\n",i,host_temp[i].block,host_temp[i].thread);
	free(host_temp);
	cudaFree(device);
	return 0;
}
