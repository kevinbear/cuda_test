#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <memory.h>
#define stnum 30
__global__ void hello (char *arg)
{
	char test[stnum]="hello cuda8.0 test function";
	for(int i=0;test[i]!='\0';i++)
		arg[i]=test[i];
}

int main()
{
	char* host_s=NULL;
	host_s=(char*)malloc(sizeof(char)*stnum);
	memset(host_s,'\0',stnum);

	char* device;
	cudaError_t error;

	/*device memory malloc*/
	error=cudaMalloc((void **) &device,sizeof(char)*stnum);
	printf("cudaMalloc():%s\n",cudaGetErrorString(error));
	/*parallel function*/
	hello<<<1,1>>>(device);
	error=cudaMemcpy(host_s,device,sizeof(char)*stnum,cudaMemcpyDeviceToHost);
	printf("cudaMemcpy(device -> host):%s\n",cudaGetErrorString(error));
	/*output device -> host memory string*/
	printf("Hello function:%s\n",host_s);
	/*free memory storage*/
	free(host_s);
	cudaFree(device);
	return 0;
}
 
