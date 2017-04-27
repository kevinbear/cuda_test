#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <memory.h>
#define num 100
int* host_a = NULL;
int* host_b = NULL;

int main()
{
	bool flag=true;
	/*set host_memory*/
	host_a=(int*)malloc(sizeof(int)*num);
	host_b=(int*)malloc(sizeof(int)*num);
	memset(host_a,0,num);
	memset(host_b,0,num);

	/*initialization*/	
	for(int i=0;i<num;i++)
	{
		host_a[i]=i+1;
	}

	/*set device memory*/
	int* device;
	cudaError_t error;
	error=cudaMalloc((void**) &device,sizeof(int)*num);
	printf("cudaMalloc():%s\n",cudaGetErrorString(error)); //set memory fail print errror

	error=cudaMemcpy(device,host_a,sizeof(int)*num,cudaMemcpyHostToDevice);// host_a -> device
	printf("cudaMemcpy(host_a => device:%s)\n",cudaGetErrorString(error));

	error=cudaMemcpy(host_b,device,sizeof(int)*num,cudaMemcpyDeviceToHost);// device -> host_b
	printf("cudaMemcpy(device => host_a:%s)\n",cudaGetErrorString(error));

	for(int i=0;i<num;i++)
	{
		if(host_a[i]!=host_b[i])
		{
			flag=false;
			break;
		}
	}
	printf("check host_a==host_b%s",flag?"pass":"worng");
	
	error=cudaFree(device);//check cuda malloc free memory space
	printf("cudaFree(device):%s\n",cudaGetErrorString(error));
	free(host_a);
	free(host_b);
	cudaFree(device);
	return 0;
}
