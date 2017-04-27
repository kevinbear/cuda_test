#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory.h>
#define w 640
#define h 480
#define ysize w*h
//======namespace=======//
using namespace std;
using namespace cv;
//======================//
//===========kernel function=========//
__global__ void yuv2rgb(unsigned char *d_y ,unsigned char *d_u,unsigned char *d_v,unsigned char *d_r,unsigned char *d_g,unsigned char *d_b)
{
	float rr=0,gg=0,bb=0;
	int ind=blockDim.x*blockIdx.x+threadIdx.x; //640*i+j thread limited 1024 so I use 480
	int ind_uv=((blockDim.x)>>1)*(blockIdx.x>>1)+(threadIdx.x>>1);
	
	rr=d_y[ind]+(1.13983 * (d_v[ind_uv] - 128));
        d_r[ind]=(rr>255 ? 255:(rr<0 ? 0:rr));
	gg= d_y[ind] - (0.39465 * (d_u[ind_uv] - 128) + (0.58060 * (d_v[ind_uv] - 128)));
        d_g[ind]=(gg>255 ? 255:(gg<0 ? 0:gg));
        bb= d_y[ind] + (2.03211 * (d_u[ind_uv] - 128));
        d_b[ind]=(bb>255 ? 255:(bb<0 ? 0:bb));
}
//=================================//

FILE *fp=fopen("input_video/1.yuv","rb");
FILE *fp1=fopen("protect.yuv","wb");

unsigned char *in_y=NULL,*in_u=NULL,*in_v=NULL,*out_r=NULL,*out_g=NULL,*out_b=NULL;
unsigned char *d_y,*d_r,*d_u,*d_v,*d_g,*d_b;

int main()
{
	cudaEvent_t dstart,dend;
	int process_frmaecount=0;
	float during_time=0,during[140]={0},total=0;
	cudaEventCreate(&dstart);
	cudaEventCreate(&dend);
	//=================CPU (host memory allocte)=================//	
	in_y=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	in_u=(unsigned char*)malloc(sizeof(unsigned char)*ysize/4);
	in_v=(unsigned char*)malloc(sizeof(unsigned char)*ysize/4);
	out_r=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	out_g=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	out_b=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	//==========================================================//
	//===============GPU (device memory allocte)================//
	cudaMalloc((void **)&d_y,sizeof(unsigned char)*ysize);
	cudaMalloc((void **)&d_u,sizeof(unsigned char)*ysize/4);
	cudaMalloc((void **)&d_v,sizeof(unsigned char)*ysize/4);
	cudaMalloc((void **)&d_r,sizeof(unsigned char)*ysize);
	cudaMalloc((void **)&d_g,sizeof(unsigned char)*ysize);
	cudaMalloc((void **)&d_b,sizeof(unsigned char)*ysize);
	
	cudaMemset(d_r,0,sizeof(unsigned char)*ysize);
	cudaMemset(d_g,0,sizeof(unsigned char)*ysize);
	cudaMemset(d_b,0,sizeof(unsigned char)*ysize);
	//==========================================================//
	Mat Frame(480,640,CV_8UC3);
	while( fread(in_y,1,ysize,fp) != NULL)
	{
		fread(in_u,1,ysize/4,fp);
		fread(in_v,1,ysize/4,fp);
		//============ data from CPU to GPU =============//	
		cudaMemcpy(d_y,in_y,sizeof(unsigned char)*ysize,cudaMemcpyHostToDevice);
		cudaMemcpy(d_u,in_u,sizeof(unsigned char)*ysize/4,cudaMemcpyHostToDevice);
		cudaMemcpy(d_v,in_v,sizeof(unsigned char)*ysize/4,cudaMemcpyHostToDevice);
		//===============================================//
		cudaEventRecord(dstart, 0);
		yuv2rgb <<<h,w>>>(d_y,d_u,d_v,d_r,d_g,d_b); //yuv2rgb kernel function
		cudaEventRecord(dend, 0);
		cudaEventSynchronize(dend);
		cudaEventElapsedTime(&during_time, dstart, dend);
		during[process_frmaecount]=during_time;
		process_frmaecount++;
		printf("frame%d using time=%f\n",process_frmaecount,during[process_frmaecount-1]);		
		//============ data from GPU to CPU =============//
		cudaMemcpy(out_r,d_r,sizeof(unsigned char)*ysize,cudaMemcpyDeviceToHost);
		cudaMemcpy(out_g,d_g,sizeof(unsigned char)*ysize,cudaMemcpyDeviceToHost);
		cudaMemcpy(out_b,d_b,sizeof(unsigned char)*ysize,cudaMemcpyDeviceToHost);	
		//===============================================//
		//==========placed the bgr componet to Frame data struct=========//
		for(int i=0;i<h*w;i++)
		{
			Frame.data[3*i]=out_b[i];
			Frame.data[3*i+1]=out_g[i];
			Frame.data[3*i+2]=out_r[i];
		}
		//==============================================================//
		//=============display the result===============//
		imshow("transsmisson",Frame);
		cvWaitKey(10);
		if(cvWaitKey(10)>=0) break;
			
	}

	for(int i=0;i<process_frmaecount;i++)
		total+=during[i];

	printf("GPU average per frame using time=%f ms\n",total/process_frmaecount);
	printf("finish job\n");
	free(in_y),free(in_u),free(in_v);
	free(out_r),free(out_g),free(out_b);
	cudaFree(d_y),cudaFree(d_u),cudaFree(d_v),cudaFree(d_r),cudaFree(d_g),cudaFree(d_b);
	fclose(fp);
	return 0;
}
