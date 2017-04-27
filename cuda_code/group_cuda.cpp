#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <time.h>
#define width 640
#define height 480
#define ysize width*height
#define uvsize ysize/4
#define block_num 4
#define CLIP(color)		(unsigned char)((color>0xFF)?0xff:((color<0)?0:color))
#define filelocation "/frame"
#define group_h_start 1
#define group_h_end 479
#define group_w_start 1
#define group_w_end 639
FILE *st=fopen(filelocation,"rb");

__global__ void color_group_cu(input_dy, input_dr, input_dg, input_db,
								std_dy, std_dr, std_dg, std_db)
{
		__share__


}

void YUV2RGB(unsigned char * input_y, unsigned char * input_u, unsigned char * input_v, unsigned char * input_r, unsigned char * input_g, unsigned char * input_b);

int main()
{
	unsigned char *input_y=NULL,*input_u=NULL,*input_v=NULL,*input_r=NULL,*input_g=NULL,*input_b=NULL;
	unsigned char *output_y=NULL,*output_r=NULL,*output_g=NULL,*output_b=NULL;
	unsigned char *dy_group=NULL,*y_group;
	double *std_dy=NULL,*std_dr=NULL,*std_dg=NULL,*std_db=NULL; 
	unsigned char *input_dy=NULL,*input_dr=NULL,*input_dg=NULL,*input_db=NULL;
	cudaError_t ey,er,eg,eb,esy,esr,esg,esb,dyg;

	input_y=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	input_u = (unsigned char*)malloc(sizeof(unsigned char) * uvsize);
	input_v = (unsigned char*)malloc(sizeof(unsigned char) * uvsize);
	input_r=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	input_g=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	input_b=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	output_y=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	output_r=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	output_g=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	output_b=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	y_group=(unsigned char*)malloc(sizeof(unsigned char)*ysize);
	//------------read yuv value---------------//
	while(1)
	{
		if((fread(input_dy,1,ysize,fp1) != '\0'))
		{
			fread(input_du,1,uvsize,fp1);
			fread(input_dv,1,uvsize,fp1);
		}
	}
	YUV2RGB(input_y,input_u,input_v,input_r,input_g,input_b);
	//----------------------------------------//
	printf("-------------INIT memory--------------\n");
	ey=cudaMalloc((void **) &input_dy,sizeof(unsigned char)*ysize);
	printf("input_dy cudaMalloc succes:%s\n",cudaGetErrorString(ey));
	er=cudaMalloc((void **) &input_dr,sizeof(unsigned char)*ysize);
	printf("input_dr cudaMalloc succes:%s\n",cudaGetErrorString(er));
	eg=cudaMalloc((void **) &input_dg,sizeof(unsigned char)*ysize);
	printf("input_dg cudaMalloc succes:%s\n",cudaGetErrorString(eg));
	eb=cudaMalloc((void **) &input_db,sizeof(unsigned char)*ysize);
	printf("input_db cudaMalloc succes:%s\n",cudaGetErrorString(eb));
	esy=cudaMalloc((void **) &std_dy,sizeof(unsigned char));
	printf("std_dy cudaMalloc succes:%s\n",cudaGetErrorString(esy));
	esr=cudaMalloc((void **) &std_dr,sizeof(unsigned char));
	printf("std_dr cudaMalloc succes:%s\n",cudaGetErrorString(esr));
	esg=cudaMalloc((void **) &std_dg,sizeof(unsigned char));
	printf("std_dg cudaMalloc succes:%s\n",cudaGetErrorString(esg));
	esb=cudaMalloc((void **) &std_db,sizeof(unsigned char));
	printf("std_db cudaMalloc succes:%s\n",cudaGetErrorString(esb));
	dyg=cudaMalloc((void **) &dy_group,sizeof(unsigned char));
	printf("dy_group cudaMalloc succes:%s\n",cudaGetErrorString(dyg));
	ey=cudaMemcpy(input_dy,input_y,sizeof(unsigned char)*ysize,cudaMemcpyHostToDevice);
	printf("cudaMemcpy(device -> host)output_y:%s\n",cudaGetErrorString(ey));
	er=cudaMemcpy(input_dr,input_r,sizeof(unsigned char)*ysize,cudaMemcpyHostToDevice);
	printf("cudaMemcpy(device -> host)output_r:%s\n",cudaGetErrorString(er));
	eg=cudaMemcpy(input_dg,input_g,sizeof(unsigned char)*ysize,cudaMemcpyHostToDevice);
	printf("cudaMemcpy(device -> host)output_g:%s\n",cudaGetErrorString(eg));
	eb=cudaMemcpy(input_db,input_b,sizeof(unsigned char)*ysize,cudaMemcpyHostToDevice);
	printf("cudaMemcpy(device -> host)output_b:%s\n",cudaGetErrorString(eb));
	printf("-------------------------------------\n");
	dim3 blocksize(640,480,4)
	dim3 threadnum(3,3);
	color_group_cu<<<blocksize ,ysize>>>(input_dy, input_dr, input_dg, input_db,
										 std_dy, std_dr, std_dg, std_db,
										 y_group,output_y);

	ey=cudaMemcpy(output_y,input_dy,sizeof(unsigned char)*ysize,cudaMemcpyDeviceToHost);
	printf("cudaMemcpy(device -> host)output_y:%s\n",cudaGetErrorString(ey));
	er=cudaMemcpy(output_r,input_dr,sizeof(unsigned char)*ysize,cudaMemcpyDeviceToHost);
	printf("cudaMemcpy(device -> host)output_r:%s\n",cudaGetErrorString(er));
	eg=cudaMemcpy(output_g,input_dg,sizeof(unsigned char)*ysize,cudaMemcpyDeviceToHost);
	printf("cudaMemcpy(device -> host)output_g:%s\n",cudaGetErrorString(eg));
	eb=cudaMemcpy(output_b,input_db,sizeof(unsigned char)*ysize,cudaMemcpyDeviceToHost);
	printf("cudaMemcpy(device -> host)output_b:%s\n",cudaGetErrorString(eb));
	dyg=cudaMemcpy(y_group,dy_group,sizeof(unsigned char)*ysize,cudaMemcpyDeviceToHost);
	printf("cudaMemcpy(device -> host)y_group:%s\n",cudaGetErrorString(dyg));

	free(input_y);
	free(input_r);
	free(input_g);
	free(input_b);
	free(output_y);
	free(output_r);
	free(output_g);
	free(output_b);
	cudaFree(std_dy);
	cudaFree(std_dr);
	cudaFree(std_dg);
	cudaFree(std_db);
	cudaFree(input_dy);
	cudaFree(input_dr);
	cudaFree(input_dg);
	cudaFree(input_db);
	int temp=0;
	temp=getc(stdin);
	return 0;
}

void YUV2RGB(unsigned char * input_y, unsigned char * input_u, unsigned char * input_v, unsigned char * input_r, unsigned char * input_g, unsigned char * input_b)
{
	int w = 0, l = 0;
	int u = 0, v = 0;
	int y1, y2;
	int v1, v2, u1, u2;			
	for(w=0;w<height;w+=2)
	{
		for(l=0;l<width;l+=2)
		{
			v = input_v[(w/2)*width/2+(l/2)] - 128;
			u = input_u[(w/2)*width/2+(l/2)] - 128;

			v1 = ((v << 10) + (v << 9) + (v << 6) + (v << 5)) >> 10;	// 1.593
			u1 = ((u << 8) + (u << 7) + (u << 4)) >> 10;				// 0.390
			v2 = ((v << 9) + (v << 4)) >> 10;							// 0.515
			u2 = ((u << 11) + (u << 4)) >> 10;							// 2.015

			y1 = input_y[w*width+l];			

			input_r[w*width+l] = CLIP ( y1 + (v1) );
			input_g[w*width+l] = CLIP ( y1 - (u1) - (v2) );
			input_b[w*width+l] = CLIP ( y1 + (u2) );	
			
			y1 = input_y[w*width+(l+1)];

			input_r[w*width+(l+1)] = CLIP ( y1 + (v1) );
			input_g[w*width+(l+1)] = CLIP ( y1 - (u1) - (v2) );
			input_b[w*width+(l+1)] = CLIP ( y1 + (u2) );			

			y2 = input_y[(w+1)*width+l];	

			input_r[(w+1)*width+l] = CLIP ( y2 + (v1) );
			input_g[(w+1)*width+l] = CLIP ( y2 - (u1) - (v2) );
			input_b[(w+1)*width+l] = CLIP ( y2 + (u2) );	
			
			y2 = input_y[(w+1)*width+(l+1)];
		
			input_r[(w+1)*width+(l+1)] = CLIP ( y2 + (v1) );
			input_g[(w+1)*width+(l+1)] = CLIP ( y2 - (u1) - (v2) );
			input_b[(w+1)*width+(l+1)] = CLIP ( y2 + (u2) );	
		}	
	}	
}

/*for (int w = group_w_start; w < group_w_end; w++)
	{//std::cout << "----------" << std::endl;
		for (int l = group_l_start; l < group_l_end; l++)
		{
			std_y[l + (w)*width] = 0;
			std_r[l + (w)*width] = 0;
			std_g[l + (w)*width] = 0;
			std_b[l + (w)*width] = 0;
			
			for (int j = 0; j < 3; j++)
			{
				for (int i = 0; i < 3; i++)
				{
					avg_y[l + (w)*width] += input_y[(l + i) + (w + j)*width];
					avg_r[l + (w)*width] += input_r[(l + i) + (w + j)*width];
					avg_g[l + (w)*width] += input_g[(l + i) + (w + j)*width];
					avg_b[l + (w)*width] += input_b[(l + i) + (w + j)*width];
				}
			}

			avg_y[l + (w)*width] = avg_y[l + (w)*width] / (3 * 3);
			avg_r[l + (w)*width] = avg_r[l + (w)*width] / (3 * 3);
			avg_g[l + (w)*width] = avg_g[l + (w)*width] / (3 * 3);
			avg_b[l + (w)*width] = avg_b[l + (w)*width] / (3 * 3);

			for (int j = 0; j < 3; j++)
			{
				for (int i = 0; i < 3; i++)
				{
					std_y[l + (w)*width] += (avg_y[l + (w)*width] - input_y[(l + i) + (w + j)*width])*(avg_y[l + (w)*width] - input_y[(l + i) + (w + j)*width]);
					std_r[l + (w)*width] += (avg_r[l + (w)*width] - input_r[(l + i) + (w + j)*width])*(avg_r[l + (w)*width] - input_r[(l + i) + (w + j)*width]);
					std_g[l + (w)*width] += (avg_g[l + (w)*width] - input_g[(l + i) + (w + j)*width])*(avg_g[l + (w)*width] - input_g[(l + i) + (w + j)*width]);
					std_b[l + (w)*width] += (avg_b[l + (w)*width] - input_b[(l + i) + (w + j)*width])*(avg_b[l + (w)*width] - input_b[(l + i) + (w + j)*width]);
				}
			}

			std_y[(l)+(w)*width] = sqrt(std_y[l + (w)*width] / (3 * 3));
			std_r[(l)+(w)*width] = sqrt(std_r[l + (w)*width] / (3 * 3));
			std_g[(l)+(w)*width] = sqrt(std_g[l + (w)*width] / (3 * 3));
			std_b[(l)+(w)*width] = sqrt(std_b[l + (w)*width] / (3 * 3));

			if (std_y[(l)+(w)*width]<stdy_thre&& std_r[(l)+(w)*width]<stdr_thre && std_g[(l)+(w)*width]<stdg_thre && std_b[(l)+(w)*width]<stdb_thre)
			{
				output_y[(l)+(w)*width] = 0;
				y_group[l + w*width] = 0;
			}
			else
			{
				output_y[(l)+(w)*width] = 255;
				y_group[l + w*width] = 1;
			}
		}
	}*/