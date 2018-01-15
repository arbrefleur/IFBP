////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "para.h"
// includes CUDA
#include <cuda_runtime.h>

// includes, project
//#include <helper_cuda.h>
//#include <helper_functions.h> // helper functions for SDK examples

float *d_gg=NULL;
__constant__ float gC_angle_sin[frameN];
__constant__ float gC_angle_cos[frameN];

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { getchar(); exit(code); }
	}
}
__global__ void gpu_fdk(float *d_gg,float *d_outatemp,int width,int height,int bottom,int k)
{
	const int i = blockDim.x*blockIdx.x + threadIdx.x;

	const int j = blockDim.y*blockIdx.y+threadIdx.y;
	//int k=blockDim.z*blockIdx.z+threadIdx.z;
	if((i<width)&&(j<height)){
		int m;

		float dlta=2*pi/frameN;
		float x=(float)(i-width/2);
		float y=(float)(j-height/2);		
		int	 z=k+bottom;
		float temp=0.0f;
			for( m =0;m<frameN;m++)
			{
				const float cos_theta = gC_angle_cos[m];
				const float sin_theta = gC_angle_sin[m];
				double tempt00 = x*sin_theta*sinf(Angle*pi/180)- y*cos_theta*sinf(Angle*pi/180)+z*cosf(Angle*pi/180)+ODD-DIS;
				float prjx= -DIS*(x*cos_theta + y*sin_theta)/tempt00 + DetectX/2+ CX;
				float prjy=-DIS*(-x*sin_theta*cosf(Angle*pi/180)+y*cos_theta*cosf(Angle*pi/180)+z*sinf(Angle*pi/180))/tempt00 + DetectZ/2+ CZ;
				int xd = (int)prjx;
				float xf = prjx -xd;
				int yd = (int) prjy;
				float yf = prjy -yd;
				if((0<xd)&&(xd<DetectX-1) &&( yd>0)&& yd<(DetectZ-1))
				{
					float szd = (1-xf)**(d_gg+m*DetectX*DetectZ+yd*DetectX+xd) + xf**(d_gg+m*DetectX*DetectZ+yd*DetectX+xd+1);
					float szd1 = (1-xf)**(d_gg+m*DetectX*DetectZ+(yd+1)*DetectX+xd) + xf**(d_gg+m*DetectX*DetectZ+(yd+1)*DetectX+xd+1);
					float z0x0y0 = (1-yf)*szd + yf*szd1;
					 temp += z0x0y0*dlta;
				}
				*(d_outatemp+k*height*width+i*width+j)=temp;//	a[i][j] = a[i][j] + Xs*dltaBeta;
			}
		
		
		//if(*(d_outatemp+k*height*width+i*width+j)<1e-6)*(d_outatemp+k*height*width+i*width+j)=0;
	}
}
__global__ void gpu_projection()
{

}
extern "C"
void gpu_fdkbackprj(float *prj,float *img,int height,int width,int imageW,int imageH,int bottom,int top)
{
	int k;
	unsigned int i;
	const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
	const dim3 gridSize((imageW+blockSize.x-1) / blockSize.x, (imageH+blockSize.y-1) / blockSize.y,1);
	float *d_outimg=NULL;
	
	gpuErrchk(cudaMalloc((void**)&d_outimg,imageH*imageW*(top-bottom)*sizeof(float)));
	gpuErrchk(cudaMemset((void*)d_outimg,0,imageH*imageW*(top-bottom)))
	gpuErrchk(cudaMemcpy(d_gg,prj,height*width*frameN*sizeof(float),cudaMemcpyHostToDevice));
	float* angle_sin = new float[frameN];
	float* angle_cos = new float[frameN];
	float* angles = new float[frameN];
	float fai_sin;float fai_cos;
	fai_sin=sinf(Angle*pi/180);
	fai_cos=cosf(Angle*pi/180);
	for ( i = 0; i < frameN; ++i)
		angles[i] = (frameN-1-i)*pi/180;
	for ( i = 0; i < frameN; ++i) {
		angle_sin[i] = sinf(angles[i]);
		angle_cos[i] = cosf(angles[i]);
	}
	gpuErrchk(cudaMemcpyToSymbol(gC_angle_sin, angle_sin, frameN*sizeof(float), 0, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMemcpyToSymbol(gC_angle_cos, angle_cos, frameN*sizeof(float), 0, cudaMemcpyHostToDevice));
	
for(k=0;k<top-bottom;k++)	
	{
		
		gpu_fdk<<<gridSize, blockSize>>>(d_gg,d_outimg,imageW,imageH,bottom,k);
		gpuErrchk(cudaPeekAtLastError());
		cudaDeviceSynchronize();
	}
	gpuErrchk(cudaMemcpy(img,d_outimg,imageW*imageH*(top-bottom)*sizeof(float),cudaMemcpyDeviceToHost));
}
////////////////////////////////////////////////////////////////////////////////
extern "C"
void gpu_molloc(int width,int height,int frames)
{	
	gpuErrchk(cudaMalloc((void**)&d_gg,width*height*frames*sizeof(float)));
}
extern "C"
void gpu_release()
{
	gpuErrchk(cudaFree(d_gg));
}