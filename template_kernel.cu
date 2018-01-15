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
//texture<float, 3, cudaReadModeElementType> gg_tex;
typedef texture<float, 3, cudaReadModeElementType> texture3D;
static texture3D img_tex;
//float *d_gg=NULL;
__constant__ float gC_angle_sin[frameN];
__constant__ float gC_angle_cos[frameN];
__constant__ float gc_WDO[frameN];

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
__device__ void sortdifferent(long* out, double* X,double* Y,double* Z,int Z0, int Z1)
{
	int i,j,k;
	i =0;j=0;
	for( k =0;i<Z0 || j<Z1;k++)
	{

		if(i<Z0 && j< Z1)
		{
			if(fabs(X[i] - Y[j])<1e-6 )
			{ 
				Z[k] = X[i++];
				j++; 
			}
			else if( X[i] < Y[j]) 
			{
				Z[k] = X[i++];
			}
			else
			{
				Z[k] = Y[j++];
			}

		}
		else{
			if(i<Z0) Z[k] = X[i++];
			if(j<Z1) Z[k] = Y[j++];

		}


	}
	*out=k;

}
__device__ void get_prjindex(float cos_theta,float sin_theta,int x,int y,int z,int* xmin,int* xmax,int* ymin,int* ymax)
{
	int i;
	float u,v,w,minxtmp,maxxtmp,minytmp,maxytmp,prjx[8],prjy[8];
	u=x*cos_theta-y*sin_theta;
	v=cosf(alpha*pi/180)*(y*cos_theta+x*sin_theta)-z*sinf(alpha*pi/180);
	w=z*cosf(alpha*pi/180)+(y*cos_theta+x*sin_theta)*sinf(alpha*pi/180);			
	float temp00=(ODD-v)/(-FOD-v);
	minxtmp=maxxtmp=prjx[0]=temp00*(-u)+u+DetectX/2+ CX;
	minytmp=maxytmp=prjy[0]=-(temp00*(-w)+w)+DetectZ/2+ CZ;
	u=(x+1)*cos_theta-y*sin_theta;
	v=cosf(alpha*pi/180)*(y*cos_theta+(x+1)*sin_theta)-z*sinf(alpha*pi/180);
	w=z*cosf(alpha*pi/180)+(y*cos_theta+(x+1)*sin_theta)*sinf(alpha*pi/180);			
 temp00=(ODD-v)/(-FOD-v);
	prjx[1]=temp00*(-u)+u+DetectX/2+ CX;
	prjy[1]=-(temp00*(-w)+w)+DetectZ/2+ CZ;
	u=x*cos_theta-(y+1)*sin_theta;
	v=cosf(alpha*pi/180)*((y+1)*cos_theta+x*sin_theta)-z*sinf(alpha*pi/180);
	w=z*cosf(alpha*pi/180)+((y+1)*cos_theta+x*sin_theta)*sinf(alpha*pi/180);			
 temp00=(ODD-v)/(-FOD-v);
	prjx[2]=temp00*(-u)+u+DetectX/2+ CX;
	prjy[2]=-(temp00*(-w)+w)+DetectZ/2+ CZ;
	u=(x+1)*cos_theta-(y+1)*sin_theta;
	v=cosf(alpha*pi/180)*((y+1)*cos_theta+(x+1)*sin_theta)-z*sinf(alpha*pi/180);
	w=z*cosf(alpha*pi/180)+((y+1)*cos_theta+(x+1)*sin_theta)*sinf(alpha*pi/180);			
 temp00=(ODD-v)/(-FOD-v);
	prjx[3]=temp00*(-u)+u+DetectX/2+ CX;
	prjy[3]=-(temp00*(-w)+w)+DetectZ/2+ CZ;
	u=(x+1)*cos_theta-(y+1)*sin_theta;
	v=cosf(alpha*pi/180)*((y+1)*cos_theta+(x+1)*sin_theta)-(z+1)*sinf(alpha*pi/180);
	w=(z+1)*cosf(alpha*pi/180)+((y+1)*cos_theta+(x+1)*sin_theta)*sinf(alpha*pi/180);			
 temp00=(ODD-v)/(-FOD-v);
	prjx[4]=temp00*(-u)+u+DetectX/2+ CX;
	prjy[4]=-(temp00*(-w)+w)+DetectZ/2+ CZ;
	u=(x)*cos_theta-(y)*sin_theta;
	v=cosf(alpha*pi/180)*((y)*cos_theta+(x)*sin_theta)-(z+1)*sinf(alpha*pi/180);
	w=(z+1)*cosf(alpha*pi/180)+((y)*cos_theta+(x)*sin_theta)*sinf(alpha*pi/180);			
 temp00=(ODD-v)/(-FOD-v);
	prjx[5]=temp00*(-u)+u+DetectX/2+ CX;
	prjy[5]=-(temp00*(-w)+w)+DetectZ/2+ CZ;
	u=(x+1)*cos_theta-(y)*sin_theta;
	v=cosf(alpha*pi/180)*((y)*cos_theta+(x+1)*sin_theta)-(z+1)*sinf(alpha*pi/180);
	w=(z+1)*cosf(alpha*pi/180)+((y)*cos_theta+(x+1)*sin_theta)*sinf(alpha*pi/180);			
 temp00=(ODD-v)/(-FOD-v);
	prjx[6]=temp00*(-u)+u+DetectX/2+ CX;
	prjy[6]=-(temp00*(-w)+w)+DetectZ/2+ CZ;
	u=(x)*cos_theta-(y+1)*sin_theta;
	v=cosf(alpha*pi/180)*((y+1)*cos_theta+(x)*sin_theta)-(z+1)*sinf(alpha*pi/180);
	w=(z+1)*cosf(alpha*pi/180)+((y+1)*cos_theta+(x)*sin_theta)*sinf(alpha*pi/180);			
 temp00=(ODD-v)/(-FOD-v);
	prjx[7]=temp00*(-u)+u+DetectX/2+ CX;
	prjy[7]=-(temp00*(-w)+w)+DetectZ/2+ CZ;
	for(i=1;i<7;i++)
	{
		if(prjx[i]<minxtmp)minxtmp=prjx[i];
		if(prjy[i]<minytmp)minytmp=prjy[i];
		if(prjx[i]>maxxtmp)maxxtmp=prjx[i];
		if(prjy[i]>maxxtmp)maxxtmp=prjy[i];
	}
	*xmin=(int)minxtmp;*xmax=(int)maxxtmp;*ymin=(int)minytmp;*ymax=(int)maxytmp;
}
__device__ void get_lenth(float cos_theta,float sin_theta,int xd,int yd,int x,int y,int z,float* len)
{
		float s1=-FOD*cosf(alpha*pi/180)*sin_theta;
		float s2=-FOD*cosf(alpha*pi/180)*cos_theta;
		float s3=FOD*sin(alpha*pi/180);
		float d1=(xd-DetectX/2-CX)*cos_theta+sin_theta*(ODD*cosf(alpha*pi/180)+(DetectZ/2-yd)*sinf(alpha*pi/180));
		float d2=-(xd-DetectX/2-CX)*sin_theta+(ODD*cosf(alpha*pi/180)+(DetectZ/2-yd)*sin(alpha*pi/180))*cos_theta;
		float d3=-ODD*sin(alpha*pi/180)+(DetectZ/2-yd)*cosf(alpha*pi/180);
		int aflag=0;int bflag=0;
		
		float xa,ya,za,xb,yb,zb,xtemp,ytemp,ztemp;
		xtemp=(z+1-s3)*(d1-s1)/(d3-s3)+s1;ytemp=(z+1-s3)*(d2-s2)/(d3-s3)+s2;
		if(xtemp<=x+1&&xtemp>=x&&ytemp>=y&&ytemp<=y+1&&(!aflag||!bflag))
		{
			xa=xtemp;ya=ytemp;za=z+1;aflag=1;
		}
		xtemp=(z-s3)*(d1-s1)/(d3-s3)+s1;ytemp=(z-s3)*(d2-s2)/(d3-s3)+s2;
		if(xtemp<=x+1&&xtemp>=x&&ytemp>=y&&ytemp<=y+1&&(!aflag||!bflag))
		{
			if(aflag){xb=xtemp;yb=ytemp;zb=z;bflag=1;}else {xa=xtemp;ya=ytemp;za=z;aflag=1;}
		}
		ytemp=(x-s1)*(d2-s2)/(d1-s1)+s2;ztemp=(x-s1)*(d3-s3)/(d1-s1)+s3;
		if(ztemp<=z+1&&ztemp>=z&&ytemp>=y&&ytemp<=y+1&&(!aflag||!bflag))
		{
			if(aflag){xb=x;yb=ytemp;zb=ztemp;bflag=1;}else {xa=x;ya=ytemp;za=ztemp;aflag=1;}
		}
		ytemp=(x+1-s1)*(d2-s2)/(d1-s1)+s2;ztemp=(x+1-s1)*(d3-s3)/(d1-s1)+s3;
		if(ztemp<=z+1&&ztemp>=z&&ytemp>=y&&ytemp<=y+1&&(!aflag||!bflag))
		{
			if(aflag){xb=x+1;yb=ytemp;zb=ztemp;bflag=1;}else {xa=x+1;ya=ytemp;za=ztemp;aflag=1;}
		}
		xtemp=(y-s2)*(d1-s1)/(d2-s2)+s1;ztemp=(y-s2)*(d3-s3)/(d2-s2)+s3;
		if(ztemp<=z+1&&ztemp>=z&&xtemp<=x+1&&xtemp>=x&&(!aflag||!bflag))
		{
			if(aflag){xb=xtemp;yb=y;zb=ztemp;bflag=1;}else {xa=xtemp;ya=y;za=ztemp;aflag=1;}
		}
		xtemp=(y+1-s2)*(d1-s1)/(d2-s2)+s1;ztemp=(y+1-s2)*(d3-s3)/(d2-s2)+s3;
		if(ztemp<=z+1&&ztemp>=z&&xtemp<=x+1&&xtemp>=x&&(!aflag||!bflag))
		{
			if(aflag){xb=xtemp;yb=y+1;zb=ztemp;bflag=1;}else {xa=xtemp;ya=y+1;za=ztemp;aflag=1;}
		}
		if(aflag&&bflag)
		{
			*len=sqrt((xa-xb)*(xa-xb)+(ya-yb)*(ya-yb)+(za-zb)*(za-zb));
		}else *len=0.0f;
}
__global__ void gpu_backProjection(float *d_gg,float *d_outatemp,const int width,const int height,int bottom,int top)
{
	const int i = blockDim.x*blockIdx.x + threadIdx.x;
	const int j = blockDim.y*blockIdx.y+threadIdx.y;
	int xmin,xmax,ymin,ymax,xd,yd;
	//int k=blockDim.z*blockIdx.z+threadIdx.z;
	//float max_r=0.0f;
	//max_r=(float)(i-width/2)*(i-width/2)/((width/2)*(width/2))+(float)(j-height/2)*(j-height/2)/((height/2)*(height/2));
	if((i<width)&&(j<height)){
	//if(max_r<1.0){
	
		int k,m,z;

		float x=(float)(i-width/2);
		float y=(float)(-j+height/2);
		float u,v,w,prjx[8],prjy[8];
		for(k=0;k<top-bottom;k++)
		{
			z=k+bottom;	
			float temp=0.0f,len=0.0f;
			for(m=0;m<2;m++)
			{
				const float cos_theta =gC_angle_cos[m];// gC_angle_cos[frameN-1-m];
				const float sin_theta =gC_angle_sin[m];// gC_angle_sin[frameN-1-m];				
				get_prjindex(cos_theta,sin_theta,x,y,z,&xmin,&xmax,&ymin,&ymax);
				 for(yd=ymin;yd<ymax;yd++)
					 for(xd=xmin;xd<xmax;xd++)
					 {
						 get_lenth(cos_theta,sin_theta,xd,yd,x,y,z,&len);
						 if(len>0)
						 {
							 temp+=len**(d_gg+m*DetectX*DetectZ+yd*DetectX+xd);
						 }
					 }
			
			}
			//if(temp<1e-6)
				//temp=0;
			*(d_outatemp+k*height*width+j*width+i)=temp;
		
		}
		//if(*(d_outatemp+k*height*width+j*width+i)<1e-6)*(d_outatemp+k*height*width+j*width+i)=0;
	}
}
__global__ void get_projection(float* img,float* prj,double* z0,double* zx,double* zy,double* zz,long* J,double* L)
{	
	const int k = blockDim.x*blockIdx.x + threadIdx.x;
	const int n = blockDim.y*blockIdx.y+threadIdx.y;
	float d1,d2;
	d1 = n -DetectZ/2;
	d2 = k - DetectX/2;


	long int i,j;
	double x1,y1,z1,x2,y2,z2;
	double xs,ys,zs,xd,yd,zd;

	/*double y= n-imageY/2;
	double x = k-imageX/2;
	x1 = FOD*sintable[m]*sinfai+imageX/2;
	y1 =-FOD*costable[m]*sinfai+imageY/2;
	z1 =FOD*cosfai-zbot;
	x2=x*costable[m]-y*sintable[m]*cosfai-ODD*sintable[m]*sinfai+imageX/2;
	y2=x*sintable[m]+y*costable[m]*cosfai+ODD*costable[m]*sinfai+imageY/2;
	z2=y*sinfai-ODD*cosfai-zbot;
	*/
	
	int m;
	if(k<DetectX&&n<DetectZ)
		for( m =0;m<frameN;m++)
		{
			const float cos_theta = gC_angle_cos[m];
			const float sin_theta = gC_angle_sin[m];
			double y= n;
			double x = k;
			x1 = -FOD*cosf(alpha*pi/180)*sin_theta+imageX/2;
			y1 =-FOD*cosf(alpha*pi/180)*cos_theta+imageY/2;
			z1 =FOD*sin(alpha*pi/180);
			x2=(x-DetectX/2-CX)*cos_theta+sin_theta*(ODD*cosf(alpha*pi/180)+(DetectZ/2-y)*sinf(alpha*pi/180))+imageX/2;
			y2=-(x-DetectX/2-CX)*sin_theta+(ODD*cosf(alpha*pi/180)+(DetectZ/2-y)*sin(alpha*pi/180))*cos_theta+imageY/2;
			z2=-ODD*sin(alpha*pi/180)+(DetectZ/2-y)*cosf(alpha*pi/180)-zbot;
		
	
	
	double kyx,kzx,kxy,kzy,kxz,kyz;
	double byx,bzx,bxy,bzy,bxz,byz;
	double zxN,zyN,yxN,yzN,xyN,xzN;
	double xmin,xmax,ymin,ymax,zmin,zmax;

	long int c1=0,c2=0,c3=0;
	for(i = 0;i<=3*imageN;i++)
	{
		z0[i] = 0;
		zx[i] = 0;
		zy[i] = 0;
		zz[i] = 0;
		J[i] = 0;
		L[i] = 0;
	}


	// x1 != x2  y1 != y2   z1 != z2 
	if( fabs(x1-x2) > 1e-6  && fabs(y1-y2)>1e-6 &&  fabs(z1-z2)>1e-6  )
	{

		kzx = (z2-z1)/(x2-x1);  bzx=z1-kzx*x1; zxN = kzx*imageX + bzx;
		kzy = (z2-z1)/(y2-y1);  bzy=z1-kzy*y1; zyN = kzy*imageY + bzy;
		kyx = (y2-y1)/(x2-x1);  byx=y1-kyx*x1; yxN = kyx*imageX + byx;
		kyz = (y2-y1)/(z2-z1);  byz=y1-kyz*z1; yzN = kyz*(ztop-zbot) + byz;
		kxy = (x2-x1)/(y2-y1);  bxy=x1-kxy*y1; xyN = kxy*imageY + bxy;
		kxz = (x2-x1)/(z2-z1);  bxz=x1-kxz*z1; xzN = kxz*(ztop-zbot) + bxz;

		if(kzx>0) { zmax = (ztop-zbot)<zxN ? (ztop-zbot):zxN; zmin = 0>bzx ? 0:bzx;	}
		else      { zmax = (ztop-zbot)<bzx ? (ztop-zbot):bzx; zmin = 0>zxN ? 0:zxN;	}


		if(zmin >= ztop-zbot || zmax <=0 )
			c1 = 0;
		else	
		{
			if(kzy>0) { zmax = zmax<zyN ? zmax:zyN;  zmin = zmin>bzy ? zmin:bzy;	}
			else      { zmax = zmax<bzy ? zmax:bzy;  zmin = zmin>zyN ? zmin:zyN;	}

			if(kyx>0) { ymax = imageY<yxN ? imageY:yxN; ymin = 0>byx ? 0:byx;	}
			else      { ymax = imageY<byx ? imageY:byx; ymin = 0>yxN ? 0:yxN;	}
			if(kyz>0) { ymax = ymax<yzN ? ymax:yzN;  ymin = ymin>byz ? ymin:byz;	}
			else      { ymax = ymax<byz ? ymax:byz;  ymin = ymin>yzN ? ymin:yzN;	}

			if(kxy>0) { xmax = imageX<xyN ? imageX:xyN; xmin = 0>bxy ? 0:bxy;	}
			else      { xmax = imageX<bxy ? imageX:bxy; xmin = 0>xyN ? 0:xyN;	}
			if(kxz>0) { xmax = xmax<xzN ? xmax:xzN;  xmin = xmin>bxz ? xmin:bxz;	}
			else      { xmax = xmax<bxz ? xmax:bxz;  xmin = xmin>xzN ? xmin:xzN;	}

			for(j=0,i=ceil(zmin);i<=(int)(zmax);i++)
			{ z0[j++] = i; 
			if(z0[j-1]>ztop-zbot) z0[j-1] = ztop-zbot;
			}
			c1 = j;

			if(kzy>0)
			{
				for(j=0,i=ceil(ymin);i<=floor(ymax);i++)
				{ zy[j++] = kzy*i + bzy;
				if(zy[j-1]>ztop-zbot) zy[j-1] = ztop-zbot; 
				}
				c2 =j;
			}
			else
			{for(j=0,i=floor(ymax);i>=ceil(ymin);i--)
			{ zy[j++] = kzy*i + bzy; 
			if(zy[j-1]>ztop-zbot) zy[j-1] = ztop-zbot;
			if(zy[j-1]<0) zy[j-1] = 0;
			}
			c2 =j;
			}

			if(kzx>0)
			{for(j=0,i=ceil(xmin);i<=floor(xmax);i++)
			{ zx[j++] = kzx*i + bzx; 
			if(zx[j-1]>ztop-zbot) zx[j-1] = ztop-zbot;
			if(zx[j-1]<0) zx[j-1] = 0;
			}
			c3 =j;
			}
			else
			{for(j=0,i=floor(xmax);i>=ceil(xmin);i--)
			{ zx[j++] = kzx*i + bzx; 
			if(zx[j-1]>ztop-zbot) zx[j-1] = ztop-zbot;
			if(zx[j-1]<0) zx[j-1] = 0;
			}
			c3 =j;
			}

			//	for(i=0;i<c2;i++)
			//		printf("与y轴共有个%d交点，交点zy[%d]=%.6f\n",c2,i,zy[i]);
			//		for(i=0;i<c3;i++)
			// 		printf("与x轴共有个%d交点，交点zx[%d]=%.6f\n",c3,i,zx[i]);
			sortdifferent(&c1,z0,zy,zz,c1,c2);
			sortdifferent(&c1,zz,zx,z0,c1,c3);
			//	for(i=0;i<c1;i++)
			//	printf("与物体共有个%d交点，交点z0[%d]=%.6f\n",c1,i,z0[i]);
			for(i=0;i<c1-1;i++)
			{
				J[i] = (int)(kxz*(z0[i+1]+z0[i])/2 + bxz) + ((int)(kyz*(z0[i+1]+z0[i])/2 + byz))*imageX +((int)((z0[i+1]+z0[i])/2))*imageX*imageY;
				L[i] = fabs(z0[i+1]-z0[i])*sqrt(1+kxz*kxz +kyz*kyz);
			}
			float bmt;
			bmt = 0;
		}
	}

	// x1 = x2 ,y1 !=y2, z1 !=z2
	else if( fabs(x1-x2) <= 1e-6  && fabs(y1-y2)>1e-6 &&  fabs(z1-z2)>1e-6  )
	{
		if(x1 >0 &&x1 <imageX)
		{

			kzy = (z2-z1)/(y2-y1);  bzy=z1-kzy*y1; zyN = kzy*imageY + bzy;
			kyz = (y2-y1)/(z2-z1);  byz=y1-kyz*z1; yzN = kyz*(ztop-zbot) + byz;

			if(kzy>0) { zmax = (ztop-zbot)<zyN ? (ztop-zbot):zyN;  zmin = 0>bzy ? 0:bzy;	}
			else      { zmax = (ztop-zbot)<bzy ? (ztop-zbot):bzy;  zmin =0>zyN ? 0:zyN;	}
			if(zmin >= ztop-zbot || zmax <=0 )
				c1 = 0;
			else{
				if(kyz>0) { ymax = imageY<yzN ? imageY:yzN;  ymin = 0>byz ? 0:byz;	}
				else      { ymax = imageY<byz ? imageY:byz;  ymin = 0>yzN ? 0:yzN;	}

				for(j=0,i=ceil(zmin);i<=floor(zmax);i++)
				{ z0[j++] = i; 
				if(z0[j-1]>ztop-zbot) z0[j-1] = ztop-zbot;
				}
				c1 = j;

				if(kyz>0)
				{
					for(j=0,i=ceil(ymin);i<=floor(ymax);i++)
					{ zy[j++] = kzy*i + bzy;
					if(zy[j-1]>ztop-zbot) zy[j-1] = ztop-zbot; 
					}
					c2 =j;
				}
				else
				{for(j=0,i=floor(ymax);i>=ceil(ymin);i--)
				{ zy[j++] = kzy*i + bzy; 
				if(zy[j-1]>ztop-zbot) zy[j-1] = ztop-zbot;
				}
				c2 =j;
				}


			    sortdifferent(&c1,z0,zy,zz,c1,c2);
				for(i=0;i<c1;i++)
					z0[i] = zz[i];
				for(i=0;i<c1-1;i++)
				{
					J[i] = (int)((x1+x2)/2) + ((int)(kyz*(z0[i+1]+z0[i])/2 + byz))*imageX +((int)((z0[i+1]+z0[i])/2))*imageX*imageY;
					L[i] = fabs(z0[i+1]-z0[i])*sqrt(1 +kyz*kyz);
				}
			}

		}	else
			c1 =0;

	}


	// x1 != x2,y1 = y2,z1 !=z2 
	else if( fabs(x1-x2) > 1e-6  && fabs(y1-y2)<=1e-6 &&  fabs(z1-z2)>1e-6  )
	{
		if(y1>0 && y1<imageY){

			kzx = (z2-z1)/(x2-x1);  bzx=z1-kzx*x1; zxN = kzx*imageX + bzx;
			kxz = (x2-x1)/(z2-z1);  bxz=x1-kxz*z1; xzN = kxz*(ztop-zbot) + bxz;

			if(kzx>0) { zmax = (ztop-zbot)<zxN ? (ztop-zbot):zxN; zmin = 0>bzx ? 0:bzx;	}
			else      { zmax = (ztop-zbot)<bzx ? (ztop-zbot):bzx; zmin = 0>zxN ? 0:zxN;	}
			if(zmin >= ztop-zbot || zmax <=0 )
				c1 = 0;
			else
			{

				if(kxz>0) { xmax = imageX<xzN ? imageX:xzN;  xmin = 0>bxz ? 0:bxz;	}
				else      { xmax = imageX<bxz ? imageX:bxz;  xmin = 0>xzN ? 0:xzN;	}

				for(j=0,i=ceil(zmin);i<=floor(zmax);i++)
				{ z0[j++] = i; 
				if(z0[j-1]>ztop-zbot) z0[j-1] = ztop-zbot;
				}  
				c1 = j;


				if(kxz>0)
				{for(j=0,i=ceil(xmin);i<=floor(xmax);i++)
				{ zx[j++] = kzx*i + bzx; 
				if(zx[j-1]>ztop-zbot) zx[j-1] = ztop-zbot;
				}
				c3 =j;
				}
				else
				{for(j=0,i=floor(xmax);i>=ceil(xmin);i--)
				{ zx[j++] = kzx*i + bzx; 
				if(zx[j-1]>ztop-zbot) zx[j-1] = ztop-zbot;
				}
				c3 =j;
				}

				sortdifferent(&c1,z0,zx,zz,c1,c3);
				for(i=0;i<c1;i++)
					z0[i] = zz[i];

				for(i=0;i<c1-1;i++)
				{
					J[i] = (int)(kxz*(z0[i+1]+z0[i])/2 + bxz) + ((int)((y1+y2)/2))*imageX +((int)((z0[i+1]+z0[i])/2))*imageX*imageY;
					L[i] = fabs(z0[i+1]-z0[i])*sqrt(1+kxz*kxz);
				}
			}
		}
		else
			c1 = 0;
	}


	// x1 = x2  ,y1 = y2  z1 != z2
	else if( fabs(x1-x2) <= 1e-6  && fabs(y1-y2)<=1e-6 &&  fabs(z1-z2)>1e-6  )
	{
		if(x1 >0 && x1 <imageX && y1 >0 && y1 <imageY)
		{
			for(j=0,i =0;i<(ztop-zbot);i++) z0[j++] = i; c1 = j;
			for(i=0;i<c1-1;i++)
			{ J[i] = (int)((x1+x2)/2) + ((int)((y1+y2)/2))*imageX +((int)((z0[i+1]+z0[i])/2))*imageX*imageY;
			L[i] = 1;
			}}
		else
			c1 =0;

	}



	// x1 != x2, y1 != y2,z1 = z2
	else if( fabs(x1-x2) >1e-6  && fabs(y1-y2)>1e-6 &&  fabs(z1-z2)<=1e-6  )
	{
		if(z1 >0 && z1 <ztop-zbot){

			kyx = (y2-y1)/(x2-x1);  byx=y1-kyx*x1; yxN = kyx*imageX + byx;
			kxy = (x2-x1)/(y2-y1);  bxy=x1-kxy*y1; xyN = kxy*imageY + bxy;




			if(kyx>0) { ymax = imageY<yxN ? imageY:yxN; ymin = 0>byx ? 0:byx;	}
			else      { ymax = imageY<byx ? imageY:byx; ymin = 0>yxN ? 0:yxN;	}


			if(kxy>0) { xmax = imageX<xyN ? imageX:xyN; xmin = 0>bxy ? 0:bxy;	}
			else      { xmax = imageX<bxy ? imageX:bxy; xmin = 0>xyN ? 0:xyN;	}


			for(j=0,i=ceil(xmin);i<=floor(xmax);i++)
			{ zx[j++] = i; 
			//if(zx[j-1]>imageN) zx[j-1] = imageN;
			}
			c1 = j;

			if(kyx>0)
			{for(j=0,i=ceil(ymin);i<=floor(ymax);i++)
			{ zy[j++] = kxy*i + bxy; 
			//if(zy[j-1]>imageN) zy[j-1] = imageN;
			}
			c3 =j;
			}
			else
			{for(j=0,i=floor(ymax);i>=ceil(ymin);i--)
			{ zy[j++] = kxy*i + bxy; 
			//if(zy[j-1]>imageN) zy[j-1] = imageN;
			}
			c3 =j;
			}

			sortdifferent(&c1,zx,zy,z0,c1,c3);

			for(i=0;i<c1-1;i++)
			{
				J[i] = (int)((z0[i+1]+z0[i])/2 ) + ((int)(kyx*(z0[i+1]+z0[i])/2 + byx))*imageX +((int)((z1+z2)/2))*imageX*imageY;
				L[i] = fabs(z0[i+1]-z0[i])*sqrt(1+kyx*kyx);
			}
		}else
			c1 = 0;

	}


	//x1 = x2,y1! = y2,z1 = z2
	else if( fabs(x1-x2) <= 1e-6  && fabs(y1-y2)>1e-6 &&  fabs(z1-z2) <= 1e-6  )
	{	if(x1>0 && x1 <imageX && z1 >0 && z1 <ztop -zbot)
	{
		for(j=0,i =0;i<imageY;i++) z0[j++] = i; c1 =j;
		for(i=0;i<c1-1;i++)
		{ J[i] = (int)((x1+x2)/2) + ((int)((z0[i+1]+z0[i])/2))*imageX +((int)((z1+z2)/2))*imageX*imageY;
		L[i] = 1;
		}
	}
	else
		c1 = 0;
	}

	//x1 ! =x2 ,y1 =y2 ,z1 =z2;
	else if( fabs(x1-x2) > 1e-6 && fabs(y1-y2)<=1e-6 &&  fabs(z1-z2)<=1e-6  )
	{

		if(y1>0 && y1 <imageY && z1 >0 && z1 <ztop -zbot)
		{
			for(j=0,i =0;i<imageX;i++) z0[j++] = i; c1 =j;
			for(i=0;i<c1-1;i++)
			{ J[i] = (int)((z0[i+1]+z0[i])/2) + ((int)((y1+y2)/2))*imageY +((int)((z1+z2)/2))*imageX*imageY;
			L[i] = 1;
			}
		}
		else
			c1 =0;


	}




	if(c1 >1)
	{		
		double tempt1 =0,tempt2=0,tempt3=0;

		for( i =0;i<c1-1;i++)
		{ 
			tempt1 += L[i]*img[J[i]];
			tempt2 += L[i]*L[i];
		}


		if(tempt2<1e-6) 
		{//printf("tempt2 =%.6f\n",tempt2);
			//float ceshi;
			// scanf("%f",&ceshi);
		}else
		{
			//tempt3 = (lmta)*(gg[m][n][k] - tempt1)/tempt2;
		//	tempt3=*(prj1+m*DetectX*DetectZ+n*DetectX+k) - tempt1;
 			//if(tempt3>0)
		//		*(prj2+m*DetectX*DetectZ+n*DetectX+k) =tempt3;
//			for( i =0;i<c1-1;i++)
// 				a[J[i]] = a[J[i]]+tempt3*L[i];
				prj[m*DetectX*DetectZ+n*DetectX+k]=tempt1;
		}

	}
	}



}
__global__ void gpu_fdk(float *d_gg,float *d_outatemp,const int width,const int height,int bottom,int top)
{
	const int i = blockDim.x*blockIdx.x + threadIdx.x;

	const int j = blockDim.y*blockIdx.y+threadIdx.y;
	//int k=blockDim.z*blockIdx.z+threadIdx.z;
	float max_r=0.0f;
	max_r=(float)(i-width/2)*(i-width/2)/((width/2)*(width/2))+(float)(j-height/2)*(j-height/2)/((height/2)*(height/2));
	if((i<width)&&(j<height)){
	//if(max_r<1.0){
		int m;
		int k;
		int z;
		float dlta=2*pi/frameN;
		float x=(float)(i-width/2)-error_DX;
		float y=(float)(-j+height/2)-error_DY;
		for(k=0;k<top-bottom;k++)
		{
			z=k+bottom;
			float temp=0.0f;
			
			for( m =0;m<frameN;m++)
			{
				const float cos_theta =gC_angle_cos[m];// gC_angle_cos[frameN-1-m];
				const float sin_theta =gC_angle_sin[m];// gC_angle_sin[frameN-1-m];
				float u=x*cos_theta-y*sin_theta;
				float v=cosf(alpha*pi/180)*(y*cos_theta+x*sin_theta)-z*sinf(alpha*pi/180);
				float w=z*cosf(alpha*pi/180)+(y*cos_theta+x*sin_theta)*sinf(alpha*pi/180);
				float U2=(FOD+w)*(FOD+w)/(DIST0*DIST0);
				//float tempt00 = x*sin_theta*sinf(Angle*pi/180)- y*cos_theta*sinf(Angle*pi/180)+z*cosf(Angle*pi/180)+ODD-DIS;
				//float prjx= -DIS*(x*cos_theta + y*sin_theta)/tempt00 + DetectX/2+ CX;
				//float prjy=-DIS*(-x*sin_theta*cosf(Angle*pi/180)+y*cos_theta*cosf(Angle*pi/180)+z*sinf(Angle*pi/180))/tempt00 + DetectZ/2+ CZ;
				float temp00=(ODD-v)/(-FOD-v);
				float prjx=temp00*(-u)+u+DetectX/2+ CX;
				float prjy=-(temp00*(-w)+w)+DetectZ/2+ CZ;

				int xd = (int)prjx;
				float xf = prjx -xd;
				int yd = (int) prjy;
				float yf = prjy -yd;
				if((0<xd)&&(xd<DetectX-1) &&( yd>0)&& yd<(DetectZ-1))
				{
					float szd = (1-xf)**(d_gg+m*DetectX*DetectZ+yd*DetectX+xd) + xf**(d_gg+m*DetectX*DetectZ+yd*DetectX+xd+1);
					float szd1 = (1-xf)**(d_gg+m*DetectX*DetectZ+(yd+1)*DetectX+xd) + xf**(d_gg+m*DetectX*DetectZ+(yd+1)*DetectX+xd+1);
					float z0x0y0 = (1-yf)*szd + yf*szd1;
					temp +=z0x0y0/U2;
					// temp +=dlta* tex3D(gg_tex,prjx,prjy,m);
				}
				//	a[i][j] = a[i][j] + Xs*dltaBeta;
			}
			//if(temp<1e-6)
				//temp=0;
			*(d_outatemp+k*height*width+j*width+i)=dlta*temp;
		
		}
		//if(*(d_outatemp+k*height*width+j*width+i)<1e-6)*(d_outatemp+k*height*width+j*width+i)=0;
	}
}
__global__ void gpu_projection(float* d_output,int xmin,int xmax,int ymin,int ymax)
{
	const int idx = blockDim.x*blockIdx.x + threadIdx.x;
	const int idy = blockDim.y*blockIdx.y+threadIdx.y;
	int m;
	if(idx<DetectX&&idy<DetectZ)
		for( m =0;m<frameN;m++)
	{
		const float cos_theta = gC_angle_cos[m];
		const float sin_theta = gC_angle_sin[m];
		float s1=-FOD*cosf(alpha*pi/180)*sin_theta+error_SX;
		float s2=-FOD*cosf(alpha*pi/180)*cos_theta+error_SY;
		float s3=FOD*sin(alpha*pi/180);
		float d1=(idx-DetectX/2-CX)*cos_theta+sin_theta*(ODD*cosf(alpha*pi/180)+(DetectZ/2-idy)*sinf(alpha*pi/180))+error_DX;
		float d2=-(idx-DetectX/2-CX)*sin_theta+(ODD*cosf(alpha*pi/180)+(DetectZ/2-idy)*sin(alpha*pi/180))*cos_theta+error_DX;
		float d3=-ODD*sin(alpha*pi/180)+(DetectZ/2-idy)*cosf(alpha*pi/180);
		int aflag=0;int bflag=0;
	//x1 = SX*sintable[m]-FOD*cosf(fai*pi/180)*sintable[m]+SZ*sinf(fai*pi/180)*sintable[m]+imageX/2;
	//y1 =-SX*sintable[m]-FOD*cosf(fai*pi/180)*costable[m]+SZ*costable[m]*sinf(fai*pi/180)+imageY/2;
	//z1 =FOD*sin(fai*pi/180)+SZ*cosf(fai*pi/180)-zbot;///////-zbot
	//x2=(x-DetectX/2-CX)*costable[m]+sintable[m]*(ODD*cosf(fai*pi/180)+(DetectZ/2-y)*sinf(fai*pi/180))+imageX/2;
	//y2=-(x-DetectX/2-CX)*sintable[m]+(ODD*cosf(fai*pi/180)+(DetectZ/2-y)*sin(fai*pi/180))*costable[m]+imageY/2;
	//z2=-ODD*sin(fai*pi/180)+(DetectZ/2-y)*cosf(fai*pi/180)-zbot;
		float xa,ya,za,xb,yb,zb,xtemp,ytemp,ztemp;
		xtemp=(ztop-1-s3)*(d1-s1)/(d3-s3)+s1;ytemp=(ztop-1-s3)*(d2-s2)/(d3-s3)+s2;
		if(xtemp<=xmax&&xtemp>=xmin&&ytemp>=ymin&&ytemp<=ymax&&(!aflag||!bflag))
		{
			xa=xtemp;ya=ytemp;za=ztop-1;aflag=1;
		}
		xtemp=(zbot-s3)*(d1-s1)/(d3-s3)+s1;ytemp=(zbot-s3)*(d2-s2)/(d3-s3)+s2;
		if(xtemp<=xmax&&xtemp>=xmin&&ytemp>=ymin&&ytemp<=ymax&&(!aflag||!bflag))
		{
			if(aflag){xb=xtemp;yb=ytemp;zb=zbot;bflag=1;}else {xa=xtemp;ya=ytemp;za=zbot;aflag=1;}
		}
		ytemp=(xmin-s1)*(d2-s2)/(d1-s1)+s2;ztemp=(xmin-s1)*(d3-s3)/(d1-s1)+s3;
		if(ztemp<=ztop&&ztemp>=zbot&&ytemp>=ymin&&ytemp<=ymax&&(!aflag||!bflag))
		{
			if(aflag){xb=xmin;yb=ytemp;zb=ztemp;bflag=1;}else {xa=xmin;ya=ytemp;za=ztemp;aflag=1;}
		}
		ytemp=(xmax-s1)*(d2-s2)/(d1-s1)+s2;ztemp=(xmax-s1)*(d3-s3)/(d1-s1)+s3;
		if(ztemp<=ztop&&ztemp>=zbot&&ytemp>=ymin&&ytemp<=ymax&&(!aflag||!bflag))
		{
			if(aflag){xb=xmax;yb=ytemp;zb=ztemp;bflag=1;}else {xa=xmax;ya=ytemp;za=ztemp;aflag=1;}
		}
		xtemp=(ymin-s2)*(d1-s1)/(d2-s2)+s1;ztemp=(ymin-s2)*(d3-s3)/(d2-s2)+s3;
		if(ztemp<=ztop&&ztemp>=zbot&&xtemp<=xmax&&xtemp>=xmin<ymax&&(!aflag||!bflag))
		{
			if(aflag){xb=xtemp;yb=ymin;zb=ztemp;bflag=1;}else {xa=xtemp;ya=ymin;za=ztemp;aflag=1;}
		}
		xtemp=(ymax-s2)*(d1-s1)/(d2-s2)+s1;ztemp=(ymax-s2)*(d3-s3)/(d2-s2)+s3;
		if(ztemp<=ztop&&ztemp>=zbot&&xtemp<=xmax&&xtemp>=xmin&&(!aflag||!bflag))
		{
			if(aflag){xb=xtemp;yb=ymax;zb=ztemp;bflag=1;}else {xa=xtemp;ya=ymax;za=ztemp;aflag=1;}
		}
		if(aflag&&bflag)
		{
			float len=sqrt((xa-xb)*(xa-xb)+(ya-yb)*(ya-yb)+(za-zb)*(za-zb));
			float step=1;//步长
			float temp,x,y,z;
			float accumprj=0;
			float tx=fabs(xa-xb),ty=fabs(ya-yb),tz=fabs(za-zb);
			if(tx>=ty&&tx>=tz)
			{
				if(xa>xb){ temp=xa;xa=xb;xb=temp;temp=ya;ya=yb;yb=temp;temp=za;za=zb;zb=temp;}
				x=xa, y=ya, z=za;
				while(x<xb)
				//while((x-xa)*(xb-x)>=0)
				{
					accumprj +=tex3D(img_tex,x+imageX/2,-y+imageY/2,z-zbot);
					x+=(xb-xa)*step/len;y+=(yb-ya)*step/len;z+=(zb-za)*step/len;														
				}
			}
			else if(ty>=tx&&ty>=tz)
			{
				if(ya>yb){ temp=xa;xa=xb;xb=temp;temp=ya;ya=yb;yb=temp;temp=za;za=zb;zb=temp;}
				x=xa, y=ya, z=za;
				while(y<yb)
				//while((y-ya)*(yb-y)>=0)
				{
					accumprj +=tex3D(img_tex,x+imageX/2,-y+imageY/2,z-zbot);
					x+=(xb-xa)*step/len;y+=(yb-ya)*step/len;z+=(zb-za)*step/len;														
				}
			}
			else if(tz>=tx&&tz>=ty)
			{
				//if(za>zb){ temp=xa;xa=xb;xb=temp;temp=ya;ya=yb;yb=temp;temp=za;za=zb;zb=temp;}
				x=xa, y=ya, z=za;
				while(z<zb)
				//while((z-za)*(zb-z)>=0)
				{
					accumprj +=tex3D(img_tex,x+imageX/2,-y+imageY/2,z-zbot);
					x+=(xb-xa)*step/len;y+=(yb-ya)*step/len;z+=(zb-za)*step/len;														
				}
			}								
			if(accumprj!=0)
				d_output[DetectZ*DetectX*m+DetectX*idy+idx]=accumprj;
		}
	}
}
__global__ void gpu_Add(const float *in1,const float* in2,float* out, float scale,int size)
{
	const int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<size)
	{
		out[i]=in1[i]+scale*in2[i];
	}
}
__global__ void gpu_Getlamta(const float *in1,const float* in2,float* sum1,float* sum2,int size)
{
	extern __shared__ float shared[];
	
	const int tid= threadIdx.x;
	const int bid=blockIdx.x;
	int i;
	shared[tid]=0;
	shared[tid+MAX_THREADNUM_PERBLOCK]=0;
	for(i=bid*MAX_THREADNUM_PERBLOCK+tid;i<size;i+=BLOCKNUM*MAX_THREADNUM_PERBLOCK)
	{
		shared[tid] +=in1[i]*in1[i];
		shared[tid+MAX_THREADNUM_PERBLOCK] +=in1[i]*in2[i];
	}
	__syncthreads();
	int offset=1,mask=1;
	while(offset<MAX_THREADNUM_PERBLOCK)
	{
		if((tid&mask)==0)
		{
			shared[tid]+=shared[tid+offset];
			shared[tid+MAX_THREADNUM_PERBLOCK]+=shared[tid+offset+MAX_THREADNUM_PERBLOCK];
		}
		offset +=offset;
		mask +=offset;
		__syncthreads();
	}
	if(tid==0){
		sum1[bid]=shared[0];
		sum2[bid]=shared[MAX_THREADNUM_PERBLOCK];
	}
}
__global__ void gpu_filter(const float* hs,float* prj,int width)
{
	const int frame = blockDim.x*blockIdx.x + threadIdx.x;
	const int idy = blockDim.y*blockIdx.y+threadIdx.y;
	float ps[3*DetectX];
	int i,j;
	if(idy<DetectZ&&frame<frameN)
	{
		float tempt0,tempt1;
		float jiaquan=0.0f;
		jiaquan=DIS/sqrtf(DIS*DIS+(idy-DetectZ/2)*(idy-DetectZ/2)+(DetectX/2)*(DetectX/2));
			tempt0 = *(prj+frame*DetectZ*DetectX+idy*DetectX)*jiaquan;
			tempt1 = (*(prj+frame*DetectZ*DetectX+idy*DetectX+DetectX-1) + *(prj+frame*DetectZ*DetectX+idy*DetectX+DetectX-2))*jiaquan/2;
			//投影数据进行扩充
			for(i = 1;i<DetectX;i++) ps[i] = tempt0;
			for(i = DetectX;i<DetectX*2;i++){
				jiaquan=DIS/sqrtf(DIS*DIS+(idy-DetectZ/2)*(idy-DetectZ/2)+(i-3*DetectX/2)*(i-3*DetectX/2));
				ps[i] = *(prj+frame*DetectZ*DetectX+idy*DetectX+i-DetectX)*jiaquan;}
			for( i =2*DetectX;i<DetectX*3;i++) ps[i] =tempt1;
			//卷积滤波
			float sum ;
			for( i =0;i<DetectX;i++)
			{ 
				sum = 0;
				for( j=1;j<2*DetectX;j++ )
					sum += hs[j]*ps[i+j] ;
				*(prj+frame*DetectZ*DetectX+idy*DetectX+i) = sum/2;
			}
		
	}
}
extern "C"
void gpu_imgfilter(const float* prj,float* out,char  type[10])
{

	clock_t start,end;
	float time;
	start=clock();
	float *h,*hs;
	long i,m,l,j;
	h = new float[DetectX+1];
	hs = new float[DetectX*2];
	if(0==strcmp(type,"FH"))//FH
	{
		for( i =1;i<=DetectX;i++) 
			if(i%2==0)
				h[i]=i*cos(alpha*pi/180)/(4*pi*pi*(i*i-1));
			else
				h[i]=cos(alpha*pi/180)/(4*pi*pi*i);		
		for(i=1;i<2*DetectX;i++)
			hs[i]=h[abs(i-DetectX)+1];
	}
	
	else if(0==strcmp(type,"FL"))
	{
		
		for( i =1;i<=DetectX;i++) 
			if(i%2==0)
				h[i]=-1/(8*pi*pi*(i));
			else
				h[i]=1/(8*pi*pi*i);		
		for(i=1;i<2*DetectX;i++)
			hs[i]=h[abs(i-DetectX)+1];
	}else if(0==strcmp(type,"SL")) //if(strcmp(type,'SL'))//SL
	{
		for( i =0;i<DetectX;i++) 
			h[i] = -cos(alpha*pi/180)/(pi*pi*(4*i*i - 1));
	 	for( i =1;i<2*DetectX;i++)
	 		hs[i] = h[abs(i-DetectX)];
	}
	else if(0==strcmp(type,"RL-HN"))//RL-HN
	{
		for( i =1;i<DetectX;i++) 
			if(i%2==0)
				h[i]=(-1)*cos(alpha*pi/180)*(1/(4*pi*pi*(i+1)*(i+1))+1/(4*pi*pi*(i-1)*(i-1)));
			else
				h[i]=(-1)*cos(alpha*pi/180)/(4*pi*pi*i*i);		
		for(i=1;i<DetectX;i++)
			hs[i]=h[DetectX-i];
		for(i=DetectX+1;i<2*DetectX;i++)
			hs[i]=h[i-DetectX];
		hs[DetectX]=(1)*cos(alpha*pi/180)/(2*pi*pi);
	}
	else if(0==strcmp(type,"RL"))//RL
	{
		for( i =1;i<DetectX;i++) 
			if(i%2==0)
				h[i]=0;//(-1)*cos(alpha*pi/180)*(1/(4*pi*pi*(i+1)*(i+1))+1/(4*pi*pi*(i-1)*(i-1)));
			else
				h[i]=(-1)*cos(alpha*pi/180)/(pi*pi*i*i);		
		for(i=1;i<DetectX;i++)
			hs[i]=h[DetectX-i];
		for(i=DetectX+1;i<2*DetectX;i++)
			hs[i]=h[i-DetectX];
		hs[DetectX]=cos(alpha*pi/180)/4;
	}
	float* d_hs=NULL;
	float* d_prj=NULL;
	const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
	const dim3 gridSize((frameN+blockSize.x-1) / blockSize.x, (DetectZ+blockSize.y-1) / blockSize.y,1);
	gpuErrchk(cudaMalloc((void**)&d_hs,DetectX*2*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_prj,DetectX*DetectZ*frameN*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_prj,prj,DetectX*DetectZ*frameN*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_hs,hs,DetectX*2*sizeof(float),cudaMemcpyHostToDevice));
	gpu_filter<<<gridSize,blockSize>>>(d_hs,d_prj,DetectX);
	gpuErrchk(cudaMemcpy(out,d_prj,DetectZ*DetectX*frameN*sizeof(float),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_hs));
	gpuErrchk(cudaFree(d_prj));

	end=clock();
	time=(end-start)/CLK_TCK;
	printf("filter success!time=%.5f\n",time);
}
extern "C"
void theta_init()
{	
	float* angle_sin = new float[frameN];
	float* angle_cos = new float[frameN];
	float* angles = new float[frameN];
//	float fai_sin;float fai_cos;
//	fai_sin=sinf(Angle*pi/180);
//	fai_cos=cosf(Angle*pi/180);
	int i;
	for ( i = 0; i < frameN; ++i)
		if(z_dir)
			angles[i] =i*pi/180;
		else 
			angles[i] = (frameN-1-i)*pi/180;

	for ( i = 0; i < frameN; ++i) {
		angle_sin[i] = sinf(angles[i]);
		angle_cos[i] = cosf(angles[i]);
	}
	gpuErrchk(cudaMemcpyToSymbol(gC_angle_sin, angle_sin, frameN*sizeof(float), 0, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMemcpyToSymbol(gC_angle_cos, angle_cos, frameN*sizeof(float), 0, cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMalloc((void**)&d_gg,width*height*frames*sizeof(float)));
}
extern "C"
void gpu_imgAdd(float *in1,float* in2,float* out, float scale,int size)
{
	printf("Start Adding Image...\n");
	clock_t start,end;
	float time;
	start=clock();
	float* d_output=NULL;
	float* d_in1=NULL;
	float* d_in2=NULL;
	const dim3 blockSize(MAX_THREADNUM_PERBLOCK, 1,1);
	const dim3 gridSize((size+blockSize.x-1) / blockSize.x,1, 1);
	gpuErrchk(cudaMalloc((void**)&d_in1,size*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_in2,size*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_in1,in1,size*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_in2,in2,size*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&d_output,size*sizeof(float)));
	gpu_Add<<<gridSize,blockSize>>>(d_in1,d_in2,d_output,scale,size);
	gpuErrchk(cudaMemcpy(out,d_output,size*sizeof(float),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_in1));
	gpuErrchk(cudaFree(d_in2));
	gpuErrchk(cudaFree(d_output));
	end=clock();
	time=(end-start)/CLK_TCK;
	printf("Image Add Succeed!time=%.5f,size=%d\n",time,size);
}
extern "C"
void gpu_getlamtak(float* out,float *in1,float* in2,int size)
{
	
	clock_t start,end;
	start=clock();
	float* d_sum1=NULL;
	float time;
	float* d_sum2=NULL;
	float* d_in1=NULL;
	float* d_in2=NULL;
	float h_sum1[BLOCKNUM]={0};
	float h_sum2[BLOCKNUM]={0};
	const dim3 gridSize(BLOCKNUM, 1,1);
	const dim3 blockSize(MAX_THREADNUM_PERBLOCK, 1,1);
//	h_sum1=(float*)malloc(BLOCKNUM*sizeof(float));
//	h_sum2=(float*)malloc(BLOCKNUM*sizeof(float));
	gpuErrchk(cudaMalloc((void**)&d_in1,size*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_in2,size*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_in1,in1,size*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_in2,in2,size*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&d_sum1,BLOCKNUM*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_sum2,BLOCKNUM*sizeof(float)));
	gpu_Getlamta<<<gridSize,blockSize,2*MAX_THREADNUM_PERBLOCK*sizeof(float)>>>(d_in1,d_in2,d_sum1,d_sum2,size);
	gpuErrchk(cudaMemcpy(h_sum1,d_sum1,BLOCKNUM*sizeof(float),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_sum2,d_sum2,BLOCKNUM*sizeof(float),cudaMemcpyDeviceToHost));
	long int i;
	float final_sum1=0;
	float final_sum2=0;
	for(i=0;i<BLOCKNUM;i++)
	{
		final_sum1 +=h_sum1[i];
		final_sum2 +=h_sum2[i];
	}
	if(0!=final_sum2)
		*out=final_sum1/final_sum2;
	gpuErrchk(cudaFree(d_in1));
	gpuErrchk(cudaFree(d_sum1));
	gpuErrchk(cudaFree(d_in2));
	gpuErrchk(cudaFree(d_sum2));
	end=clock();
	time=(end-start)/CLK_TCK;
	printf("Get lamtak Succeed!lamtak=%.5f\n",*out);
}
extern "C"
void gpu_MyArtprojection( float *img,float* outgg,int xmin,int xmax,int ymin,int ymax,int depth)
{
	printf("Start Projection calculating...\n");
	clock_t start,end;
	float time;
	start=clock();
	float* d_output=NULL;
	float* d_imgin=NULL;
	//caculate_WD();
	theta_init();
	gpuErrchk(cudaMalloc((void**)&d_output,DetectX*DetectZ*frameN*sizeof(float)));
	gpuErrchk(cudaMemset((void*)d_output,0,DetectX*DetectZ*frameN));
	cudaArray* cuArray_a;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extentA;
	extentA.width = imageX;
	extentA.height = imageY;
	extentA.depth = depth;
	gpuErrchk(cudaMalloc3DArray(&cuArray_a, &channelDesc, extentA)) ;
	cudaMemcpy3DParms myparms = {0};
    myparms.srcPos = make_cudaPos(0,0,0);
    myparms.dstPos = make_cudaPos(0,0,0);
    myparms.srcPtr = make_cudaPitchedPtr(img, imageX * sizeof(float), imageX, imageX);
    myparms.dstArray = cuArray_a;
    myparms.extent = make_cudaExtent(imageX, imageY, depth);
    myparms.kind = cudaMemcpyHostToDevice;
    gpuErrchk(cudaMemcpy3D(&myparms));

	img_tex.addressMode[0] = cudaAddressModeBorder;
	img_tex.addressMode[1] = cudaAddressModeBorder;
	img_tex.addressMode[2] = cudaAddressModeBorder;
	img_tex.filterMode = cudaFilterModeLinear;
	img_tex.normalized = false;
	cudaBindTextureToArray(img_tex, cuArray_a, channelDesc);
	//gpuErrchk(cudaMalloc((void**)&d_ggin,imageH*imageW*(ztop-zbot)*sizeof(float)));
	//gpuErrchk(cudaMemcpy(d_ggin,img,height*width*frameN*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset((void*)d_output,0,DetectX*DetectZ*frameN*sizeof(float)));
	const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
	const dim3 gridSize(16, 16,1);
	gpu_projection<<<gridSize,blockSize>>>(d_output,xmin,xmax,ymin,ymax);
	gpuErrchk(cudaPeekAtLastError());
	cudaDeviceSynchronize();
	gpuErrchk(cudaMemcpy(outgg,d_output,DetectX*DetectZ*frameN*sizeof(float),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_output));
	cudaUnbindTexture(img_tex);
	gpuErrchk(cudaFreeArray(cuArray_a));
	//gpuErrchk(cudaFree(d_ggin));
	end=clock();
	time=(end-start)/CLK_TCK;
	printf("Projection Succeed!time=%.5f\n",time);
}
extern "C"
void gpu_Artprojection( float *img,float* outgg,int xmin,int xmax,int ymin,int ymax,int depth)
{
	printf("Start Projection calculating...\n");
	clock_t start,end;
	float time;
	start=clock();
	float* d_output=NULL;
	float* d_imgin=NULL;
	double* z0=NULL;double*zx=NULL;double* zy=NULL;double *zz=NULL;
	long * J=NULL;
	double* L=NULL;
	//caculate_WD();
	theta_init();
	gpuErrchk(cudaMalloc((void**)&z0,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMemset((void*)z0,0,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&zx,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMemset((void*)zx,0,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&zy,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMemset((void*)zy,0,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&zz,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMemset((void*)zz,0,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&J,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMemset((void*)J,0,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&L,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMemset((void*)L,0,(imageN*3+1)*sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_output,DetectX*DetectZ*frameN*sizeof(float)));
	gpuErrchk(cudaMemset((void*)d_output,0,DetectX*DetectZ*frameN*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_imgin,imageX*imageY*(ztop-zbot)*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_imgin,img,imageX*imageY*(ztop-zbot)*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset((void*)d_output,0,DetectX*DetectZ*frameN*sizeof(float)));
	const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
	const dim3 gridSize((DetectX+blockSize.x-1) / blockSize.x, (DetectZ+blockSize.y-1) / blockSize.y,1);
	//gpu_projection<<<gridSize,blockSize>>>(d_output,xmin,xmax,ymin,ymax);
	get_projection<<<gridSize,blockSize>>>(d_imgin,d_output,z0,zx,zy,zz,J,L);
	gpuErrchk(cudaPeekAtLastError());
	cudaDeviceSynchronize();
	gpuErrchk(cudaMemcpy(outgg,d_output,DetectX*DetectZ*frameN*sizeof(float),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_output));
	gpuErrchk(cudaFree(d_imgin));
	gpuErrchk(cudaFree(z0));
	gpuErrchk(cudaFree(zx));
	gpuErrchk(cudaFree(zy));
	gpuErrchk(cudaFree(zz));
	gpuErrchk(cudaFree(J));
	gpuErrchk(cudaFree(L));
	end=clock();
	time=(end-start)/CLK_TCK;
	printf("Projection Succeed!time=%.5f\n",time);
}

extern "C"
void gpu_fdkbackprj(float *prj,float *img,int width,int height,int imageW,int imageH,int bottom,int top)
{
	printf("Start BackProjection calculating...\n");
	clock_t start,end;
	float time;
	start=clock();
	theta_init();
	int k;
	unsigned int i;
	const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
	const dim3 gridSize((imageW+blockSize.x-1) / blockSize.x, (imageH+blockSize.y-1) / blockSize.y,1);
	float *d_outimg=NULL;
	float *d_gg=NULL;
	gpuErrchk(cudaMalloc((void**)&d_outimg,imageH*imageW*(top-bottom)*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_gg,width*height*frameN*sizeof(float)));
	gpuErrchk(cudaMemset((void*)d_outimg,0,imageH*imageW*(top-bottom)*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_gg,prj,height*width*frameN*sizeof(float),cudaMemcpyHostToDevice));
	/*cudaArray* cuArray=allocateProjectionArray(width,height,frameN);
	if(!transferProjectionsToArray( prj, cuArray,  width, height,frameN))
		return;
	else
		if(!bindProjDataTexture(cuArray))return;	*/
	gpu_fdk<<<gridSize, blockSize>>>(d_gg,d_outimg,imageW,imageH,zbot,ztop);
	gpuErrchk(cudaPeekAtLastError());
	cudaDeviceSynchronize();
	
	gpuErrchk(cudaMemcpy(img,d_outimg,imageW*imageH*(top-bottom)*sizeof(float),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_outimg));

	gpuErrchk(cudaFree(d_gg));
	end=clock();
	time=(end-start)/CLK_TCK;
	printf("BacProjection Succeed!time=%.5f\n",time);
}
extern "C"
void gpu_backprj(float *prj,float *img,int width,int height,int imageW,int imageH,int bottom,int top)
{
	printf("Start BackProjection calculating...\n");
	clock_t start,end;
	float time;
	start=clock();
	theta_init();
	int k;
	unsigned int i;
	const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
	const dim3 gridSize((imageW+blockSize.x-1) / blockSize.x, (imageH+blockSize.y-1) / blockSize.y,1);
	float *d_outimg=NULL;
	float *d_gg=NULL;
	gpuErrchk(cudaMalloc((void**)&d_outimg,imageH*imageW*(top-bottom)*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_gg,width*height*frameN*sizeof(float)));
	gpuErrchk(cudaMemset((void*)d_outimg,0,imageH*imageW*(top-bottom)*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_gg,prj,height*width*frameN*sizeof(float),cudaMemcpyHostToDevice));
	/*cudaArray* cuArray=allocateProjectionArray(width,height,frameN);
	if(!transferProjectionsToArray( prj, cuArray,  width, height,frameN))
		return;
	else
		if(!bindProjDataTexture(cuArray))return;	*/
	gpu_backProjection<<<gridSize, blockSize>>>(d_gg,d_outimg,imageW,imageH,zbot,ztop);
	gpuErrchk(cudaPeekAtLastError());
	cudaDeviceSynchronize();
	
	gpuErrchk(cudaMemcpy(img,d_outimg,imageW*imageH*(top-bottom)*sizeof(float),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_outimg));

	gpuErrchk(cudaFree(d_gg));
	end=clock();
	time=(end-start)/CLK_TCK;
	printf("BacProjection Succeed!time=%.5f\n",time);
}
////////////////////////////////////////////////////////////////////////////////

void gpu_release()
{
	//gpuErrchk(cudaFree(d_gg));
}

void caculate_WD()
{  
	float* WDO1=new float[frameN];
	float M[N0],N[N0],Dpq[N0][N0],up[N0],thitap[N0],wq[N0],aver_dp[N0],aver_up[N0],aver_thitap[N0],Dp[N0];
	float tempt0,tempt1,sum0,sum1;
	int i,j,k,p,q;
	for(i=0;i<N0-1;i++)
	{
		M[i] = (i+1)*360.0/frameN;   //保存未进行加权距离正交计算的角度
		N[i] = 0;     //保存进行加权距离正交计算后的角度
	}
	int L,Q;
	Q=1;
	for(L=N0-1;L>0;L--)
	{
		for(p=0;p<L;p++)
			for(q=0;q<Q;q++)
			{ tempt0 = abs(M[p] - N[q]);
		Dpq[p][q] =  tempt0 < 90 - tempt0 ? tempt0 : 90-tempt0;
		}  //计算投影p、q之间的 距离


		for(q=0;q<Q;q++)
			wq[q] = (q+1)*1.0/Q ; //

		sum0 = 0;
		for(q=0;q<Q;q++)
			sum0 += wq[q]; //

		for(p=0;p<L;p++)
		{ sum1 = 0;
		for(q=0;q<Q;q++)
			sum1 += wq[q]*(45-Dpq[p][q]) ;
		up[p] = sum1*1.0/sum0;
		}    //

		for(p=0;p<L;p++)
		{   sum1 =0;
		for(q=0;q<Q;q++)
			sum1 += Dpq[p][q];
		aver_dp[p] = sum1/Q ;
		}//

		for(p=0;p<L;p++)
		{  sum1 =0;
		for(q=0;q<Q;q++)
			sum1 += wq[q]*(Dpq[p][q] - aver_dp[p])*(Dpq[p][q] - aver_dp[p]);
		thitap[p] = sqrt(sum1/sum0);
		}
		//

		float min_up,max_up,min_thitap,max_thitap;
		min_up = up[0];
		max_up = up[0];
		min_thitap = thitap[0];
		max_thitap = thitap[0];
		for(p=1;p<L;p++)
		{
			if(min_up > up[p] ) min_up = up[p];
			if(max_up < up[p] ) max_up = up[p];
			if(min_thitap > thitap[p] ) min_thitap = thitap[p] ;
			if(max_thitap < thitap[p] ) max_thitap = thitap[p] ;

		}		


		for(p=0;p<L;p++)
		{
			aver_up[p] = (up[p] - min_up)/(max_up -min_up);
			aver_thitap[p] = (thitap[p] - min_thitap)/(max_thitap - min_thitap);
			Dp[p] = aver_up[p]*aver_up[p] + 0.5*aver_thitap[p]*aver_thitap[p];
		}

		float min_D;
		min_D = Dp[0];
		k=0;
		for(p=1;p<L;p++)
		{
			if(min_D>Dp[p])
			{
				min_D = Dp[p];
				k=p;
			}

		}
		N[Q] = M[k];
		M[k] = M[L-1];

		Q++ ;

	}

	N[N0-1] = M[0];
	for(k=0;k<N0;k++)
	{
		WDO1[k*4] = N[k]*frameN/360;
		WDO1[k*4+1] =( N[k] +90)*frameN/360;
		WDO1[k*4+2] =(N[k]+ 180)*frameN/360;
		WDO1[k*4 + 3] = (N[k] + 270)*frameN/360;
	}
gpuErrchk( cudaMemcpyToSymbol(gc_WDO, WDO1, frameN*sizeof(float), 0, cudaMemcpyHostToDevice));
}