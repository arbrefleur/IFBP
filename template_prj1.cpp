
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <dos.h>
#include <conio.h>
#include <io.h>
#include <string.h>
#include <time.h>
#include "devicehost.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "para.h"

typedef unsigned int  uint;
#define  N0  frameN/4
struct BIN_HEADER {	//********************* *.BIN file header struct
	char	s[492];		// Reserved
	float	min;		// Minimal value of data
	float	max;		// Maximal value of data
	int		width;		// Width of data
	int     height;		// Height of data
	int     depth;		// Depth of data (slices)
} ;
BIN_HEADER dataheader;
double z0[imageN*3+1],zx[imageN*3+1],zy[imageN*3+1],zz[imageN*3+1];
long int J[imageN*3+1];
double L[imageN*3+1];
int WDO[frameN];
double sinfai,cosfai;
float gg[frameN*DetectZ*DetectX]; //投影数据储存数组
float gg1[frameN*DetectZ*DetectX]; //投影数据储存数组
float gg2[frameN*DetectZ*DetectX]; //投影数据储存数组
float a[imageX*imageY*(ztop-zbot)];//重建图像数组
float atemp[imageX*imageY*(ztop-zbot)];
float atemp2[imageX*imageY*(ztop-zbot)];
float sintable[frameN];
float costable[frameN];
/*******************************************************/
void Load(char *filename);
void Loada(char *filename);
void save(int k,char *filename);
void save_temp(int k,char *filename);
int sortdifferent(double X[],double Y[],double Z[],int Z0, int Z1);//数据大小排序
void ART_restruct(int length,int m, int n, int k);
void ART_Psk_restruct(int length,int m, int n, int k);
void ART_Phk_restruct(int length,int m, int n, int k);
float lamtak_get(float* ,float*);
void start_filter();
void filter(float* prj);
void fdk_restruct(float* prjs,float* img);
void img_add(float*,float*,float);
void constraint(float *img,int xmin,int xmax,int ymin,int ymax,int zmin,int zmax,float im_min,float im_max);
void loadpixelvalue(char *filename);
void Save_proj_data(char *filename);
void Save_proj_data1(char *filename);
void aculate_WDO();
void start_restruct_fdk(float *filter_prj,float* outimg);
extern "C" void gpu_fdkbackprj(float *prj,float *img,int height,int width,int imageW,int imageH,int bottom,int top);
extern "C" void gpu_molloc(int width,int height,int frames);
extern "C" void gpu_release();
/*******************************************************/

int main(int argc, char* argv[])
{

	int i,j,m,n,k,num,m0;
	int length;
	dataheader.height = imageY;
	dataheader.width =imageX;
	dataheader.depth = (ztop -zbot);
	double fai= ((double)1.0)*Angle*pi/180;
	sinfai = sin(fai);
	cosfai = cos(fai);
	for(i =0;i<frameN;i++)
	{
		sintable[i] = sin((frameN-1-i)*2*pi/frameN);
		costable[i] = cos((frameN-1-i)*2*pi/frameN);//逆时针旋转
		//sintable[i] = sin(i*2*pi/frameN);
		//costable[i] = cos(i*2*pi/frameN);
	}
	for(i=0;i<imageX*imageY*(ztop-zbot);i++) a[i] = 0;		
	for(i=0;i<imageX*imageY*(ztop-zbot);i++) atemp[i] = 0;		
	
	for(i = 0;i<=3*imageN;i++)
	{
		z0[i] = 0;
		zx[i] = 0;
		zy[i] = 0;
		zz[i] = 0;
		J[i] = 0;
		L[i] = 0;
	}
	float thita = 0;
	double time;
	clock_t start,end;
	char *proj_file = "data\\15_c1z06_29.bin";  //读取投影数据		
	Loada(proj_file);    //读取投影数据	
// 
// 	start=clock();
// 	gpu_molloc(DetectX,DetectZ,frameN);
// 	gpu_fdkbackprj(gg,a,DetectZ,DetectX,imageX,imageY,zbot,ztop);
// 	gpu_release();
// 	start_restruct_fdk(gg,a);
// 	end=clock();
// 	time = (double)(end-start)/CLK_TCK;
// 	printf("time1 : %.5f\n",time);
// 	save(0,"gpu1");
// 	return 0;
	aculate_WDO();

// 	for(m0=0;m0<frameN;m0++)
// 	{  printf("frames:%d\n",m0);
// 		m = WDO[m0];
// 		for(n=0;n<DetectZ;n++)
// 		{ 
// 			for(k=0;k<DetectX;k++)
// 			{	
// 				ART_restruct(0,m,n,k);  //调用ART重建程序
// 			}
// 		}
// 	}
// 	end = clock();	
// 	time = (double)(end-start)/CLK_TCK;
// 	printf("ART restruct time:%f\n",time);
// 	save(0,"ARTresult");
// 	return 1;
	//start_filter();
	//start=clock();

	char *prj_data_file="result\\gg1.bin";
	char *filterprj_data_file="result\\filtergg1.bin";
	//Save_proj_data(prj_data_file);   //保存投影数据

	char * pixel_file ="result\\gpu0.bin";
	loadpixelvalue(pixel_file);   //读取图像初始值IFBP
	char *image_file1 = "result\\15_70_fh-";
	//start_filter();
	//Save_proj_data1(filterprj_data_file); 
	//start_restruct_fdk(gg,a);
	//save(0,image_file1);


	
	for(num =1;num<=NumCount;num++)
	{
		for(k=0;k<frameN;k++)
			for(i=0;i<DetectZ;i++)
				for(j=0;j<DetectX;j++)
					*(gg1+k*DetectX*DetectZ+i*DetectX+j)=0;			
		printf("----------------------Iterate num: %d-------------------------\n",num);
		printf("begin forward projection...\n");
		start=clock();
		length = 0;		
		/***************get (b-PSk)*****************************/

		for(m0=0;m0<frameN;m0++)
		{  
			m = WDO[m0];
			for(n=0;n<DetectZ;n++)
			{ 
				for(k=0;k<DetectX;k++)
				{	
					ART_Psk_restruct(length,m,n,k);  //调用ART重建程序
				}
			}
		}
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get (b-Phk) time : %.5f\n",time);
		/****************** get hk=B(b-Psk)************************/
		//
	//	save_temp(num,image_file1);
		//Save_proj_data(prj_data_file);
		Save_proj_data(prj_data_file);
		/*********  get B(b-Phk) **********/
		filter(gg1);
		Save_proj_data1(filterprj_data_file); 
		//fdk_restruct(gg1,atemp);//restruct_fdk(float* prjs,float* img)
		gpu_molloc(DetectX,DetectZ,frameN);
		gpu_fdkbackprj(gg1,atemp,DetectZ,DetectX,imageX,imageY,zbot,ztop);
		gpu_release();
		save_temp(0,"atemp");
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get hk=B(b-Phk) time : %.5f\n",time);
		/*****************get lmtak  *********************************/
		/*****************get gg2=(Phk)  *********************************/
		for(m0=0;m0<frameN;m0++)
		{  
			m = WDO[m0];
			for(n=0;n<DetectZ;n++)
			{ 
				for(k=0;k<DetectX;k++)
				{	
					ART_Phk_restruct(length,m,n,k);
				}
			}
		}
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get (Phk) time : %.5f\n",time);
		/*****      get (B(b-Phk)  ****/
		filter(gg2);
		//fdk_restruct(gg2,atemp2);
		gpu_molloc(DetectX,DetectZ,frameN);
		gpu_fdkbackprj(gg2,atemp2,DetectZ,DetectX,imageX,imageY,zbot,ztop);
		gpu_release();
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get BPhk time : %.5f\n",time);
		
		/*****************get hkTBPhk*********************/
		float lamtak=0.1;
		lamtak=lamtak_get(atemp,atemp2);
		printf("lamtak=  : %.5f\n",lamtak);
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get lamtak time : %.5f\n",time);
		img_add(a,atemp,lamtak);
		save(0,"A");//保存重建图像
		constraint(a,-120,120,-100,100,-30,30,0.2,2);//constraint(float *img,int xmin,int xmax,int ymin,int ymax,int zmin,int zmax,float im_min,float im_max);
		save(0,"constrainA");//保存重建图像
		
	  
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("iteration:%d with time : %.5f\n",num,time);

	}
	
	printf("OOOOOOOOKKKKKKKKK!");
	return 0;
}
void Load(char *filename)
{
	char filepath[100];
	char str[10];
	FILE *fp;
	int i,j,k; 
	for(k=0;k<frameN;k++)
	{
		strcpy(filepath,filename);
		sprintf_s(str,"%d",k+1);
		strcat(filepath,str);
		strcat(filepath,".bin");
		fp = fopen(filepath,"rb");
		if(fp == NULL)
		{
			printf("File Open Error!");
			return;
		}
		for(i=0;i<DetectZ;i++)
			for(j=0;j<DetectX;j++)
        fread((gg+k*DetectX*DetectZ+i*DetectX+j),4,1,fp);
		fclose(fp);
	}

	
}
//读取投影文件数据
void Loada(char* filename)
{
	printf("Loading projections...\n");
	
	FILE *fp;
	int i,j,k;
	fp = fopen(filename,"rb");
	if(fp == NULL)
	{
		printf("File Open Error!");
		return;
	}
	for(k=0;k<frameN;k++)
		for(i=0;i<DetectZ;i++)
			for(j=0;j<DetectX;j++)		
			fread((gg+k*DetectX*DetectZ+i*DetectX+j),4,1,fp);
			fclose(fp);
	printf("Loada projections  success!\n");
}


//读取重建图像初始值
void loadpixelvalue(char *filename)
{
	printf("Loading pixels value...\n");
	FILE *fp;
	long  i,j,k;
	fp = fopen(filename,"rb");
	if(fp == NULL)
	{
		printf("File Open Error!");
		return;
	}
	for(i=0;i<imageX*imageY*(ztop-zbot);i++)
		fread(&a[i],4,1,fp);
	fclose(fp);

	printf("load image data success!\n");
}

//保存重建图像
void save(int k0,char *filename)
{
	long int i,j,k;
	for(i=0;i<imageX*imageY*(ztop-zbot);i++) 
		if(a[i]<0)a[i] = 0;
	FILE *fp;
	char file[100];
	char str[10];
	char str0[10] = ".bin";
	strcpy(file,filename);
	sprintf(str,"%d",k0);
	strcat(file,str);
	strcat(file,str0);
	fp = fopen(file,"wb");
	if(fp == NULL)
	{
		printf("File Open Error!");
		return;
	}
	dataheader.depth=ztop-zbot;
	dataheader.min= a[0];
	dataheader.max = a[0];
	for(k=0;k<ztop-zbot;k++)
		for(j=0;j<imageY;j++)
			for(i=0;i<imageX;i++)
			{ 	
				if(dataheader.min > a[k*imageX*imageY+j*imageY+i]) dataheader.min = a[k*imageX*imageY+j*imageY+i];
				if(dataheader.max <a[k*imageX*imageY+j*imageY+i] ) dataheader.max = a[k*imageX*imageY+j*imageY+i];
				fwrite(&a[k*imageX*imageY+j*imageY+i],4,1,fp);
			}
	fwrite(&dataheader,sizeof(BIN_HEADER),1,fp);
	fclose(fp);

	printf("save to (%s) success!\n",filename);
}
void save_temp(int k0,char *filename)
{
	long int i,j,k;
	for(i=0;i<imageX*imageY*(ztop-zbot);i++) 
		if(a[i]<0)a[i] = 0;
	FILE *fp;
	char file[100];
	char str[10];
	char str0[10] = ".bin";
	char str1[10] = "_temp";
	strcpy(file,filename);
	sprintf(str,"%d",k0);
	strcat(file,str1);
	strcat(file,str);
	strcat(file,str0);
	fp = fopen(file,"wb");
	if(fp == NULL)
	{
		printf("File Open Error!");
		return;
	}
	dataheader.depth=ztop-zbot;
	dataheader.min= atemp[0];
	dataheader.max = atemp[0];
	for(k=0;k<ztop-zbot;k++)
		for(j=0;j<imageY;j++)
			for(i=0;i<imageX;i++)
			{ 	
				if(dataheader.min > atemp[k*imageX*imageY+j*imageY+i]) dataheader.min = atemp[k*imageX*imageY+j*imageY+i];
				if(dataheader.max <atemp[k*imageX*imageY+j*imageY+i] ) dataheader.max = atemp[k*imageX*imageY+j*imageY+i];
				fwrite(&atemp[k*imageX*imageY+j*imageY+i],4,1,fp);
			}
			fwrite(&dataheader,sizeof(BIN_HEADER),1,fp);
			fclose(fp);

			printf("save temp image to (%s) success!\n",file);
}

//切片形式保存未滤波差影数据
void Save_proj_data(char *filename)
{
	printf("Saving (b-PSk)...\n");
	int i,j,k;
	FILE *fp;
	dataheader.min=*gg1;
	dataheader.max = *gg1;
	dataheader.depth=frameN;
	dataheader.height=DetectZ;
	dataheader.width=DetectX;
	fp = fopen(filename,"wb");
	if( fp == NULL) 
	{
		printf("Open File Error!");
		return;
	}
	for( i = 0;i<frameN;i++)
		for( j = 0;j<DetectZ;j++)
			for( k = 0;k<DetectX;k++)
			{
				if(dataheader.max<gg1[i*DetectX*DetectZ+DetectX*j+k])dataheader.max=gg1[i*DetectX*DetectZ+DetectX*j+k];
				if(dataheader.min>gg1[i*DetectX*DetectZ+DetectX*j+k])dataheader.min=gg1[i*DetectX*DetectZ+DetectX*j+k];
				fwrite(&gg1[i*DetectX*DetectZ+DetectX*j+k],4,1,fp);}
	fwrite(&dataheader,sizeof(BIN_HEADER),1,fp);
	fclose(fp);
	printf("Save <b-PSk> to (%s) success!\n",filename);

}
//切片形式保存gg1差分投影滤波数组
void Save_proj_data1(char *filename)
{
	printf("Saving filter projection data...\n");
	int i,j,k;
	FILE *fp;
	dataheader.min= *(gg1);
	dataheader.max = *gg1;
	dataheader.depth=frameN;
	dataheader.height=DetectZ;
	dataheader.width=DetectX;
	fp = fopen(filename,"wb");
	if( fp == NULL) 
	{
		printf("Open File Error!");
		return;
	}
	for( i = 0;i<frameN;i++)
		for( j = 0;j<DetectZ;j++)
			for( k = 0;k<DetectX;k++)
			{
				if(dataheader.max<gg1[i*DetectX*DetectZ+DetectX*j+k])dataheader.max=gg1[i*DetectX*DetectZ+DetectX*j+k];
				if(dataheader.min>gg1[i*DetectX*DetectZ+DetectX*j+k])dataheader.min=gg1[i*DetectX*DetectZ+DetectX*j+k];
				fwrite(&gg1[i*DetectX*DetectZ+DetectX*j+k],4,1,fp);}
			fwrite(&dataheader,sizeof(BIN_HEADER),1,fp);
			fclose(fp);
	printf("save to(%s) success!\n",filename);
}

int sortdifferent( double X[],double Y[],double Z[],int Z0, int Z1)
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

	return k;
}
void ART_Psk_restruct(int length,int m, int n, int k)
{	
	
	float d1,d2;
	d1 = n -DetectZ/2;
	d2 = k - DetectX/2;


	long int i,j;
	double x1,y1,z1,x2,y2,z2;
	double xs,ys,zs,xd,yd,zd;

	double y= n-imageY/2;
	double x = k-imageX/2;
	x1 = FOD*sintable[m]*sinfai+imageX/2;
	y1 =-FOD*costable[m]*sinfai+imageY/2;
	z1 =FOD*cosfai-zbot;
	x2=x*costable[m]-y*sintable[m]*cosfai-ODD*sintable[m]*sinfai+imageX/2;
	y2=x*sintable[m]+y*costable[m]*cosfai+ODD*costable[m]*sinfai+imageY/2;
	z2=y*sinfai-ODD*cosfai-zbot;

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
			c1 = sortdifferent(z0,zy,zz,c1,c2);
			c1 = sortdifferent(zz,zx,z0,c1,c3);
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


				c1 = sortdifferent(z0,zy,zz,c1,c2);
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

				c1 = sortdifferent(z0,zx,zz,c1,c3);
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

			c1 = sortdifferent(zx,zy,z0,c1,c3);

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
		for( i =0;i<c1-1;i++)
		{if(L[i] >1.733) printf("相交长度较大 m=%d,n=%d,k=%d\n",m,n,k);
		if(J[i] >imageX*imageY*(ztop-zbot)) printf("体素编号较大c1=%d, m=%d,n=%d,k=%d,编号i=%d,体素号%d,长度%.3f\n",c1,m,n,k,i,J[i],L[i]);
		}

		double tempt1 =0,tempt2=0,tempt3=0;

		for( i =0;i<c1-1;i++)
		{ 
			tempt1 += L[i]*a[J[i]];
			tempt2 += L[i]*L[i];
		}


		if(tempt2<1e-6) 
		{//printf("tempt2 =%.6f\n",tempt2);
			//float ceshi;
			// scanf("%f",&ceshi);
		}else
		{
			//tempt3 = (lmta)*(gg[m][n][k] - tempt1)/tempt2;
			tempt3=*(gg+m*DetectX*DetectZ+n*DetectX+k) - tempt1;
 			//if(tempt3>0)
				*(gg1+m*DetectX*DetectZ+n*DetectX+k) =tempt3;
//			for( i =0;i<c1-1;i++)
// 				a[J[i]] = a[J[i]]+tempt3*L[i];
		}

	}




}
void aculate_WDO()
{  

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
		WDO[k*4] = N[k]*frameN/360;
		WDO[k*4+1] =( N[k] +90)*frameN/360;
		WDO[k*4+2] =(N[k]+ 180)*frameN/360;
		WDO[k*4 + 3] = (N[k] + 270)*frameN/360;
	}

}
void fdk_restruct(float* prjs,float* img){
	printf("begin back projection:\n");
	float dlta = 2*pi/frameN;
		float x,y;
		float tx,ty;
		float prjx,prjy;
		int i,j,m,z,k;
		float s;
		int zoom=1;
		int offset=0;
		for(i=0;i<imageX*imageY*(ztop-zbot);i++) img[i] = 0;		
		for(k=0;k<ztop-zbot;k++)
		{
			z= k+zbot;
			for( i =0;i<imageY;i++)
			{ 
				y = i-imageY/2;
				for( j =0;j<imageX;j++)
				{
					x = j -imageX/2;
					//if(x*x+ y*y <= imageY*imageX/4)
					{
						tx =x*1.0/zoom;
						ty = y*1.0/zoom;
						for( m =0;m<frameN;m++)
						{
							double tempt00 = x*sintable[m]*sinfai- y*costable[m]*sinfai+z*cosfai+ODD-DIS;
							prjx= -DIS*(x*costable[m] + y*sintable[m])/tempt00 + DetectX/2+ CX;
							prjy=-DIS*(-x*sintable[m]*cosfai+y*costable[m]*cosfai+z*sinfai)/tempt00 + DetectZ/2+ CZ;
							int xd = (int)prjx;
							float xf = prjx -xd;
							int yd = (int) prjy;
							float yf = prjy -yd;
							if(0<xd&&xd<DetectX-1 && yd>0&& yd<DetectZ-1)
							{
								float szd = (1-xf)**(prjs+m*DetectZ*DetectX+yd*DetectX+xd) + xf**(prjs+m*DetectZ*DetectX+yd*DetectX+xd+1);
								float szd1 = (1-xf)**(prjs+m*DetectZ*DetectX+(yd+1)*DetectX+xd) + xf**(prjs+m*DetectZ*DetectX+(yd+1)*DetectX+xd+1);
								float z0x0y0 = (1-yf)*szd + yf*szd1;
								img[k*imageX*imageY+i*imageX+j] += z0x0y0*dlta;
							}
						//	a[i][j] = a[i][j] + Xs*dltaBeta;
						}
						//if(atemp[k*imageX*imageY+i*imageX+j]<1e-6)atemp[k*imageX*imageY+i*imageX+j]=0;
					}
					//else
					//	atemp[k*imageX*imageY+i*imageX+j] = 0;
				}

			}
		}
		
printf("back projection success!\n");
	}
void filter(float* prj)
{
	printf("begin filtering...\n");
	float *h,*hs;
	long i,m,l,j;
	h = new float[DetectX+1];
	hs = new float[DetectX*2];

	for( i =1;i<=DetectX;i++) 
		if(i%2==0)
			h[i]=i*cos(15*pi/180)/(4*pi*pi*(i*i-1));
		else
			h[i]=cos(15*pi/180)/(4*pi*pi*i);		
	for(i=1;i<2*DetectX;i++)
		hs[i]=h[abs(i-DetectX)+1];

// 	for( i =0;i<DetectX;i++) h[i] = -2.0/(pi*pi*(4*i*i - 1));
// 	for( i =1;i<2*DetectX;i++)
// 		hs[i] = h[abs(i-DetectX)];

	float *p0,*ps;
	p0 =new float[DetectX];
	ps = new float[3*DetectX];
	double tempt0,tempt1;
	for( m =0;m<frameN;m++)
	{
		for( l =0; l<DetectZ;l++)
		{
			tempt0 = *(prj+m*DetectZ*DetectX+l*DetectX);
			tempt1 = (*(prj+m*DetectZ*DetectX+l*DetectX+DetectX-1) + *(prj+m*DetectZ*DetectX+l*DetectX+DetectX-2))/2;
			//投影数据进行扩充
			for(i = 1;i<DetectX;i++) ps[i] = tempt0;
			for(i = DetectX;i<DetectX*2;i++) ps[i] = *(prj+m*DetectZ*DetectX+l*DetectX+i-DetectX);
			for( i =2*DetectX;i<DetectX*3;i++) ps[i] =tempt1;
			//卷积滤波
			float sum ;
			for( i =0;i<DetectX;i++)
			{ 
				sum = 0;
				for( j=1;j<2*DetectX;j++ )
					sum += hs[j]*ps[i+j] ;
				*(prj+m*DetectZ*DetectX+l*DetectX+i) = sum/2;
			}
		}
		
	}
	printf("filter success\n");
}
void start_restruct_fdk(float *filter_prj,float* outimg){
	printf("----------------reconstruct s0----------\n");
	float dlta = 2*pi/frameN;
	float x,y;
	float tx,ty;
	float prjx,prjy;
	int i,j,m,z,k;
	float s;
	int zoom=1;
	int offset=0;
	for(k=0;k<ztop-zbot;k++)
	{
		z= k+zbot;
		for( i =0;i<imageY;i++)
		{ 
			y = i-imageY/2;
			for( j =0;j<imageX;j++)
			{
				x = j -imageX/2;
				if(x*x+ y*y <= imageY*imageX/4)
				{
					float temp=0;
					for( m =0;m<frameN;m++)
					{
						double tempt00 = x*sintable[m]*sinfai- y*costable[m]*sinfai+z*cosfai+ODD-DIS;
						prjx= -DIS*(x*costable[m] + y*sintable[m])/tempt00 + DetectX/2+ CX;
						prjy=-DIS*(-x*sintable[m]*cosfai+y*costable[m]*cosfai+z*sinfai)/tempt00 + DetectZ/2+ CZ;
						int xd = (int)prjx;
						float xf = prjx -xd;
						int yd = (int) prjy;
						float yf = prjy -yd;
						
						if(0<xd&&xd<DetectX-1 && yd>0&& yd<DetectZ-1)
						{
							float szd = (1-xf)**(filter_prj+m*DetectZ*DetectX+yd*DetectX+xd) + xf**(filter_prj+m*DetectZ*DetectX+yd*DetectX+xd+1);
							float szd1 = (1-xf)**(filter_prj+m*DetectZ*DetectX+(yd+1)*DetectX+xd) + xf**(filter_prj+m*DetectZ*DetectX+(yd+1)*DetectX+xd+1);
							float z0x0y0 = (1-yf)*szd + yf*szd1;
							temp += z0x0y0*dlta;
						}
						//	a[i][j] = a[i][j] + Xs*dltaBeta;
					}
					if(temp<1e-6)temp=0;
					outimg[k*imageX*imageY+i*imageX+j]=temp;
					
				}
				else
					outimg[k*imageX*imageY+i*imageX+j] = 0;
			}

		}
	}
	printf("get s0 success!\n");
}
void start_filter()
{
	printf("begin start filtering...\n");
	float *h,*hs;
	long i,m,l,j;
	h = new float[DetectX+1];
	hs = new float[DetectX*2];

// 	for( i =1;i<=DetectX;i++) 
// 		if(i%2==0)
// 			h[i]=i*cosfai/(4*pi*pi*(i*i-1));
// 		else
// 			h[i]=cosfai/(4*pi*pi*i);		
// 	for(i=1;i<2*DetectX;i++)
// 		hs[i]=h[abs(i-DetectX)+1];

	 	for( i =0;i<DetectX;i++) h[i] = -2.0/(pi*pi*(4*i*i - 1));
	 	for( i =1;i<2*DetectX;i++)
	 		hs[i] = h[abs(i-DetectX)];

	float *p0,*ps;
	p0 =new float[DetectX];
	ps = new float[3*DetectX];
	double tempt0,tempt1;
	for( m =0;m<frameN;m++)
	{
		for( l =0; l<DetectZ;l++)
		{
			tempt0 = *(gg+m*DetectZ*DetectX+l*DetectX);
			tempt1 = (*(gg+m*DetectZ*DetectX+l*DetectX+DetectX-1) + *(gg+m*DetectZ*DetectX+l*DetectX+DetectX-2))/2;
			//投影数据进行扩充
			for(i = 1;i<DetectX;i++) ps[i] = tempt0;
			for(i = DetectX;i<DetectX*2;i++) ps[i] = *(gg+m*DetectZ*DetectX+l*DetectX+i-DetectX);
			for( i =2*DetectX;i<DetectX*3;i++) ps[i] =tempt1;
			//卷积滤波
			float sum ;
			for( i =0;i<DetectX;i++)
			{ 
				sum = 0;
				for( j=1;j<2*DetectX;j++ )
					sum += hs[j]*ps[i+j] ;
				*(gg+m*DetectZ*DetectX+l*DetectX+i) = sum/2;
			}
		}

	}
	printf("start filter success\n");

}
void ART_Phk_restruct(int length,int m, int n, int k)
{	

	float d1,d2;
	d1 = n -DetectZ/2;
	d2 = k - DetectX/2;


	long int i,j;
	double x1,y1,z1,x2,y2,z2;
	double xs,ys,zs,xd,yd,zd;

	double y= n-imageY/2;
	double x = k-imageX/2;
	x1 = FOD*sintable[m]*sinfai+imageX/2;
	y1 =-FOD*costable[m]*sinfai+imageY/2;
	z1 =FOD*cosfai-zbot;
	x2=x*costable[m]-y*sintable[m]*cosfai-ODD*sintable[m]*sinfai+imageX/2;
	y2=x*sintable[m]+y*costable[m]*cosfai+ODD*costable[m]*sinfai+imageY/2;
	z2=y*sinfai-ODD*cosfai-zbot;

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
			c1 = sortdifferent(z0,zy,zz,c1,c2);
			c1 = sortdifferent(zz,zx,z0,c1,c3);
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


				c1 = sortdifferent(z0,zy,zz,c1,c2);
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

				c1 = sortdifferent(z0,zx,zz,c1,c3);
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

			c1 = sortdifferent(zx,zy,z0,c1,c3);

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
		for( i =0;i<c1-1;i++)
		{if(L[i] >1.733) printf("相交长度较大 m=%d,n=%d,k=%d\n",m,n,k);
		if(J[i] >imageX*imageY*(ztop-zbot)) printf("体素编号较大c1=%d, m=%d,n=%d,k=%d,编号i=%d,体素号%d,长度%.3f\n",c1,m,n,k,i,J[i],L[i]);
		}

		double tempt1 =0,tempt2=0,tempt3=0;

		for( i =0;i<c1-1;i++)
		{ 
			tempt1 += L[i]*atemp[J[i]];
			tempt2 += L[i]*L[i];
		}


		if(tempt2<1e-6) 
		{//printf("tempt2 =%.6f\n",tempt2);
			//float ceshi;
			// scanf("%f",&ceshi);
		}else
		{
			//tempt3 = (lmta)*(gg[m][n][k] - tempt1)/tempt2;
			
			//if(tempt3>0)
			*(gg2+m*DetectX*DetectZ+n*DetectX+k) =tempt1;
			//			for( i =0;i<c1-1;i++)
			// 				a[J[i]] = a[J[i]]+tempt3*L[i];
		}

	}




}
float lamtak_get(float* tempimg ,float* tempimg1 )
{
	long i;
	float temp1=0;
	float temp2=0;
	for(i=0;i<imageX*imageY*(ztop-zbot);i++)
	{
		temp1 +=tempimg[i]*tempimg[i];
		temp2 +=tempimg[i]*tempimg1[i];
	}
	if(temp2==0)
		return 0.01;
	return temp1/temp2;
}
void constraint(float *img,int xmin,int xmax,int ymin,int ymax,int zmin,int zmax,float im_min,float im_max)
{
	int i,j,k;
	for(k=0;k<ztop-zbot;k++)
		for(i=0;i<imageY;i++)
			for(j=0;j<imageX;j++)
				//if(k>5&&k<65&&i>=100&&i<=300&&j>=78&&j<321)
					if((j-200)*(j-200)/(120*120)+(i-200)*(i-200)/(120*120)+(k-35)*(k-35)/(30*30)<1)
					{ 
						img[k*imageX*imageY+i*imageX+j] += atemp[i];	
						if(img[k*imageX*imageY+i*imageX+j]>im_max)img[k*imageX*imageY+i*imageX+j]=im_max;
						if(img[k*imageX*imageY+i*imageX+j]<im_min)img[k*imageX*imageY+i*imageX+j]=im_min;
					}
					else img[k*imageX*imageY+i*imageX+j]=0;
}
void img_add(float* img,float* atemp,float lamtak)
{
	long i;
	for(i=0;i<imageX*imageY*(ztop-zbot);i++)
		img[i]+=lamtak*atemp[i];
}
void ART_restruct(int length,int m, int n, int k)
{	

	float d1,d2;
	d1 = n -DetectZ/2;
	d2 = k - DetectX/2;


	long int i,j;
	double x1,y1,z1,x2,y2,z2;
	double xs,ys,zs,xd,yd,zd;

	double y= n-imageY/2;
	double x = k-imageX/2;
	x1 = FOD*sintable[m]*sinfai+imageX/2;
	y1 =-FOD*costable[m]*sinfai+imageY/2;
	z1 =FOD*cosfai-zbot;
	x2=x*costable[m]-y*sintable[m]*cosfai-ODD*sintable[m]*sinfai+imageX/2;
	y2=x*sintable[m]+y*costable[m]*cosfai+ODD*costable[m]*sinfai+imageY/2;
	z2=y*sinfai-ODD*cosfai-zbot;

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
			c1 = sortdifferent(z0,zy,zz,c1,c2);
			c1 = sortdifferent(zz,zx,z0,c1,c3);
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


				c1 = sortdifferent(z0,zy,zz,c1,c2);
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

				c1 = sortdifferent(z0,zx,zz,c1,c3);
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

			c1 = sortdifferent(zx,zy,z0,c1,c3);

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
		for( i =0;i<c1-1;i++)
		{if(L[i] >1.733) printf("相交长度较大 m=%d,n=%d,k=%d\n",m,n,k);
		if(J[i] >imageX*imageY*(ztop-zbot)) printf("体素编号较大c1=%d, m=%d,n=%d,k=%d,编号i=%d,体素号%d,长度%.3f\n",c1,m,n,k,i,J[i],L[i]);
		}

		double tempt1 =0,tempt2=0,tempt3=0;

		for( i =0;i<c1-1;i++)
		{ 
			tempt1 += L[i]*a[J[i]];
			tempt2 += L[i]*L[i];
		}


		if(tempt2<1e-6) 
		{//printf("tempt2 =%.6f\n",tempt2);
			//float ceshi;
			// scanf("%f",&ceshi);
		}else
		{
			tempt3 = (lmta)*(*(gg+m*DetectX*DetectZ+n*DetectX+k) - tempt1)/tempt2;
			//tempt3=*(gg+m*DetectX*DetectZ+n*DetectX+k) - tempt1;
			if(tempt3>0)
			//*(gg1+m*DetectX*DetectZ+n*DetectX+k) =tempt3;
						for( i =0;i<c1-1;i++)
			 				a[J[i]] = a[J[i]]+tempt3*L[i];
		}

	}




}