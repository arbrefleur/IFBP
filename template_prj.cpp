
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
float aa[imageX*imageY*(ztop-zbot)];//重建图像数组
float atemp[imageX*imageY*(ztop-zbot)];
float atemp2[imageX*imageY*(ztop-zbot)];
float sintable[frameN];
float costable[frameN];
/*******************************************************/
void Load(char *filename);
void Loada(float*,char *filename);
void save(float*,int k,char *filename);
void save_temp(int k,char *filename);
int sortdifferent(double X[],double Y[],double Z[],int Z0, int Z1);//数据大小排序
void ART_restruct(int length,int m, int n, int k,float* img);
void cpu_projection(int length,int m, int n, int k,float* img=a,float* prj=gg);
void myArt_restruct(int length,int m, int n, int k,float* img=a,float* prj=gg);
void ART_Phk_restruct(int length,int m, int n, int k,float* img);
float lamtak_get(float* ,float*);
void diff_prj(float* prj);
void filter(float* prj,char type[10]);
void fdk_restruct(float* prjs,float* img);
void img_add(float* out,float* in1,float* in2,float tmp,long size);
void constraint(float *img,int x1,int y1,int x2,int y2,int x3,int y3,int x4,int y4,int zmin,int zmax,float im_min,float im_max);//顺时针左上角开始
void loadpixelvalue(float*,char *filename);
void Save_proj_data(float* prj,int num,char *filename);
void zhuanzhi(float* img1,float * img2);
void aculate_WDO();
void start_restruct_fdk(float *filter_prj,float* outimg);
extern "C" void gpu_getlamtak(float*,float *in1,float* in2,int);
extern "C" void gpu_fdkbackprj(float *prj,float *img,int detector_width,int detector_height,int imageW,int imageH,int bottom,int top);
extern "C" void gpu_Artprojection(const float *img,float* outgg,int xmin,int xmax,int ymin,int ymax,int depth);
extern "C" void gpu_MyArtprojection(const float *img,float* outgg,int xmin,int xmax,int ymin,int ymax,int depth);
extern "C" void gpu_imgAdd(float *in1,float* in2,float* out, float scale,int size);
extern "C" void theta_init();
extern "C" void gpu_imgfilter(float* prj,float* gg1,char type[10]);
extern "C" void gpu_backprj(float *prj,float *img,int width,int height,int imageW,int imageH,int bottom,int top);
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
		//sintable[i] = sin((frameN-1-i)*2*pi/frameN);
		//costable[i] = cos((frameN-1-i)*2*pi/frameN);//逆时针旋转,仅用于CPU重建函数，GPU的一律用theta_init()
		if(z_dir)
		{
			sintable[i] = sin(i*2*pi/frameN);
			costable[i] = cos(i*2*pi/frameN);
		}
		else
		{
			sintable[i] = sin((frameN-1-i)*2*pi/frameN);
			costable[i] = cos((frameN-1-i)*2*pi/frameN);
		}
	}
	for(i=0;i<imageX*imageY*(ztop-zbot);i++) a[i] = 0;
	for(i=0;i<imageX*imageY*(ztop-zbot);i++) atemp[i] = 0;	
	for(i=0;i<imageX*imageY*(ztop-zbot);i++) atemp2[i] = 0;	
	for(i=0;i<DetectX*DetectZ*frameN;i++) gg[i] = 0;	
	for(i=0;i<DetectX*DetectZ*frameN;i++) gg1[i] = 0;
	//for(i=0;i<DetectX*DetectZ*frameN;i++) gg2[i] = 0;
	
	for(i = 0;i<=3*imageN;i++){		z0[i] = 0;zx[i] = 0;zy[i] = 0;zz[i] = 0;J[i] = 0;L[i] = 0;	}
	float thita = 0;
	double time;
	clock_t start,end;
	
	char *proj_file = "..\\..\\IFBP仿真\\prj\\prj_SX20DX20-0.bin";  //读取投影数据	
	char *filt_proj_file = "..\\..\\IFBP仿真\\result\\filt_prj-"; 
	char *p_sk="..\\..\\IFBP仿真\\result\\p-sk-"; 
	char * hk_file="..\\..\\IFBP仿真\\result\\hk-"; 
	char * fbpImg_file ="..\\..\\IFBP仿真\\result\\fbp_sl_nop_s";
	//char * s0Img_file="D:\\huxinhua\\仿真\\0812\\result\\s-fl-10.bin";
	char *s0Img_file = "..\\..\\IFBP仿真\\model\\newline.bin"; 
	char * skImg_file ="..\\..\\IFBP仿真\\0812\\result\\s-sl1-";
	char * sk_noconstraint ="..\\..\\\\IFBP仿真\\result\\s-";
	char * prj_data_file="..\\..\\IFBP仿真\\prjs\\prj-";
	char * stemp_Img_file="..\\..\\IFBP仿真\\result\\stemp-";
/*	
	char *proj_file = "D:\\huxinhua\\RAM\\RAM.bin";  //读取投影数据	
	//char *proj_file = "D:\\huxinhua\\FC-1\\FC-1.bin";  //读取投影数据	
	//char * s0Img_file="D:\\huxinhua\\FC-1\\result\\snop-0.bin";
	char * s0Img_file="D:\\huxinhua\\RAM\\result\\s1.bin";
	char *filt_proj_file = "D:\\huxinhua\\FC-1\\0727result\\filt_gg-"; 
	char *p_sk="D:\\huxinhua\\FC-1\\0727result\\p-sk-"; 
	char * hk_file="D:\\huxinhua\\仿真\\result\\hk-"; 
	char * fbpImg_file ="D:\\huxinhua\\FC-1\\result\\fbp_sl_nop_s";
	char * stemp_Img_file="D:\\huxinhua\\FC-1\\0727result\\stemp-";
	//char * skImg_file ="D:\\huxinhua\\FC-1\\result\\s-art-";
	char * skImg_file ="D:\\huxinhua\\RAM\\result\\snop-";
	//char * sk_noconstraint ="D:\\huxinhua\\FC-1\\result\\snop-";
	char * sk_noconstraint ="D:\\huxinhua\\RAM\\result\\snop-";
	char * prj_data_file="D:\\huxinhua\\FC-1\\result\\psk-";
	*/
	//Load(proj_file);
//	Loada(gg,proj_file);    //读取投影数据	
//	loadpixelvalue(a,"D:\\huxinhua\\仿真\\0812\\model\\line_model.bin");   //读取图像初始值IFBP
	loadpixelvalue(a,s0Img_file);   //读取图像初始值IFBP
	theta_init();
//	gpu_Artprojection(a,gg,-imageX/2,imageX/2,-imageY/2,imageY/2,ztop-zbot);
	gpu_MyArtprojection(a,gg,-imageX/2,imageX/2,-imageY/2,imageY/2,ztop-zbot);
//	Save_proj_data(gg,0,"D:\\huxinhua\\仿真\\0812\\prj\\line_prj2-");
	Save_proj_data(gg,0,prj_data_file);
	gpu_imgfilter(gg,gg,"SL");
	Save_proj_data(gg,0,filt_proj_file);
	//Save_proj_data(gg,0,"D:\\huxinhua\\FC-1\\FC-11_gpufilt-");return 1;
	gpu_fdkbackprj(gg,a,DetectX,DetectZ,imageX,imageY,zbot,ztop);
	//filter(gg,"SL");
	//fdk_restruct(gg,a);
	save(a,0,sk_noconstraint);return 1;
//	save(a,0,"D:\\huxinhua\\FC-1\\0814result\\s-sl-");
//	return 1;
//	constraint(a,26,88,215,38,255,197,73,253,4,23,0,70);
//	save(a,0,skImg_file);
	//Save_proj_data(gg,0,filt_proj_file);return 1;

	//gpu_backprj(gg,a,DetectX,DetectZ,imageX,imageY,zbot,ztop);
	//save(a,0,"D:\\huxinhua\\仿真\\0717result\\newback-");
	//gpu_MyArtprojection(a,gg,-imageX/2,imageX/2,-imageY/2,imageY/2,ztop-zbot)	;
	//Save_proj_data(gg,0,prj_data_file);
	//gpu_fdkbackprj(gg,a,DetectX,DetectZ,imageX,imageY,zbot,ztop);
	//constraint(a,49,101,203,57,246,206,91,250,4,23,0,0.4);
	//fdk_restruct(gg,a);	
	//save(a,0,fbpImg_file);
	//return 1;
//	aculate_WDO();
	for(num =2;num<=2;num++)
	{
		start=clock();
		length = 0;		
		//***************get (b-PSk)*****************************
/*	//	zhuanzhi(aa,a);
		for(m0=0;m0<frameN;m0++)
		{  
			m = WDO[m0];
			for(n=0;n<DetectZ;n++)
			{ 
				for(k=0;k<DetectX;k++)
				{	
					myArt_restruct(length,m,n,k,a,gg);  //调用ART重建程序
				}
			}
		}
	//	zhuanzhi(a,aa);
		save(a,num,skImg_file);
		printf("<---------Iteration:%d	 ---------------->\n",num);
		continue;
		*/
		gpu_MyArtprojection(a,gg1,-imageX/2,imageX/2,-imageY/2,imageY/2,ztop-zbot);	
		//	diff_prj(gg1);
		gpu_imgAdd(gg,gg1,gg1,-1,DetectX*DetectZ*frameN);
		//Save_proj_data(gg1,num,p_sk);return 1;
		gpu_imgfilter(gg1,gg1,"SL");	
		//Save_proj_data(gg1,num,filt_proj_file);
		gpu_fdkbackprj(gg1,atemp,DetectX,DetectZ,imageX,imageY,zbot,ztop);//hk=B(b_PS0)
		//save(atemp,num,hk_file);
//		gpu_MyArtprojection(atemp,gg1,-imageX/2,imageX/2,-imageY/2,imageY/2,ztop-zbot);//Phk
//		gpu_imgfilter(gg1,gg1,"SL");
//		gpu_fdkbackprj(gg1,atemp2,DetectX,DetectZ,imageX,imageY,zbot,ztop);//BPhk
		float lamtak;
		//if(num==1)	lamtak=0.6;else
			lamtak=0.3;
//		gpu_getlamtak(&lamtak,atemp,atemp2,imageX*imageY*(ztop-zbot));//hkT*hk/(hkT*BPhk)
		gpu_imgAdd(a,atemp,a,lamtak,imageX*imageY*(ztop-zbot));//s0+lamta0*h0
		//	save(a,num,sk_noconstraint);return 1;
		//constraint(a,37,106,95,50,148,102,77,172,6,67,0,2);
		//	constraint(a,26,88,215,38,255,197,73,253,4,23,0,70);
		//constraint(a,17,95,265,44,296,187,54,264,2,23,0,0.8);
	//	constraint(a,80,100,320,100,320,300,80,300,5,22,0,1.6);
	//	constraint(a,49,101,203,57,246,206,91,250,3,24,0,1.6);
		save(a,num,skImg_file);
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("<---------Iteration:%d	    Time : %.5f---------------->\n",num,time);
	}
	return 1;
	aculate_WDO();
	for(num =1;num<=10;num++)
	{
		
			for(i=0;i<DetectZ*frameN*DetectX;i++)				
					*(gg1+i)=0;			
		printf("----------------------Iterate num: %d-------------------------\n",num);
		printf("begin forward projection...\n");
		start=clock();
		length = 0;		
		//***************get (b-PSk)*****************************
		zhuanzhi(aa,a);
		for(m0=0;m0<frameN;m0++)
		{  
			m = WDO[m0];
			for(n=0;n<DetectZ;n++)
			{ 
				for(k=0;k<DetectX;k++)
				{	
					cpu_projection(length,m,n,k,aa,gg1);  //调用ART重建程序
				}
			}
		}
	Save_proj_data(gg1,num,prj_data_file);return 1;
		img_add(gg1,gg,gg1,-1,DetectX*DetectZ*frameN);
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get (b-Phk) time : %.5f\n",time);
		//****************** get hk=B(b-Psk)************************
		//
	//	save_temp(num,image_file1);
		Save_proj_data(gg1,num,prj_data_file);
		//Save_proj_data(prj_data_file);
		//*********  get B(b-Phk) **********
		filter(gg1,"RL-HN");
		Save_proj_data(gg1,num,filt_proj_file); 
		fdk_restruct(gg1,atemp);//restruct_fdk(float* prjs,float* img)
		
		//gpu_fdkbackprj(gg1,atemp,DetectZ,DetectX,imageX,imageY,zbot,ztop);

		save(atemp,num,stemp_Img_file);
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get hk=B(b-Phk) time : %.5f\n",time);
		//*****************get lmtak  *********************************
		//*****************get gg2=(Phk)  *********************************
		 zhuanzhi(aa,atemp);
		for(m0=0;m0<frameN;m0++)
		{  
			m = WDO[m0];
			for(n=0;n<DetectZ;n++)
			{ 
				for(k=0;k<DetectX;k++)
				{	
					cpu_projection(length,m,n,k,aa,gg2);
				}
			}
		}
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get (Phk) time : %.5f\n",time);
		//*****      get (B(b-Phk)  ****
		filter(gg2,"RL-HN");
		fdk_restruct(gg2,atemp2);
	
		//gpu_fdkbackprj(gg2,atemp2,DetectZ,DetectX,imageX,imageY,zbot,ztop);

		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get BPhk time : %.5f\n",time);
		
		//*****************get hkTBPhk*********************
		float lamtak=0.1;
		lamtak=lamtak_get(atemp,atemp2);
		printf("lamtak=  : %.5f\n",lamtak);
		end = clock();	
		time = (double)(end-start)/CLK_TCK;
		start=end;
		printf("get lamtak time : %.5f\n",time);
		//gpu_imgAdd(a,atemp,a, lamtak,imageX*imageY*(ztop-zbot));
		img_add(a,a,atemp,lamtak,imageX*imageY*(ztop-zbot));
		save(a,num,sk_noconstraint);//保存重建图像
		constraint(a,26,88,215,38,255,197,73,253,4,23,0,0.02);
		//constraint(float *img,int xmin,int xmax,int ymin,int ymax,int zmin,int zmax,float im_min,float im_max);
		save(a,num,skImg_file);//保存重建图像
		
	}
}	
void diff_prj(float* prj)	
{
	int k,j,i;
	for(k=0;k<frameN;k++)
		for(j=0;j<DetectZ;j++)
			for(i=0;i<DetectX;i++)
				if(i>0)
					prj[k*DetectX*DetectZ+j*DetectX+i] -=prj[k*DetectX*DetectZ+j*DetectX+i-1];
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
	printf("Load projections  success!\n");
	
}
//读取投影文件数据
void Loada(float* prj,char* filename)
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
		
			fread((prj),4,frameN*DetectX*DetectZ,fp);
			fclose(fp);
	printf("Loada projection  success!\n");
}


//读取重建图像初始值
void loadpixelvalue(float* img,char *filename)
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
		fread(&img[i],4,1,fp);
	fclose(fp);

	printf("load image data success!\n");
}

//保存重建图像
void save(float* img,int k0,char *filename)
{
	long int i,j,k;
	
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
		fp=fopen("default.bin","wb");
		if(fp==NULL)
			return;
		else
			printf("Open <default.bin> instead\n");

	}
	dataheader.depth=ztop-zbot;
	dataheader.width=imageX;
	dataheader.height=imageY;
	dataheader.min= img[0];
	dataheader.max = img[0];
	for(k=0;k<ztop-zbot;k++)
		for(j=0;j<imageY;j++)
			for(i=0;i<imageX;i++)
			{ 	
				if(dataheader.min > img[k*imageX*imageY+j*imageY+i]) dataheader.min = img[k*imageX*imageY+j*imageY+i];
				if(dataheader.max <img[k*imageX*imageY+j*imageY+i] ) dataheader.max = img[k*imageX*imageY+j*imageY+i];
				fwrite(&img[k*imageX*imageY+j*imageY+i],4,1,fp);
			}
	fwrite(&dataheader,sizeof(BIN_HEADER),1,fp);
	fclose(fp);
	printf("save to (%s) success!\n",file);
}
void save_temp(int k0,char *filename,float* img)
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
	dataheader.min= img[0];
	dataheader.max = img[0];
	for(k=0;k<ztop-zbot;k++)
		for(j=0;j<imageY;j++)
			for(i=0;i<imageX;i++)
			{ 	
				if(dataheader.min > img[k*imageX*imageY+j*imageY+i]) dataheader.min = img[k*imageX*imageY+j*imageY+i];
				if(dataheader.max <img[k*imageX*imageY+j*imageY+i] ) dataheader.max = img[k*imageX*imageY+j*imageY+i];
				fwrite(&img[k*imageX*imageY+j*imageY+i],4,1,fp);
			}
			fwrite(&dataheader,sizeof(BIN_HEADER),1,fp);
			fclose(fp);

			printf("save temp image to (%s) success!\n",file);
}

//切片形式保存未滤波差影数据
void Save_proj_data(float* prj,int num,char *filename)
{
	printf("Saving Projection...\n");
	char str[10]=".bin";
	char str1[10];
	char file[100];
	int i,j,k;
	FILE *fp;
	dataheader.min=*prj;
	dataheader.max = *prj;
	dataheader.depth=frameN;
	dataheader.height=DetectZ;
	dataheader.width=DetectX;
	strcpy(file,filename);
	sprintf(str1,"%d",num);
	strcat(file,str1);
	strcat(file,str);
	fp = fopen(file,"wb");
	if( fp == NULL) 
	{
		printf("Open File Error!");
		return;
	}
	for( i = 0;i<frameN;i++)
		for( j = 0;j<DetectZ;j++)
			for( k = 0;k<DetectX;k++)
			{
				if(dataheader.max<prj[i*DetectX*DetectZ+DetectX*j+k])dataheader.max=prj[i*DetectX*DetectZ+DetectX*j+k];
				if(dataheader.min>prj[i*DetectX*DetectZ+DetectX*j+k])dataheader.min=prj[i*DetectX*DetectZ+DetectX*j+k];
				fwrite(&prj[i*DetectX*DetectZ+DetectX*j+k],4,1,fp);}
	fwrite(&dataheader,sizeof(BIN_HEADER),1,fp);
	fclose(fp);
	printf("Save projection to (%s) success!\n",file);

}
//切片形式保存gg1差分投影滤波数组


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
void cpu_projection(int length,int m, int n, int k,float* img,float* prj)
{	
	
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

	double y= n;
	double x = k;
	x1 = -FOD*cosf(alpha*pi/180)*sintable[m]+imageX/2;
	y1 =-FOD*cosf(alpha*pi/180)*costable[m]+imageY/2;
	z1 =FOD*sin(alpha*pi/180);
	x2=(x-DetectX/2-CX)*costable[m]+sintable[m]*(ODD*cosf(alpha*pi/180)+(DetectZ/2-y)*sinf(alpha*pi/180))+imageX/2;
	y2=-(x-DetectX/2-CX)*sintable[m]+(ODD*cosf(alpha*pi/180)+(DetectZ/2-y)*sin(alpha*pi/180))*costable[m]+imageY/2;
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
void myArt_restruct(int length,int m, int n, int k,float* img,float* prj)
{	
	
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

	double y= n;
	double x = k;
	x1 = -FOD*cosf(alpha*pi/180)*sintable[m]+imageX/2;
	y1 =-FOD*cosf(alpha*pi/180)*costable[m]+imageY/2;
	z1 =FOD*sin(alpha*pi/180);
	x2=(x-DetectX/2-CX)*costable[m]+sintable[m]*(ODD*cosf(alpha*pi/180)+(DetectZ/2-y)*sinf(alpha*pi/180))+imageX/2;
	y2=-(x-DetectX/2-CX)*sintable[m]+(ODD*cosf(alpha*pi/180)+(DetectZ/2-y)*sin(alpha*pi/180))*costable[m]+imageY/2;
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
			tempt1 += L[i]*img[J[i]];
			tempt2 += L[i]*L[i];
		}


		if(tempt2<1e-6) 
		{//printf("tempt2 =%.6f\n",tempt2);
			//float ceshi;
			// scanf("%f",&ceshi);
		}else
		{
			tempt3 = (lmta)*(prj[m*DetectX*DetectZ+n*DetectX+k] - tempt1)/tempt2;
		//	tempt3=*(prj1+m*DetectX*DetectZ+n*DetectX+k) - tempt1;
 			//if(tempt3>0)
		//		*(prj2+m*DetectX*DetectZ+n*DetectX+k) =tempt3;
			for( i =0;i<c1-1;i++)
 				img[J[i]] = img[J[i]]+tempt3*L[i];
	//			prj[m*DetectX*DetectZ+n*DetectX+k]=tempt1;
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
				y =- i+imageY/2;
				for( j =0;j<imageX;j++)
				{
					x = j -imageX/2;
					//if(x*x+ y*y <= imageY*imageX/4)
					{
						tx =x*1.0/zoom;
						ty = y*1.0/zoom;
						float temp=0.0f;
						for( m =0;m<frameN;m++)
						{
						    float cos_theta =costable[m];// gC_angle_cos[frameN-1-m];
						    float sin_theta =sintable[m];// gC_angle_sin[frameN-1-m];
							float u=x*cos_theta-y*sin_theta;
							float v=cosf(alpha*pi/180)*(y*cos_theta+x*sin_theta)-z*sinf(alpha*pi/180);
							float w=z*cosf(alpha*pi/180)+(y*cos_theta+x*sin_theta)*sinf(alpha*pi/180);
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
								float szd = (1-xf)**(prjs+m*DetectZ*DetectX+yd*DetectX+xd) + xf**(prjs+m*DetectZ*DetectX+yd*DetectX+xd+1);
								float szd1 = (1-xf)**(prjs+m*DetectZ*DetectX+(yd+1)*DetectX+xd) + xf**(prjs+m*DetectZ*DetectX+(yd+1)*DetectX+xd+1);
								float z0x0y0 = (1-yf)*szd + yf*szd1;
								temp +=z0x0y0;
								// temp +=dlta* tex3D(gg_tex,prjx,prjy,m);
							}
							//	a[i][j] = a[i][j] + Xs*dltaBeta;
						}
						//if(temp<1e-6)
						//temp=0;
						img[k*imageX*imageY+i*imageX+j]=temp;
						
						//if(atemp[k*imageX*imageY+i*imageX+j]<1e-6)atemp[k*imageX*imageY+i*imageX+j]=0;
					}
					//else
						//img[k*imageX*imageY+i*imageX+j] = 0;
				}

			}
		}
		
printf("back projection success!\n");
	}
void filter(float* prj,char type[10])
{
	printf("begin filtering...\n");
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
	else if(0==strcmp(type,"SL"))
	{

		for( i =1;i<=DetectX;i++) 
			if(i%2==0)
				h[i]=-1/(8*pi*pi*(i));
			else
				h[i]=1/(8*pi*pi*i);		
		for(i=1;i<2*DetectX;i++)
			hs[i]=h[abs(i-DetectX)+1];
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
		hs[DetectX]=(-1)*cos(alpha*pi/180)/(2*pi*pi);
	}
	else if(0==strcmp(type,"RL"))//RL
	{
		for( i =1;i<DetectX;i++) 
			if(i%2==0)
				h[i]=0;//(-1)*cos(alpha*pi/180)*(1/(4*pi*pi*(i+1)*(i+1))+1/(4*pi*pi*(i-1)*(i-1)));
			else
				h[i]=(-1)*cos(alpha*pi/180)/(2*pi*pi*i*i);		
		for(i=1;i<DetectX;i++)
			hs[i]=h[DetectX-i];
		for(i=DetectX+1;i<2*DetectX;i++)
			hs[i]=h[i-DetectX];
		hs[DetectX]=cos(alpha*pi/180)/8;
	}
	else //SL
	{
		for( i =0;i<DetectX;i++) 
			h[i] = -2.0/(pi*pi*(4*i*i - 1));
	 	for( i =1;i<2*DetectX;i++)
	 		hs[i] = h[abs(i-DetectX)];
	}		
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

void ART_Phk_restruct(int length,int m, int n, int k,float* img,float* prj)
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
			
			//if(tempt3>0)
			*(prj+m*DetectX*DetectZ+n*DetectX+k) =tempt1;
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
void constraint(float *img,int x1,int y1,int x2,int y2,int x3,int y3,int x4,int y4,int zmin,int zmax,float im_min,float im_max)
{
	int i,j,k;
	for(k=0;k<ztop-zbot;k++)
		for(j=0;j<imageY;j++)
			for(i=0;i<imageX;i++)
			{
				float x=i;float y=j;
		if((k<zmax)&&(k>zmin))
				//if(((x-54)/(315-54)-(y-142)/(74-142))>0&&((x-92)/(358-92)-(y-311)/(238-311)<0)&&((x-54)/(92-54)-(y-142)/(311-142)>0)&&((x-315)/(358-315)-(y-74)/(238-74)<0)&&(k<zmax)&&(k>zmin))
				//if((x>=x1)&&(x<=x2)&&(y>=y1)&&(y<=y4)&&(k<zmax)&&(k>=zmin))	
		//	if(((x-x1)/(x2-x1)-(y-y1)/(y2-y1))>0&&((x-x4)/(x3-x4)-(y-y4)/(y3-y4)<0)&&((x-x1)/(x4-x1)-(y-y1)/(y4-y1)>0)&&((x-x2)/(x3-x2)-(y-y2)/(y3-y2)<0)&&(k<zmax)&&(k>zmin))
			   //	if((i>xmin)&&(i<xmax)&&(j>ymin)&&(j<ymax)&&(k<zmax)&&(k>zmin))
					{ 
						
						if(img[k*imageX*imageY+j*imageX+i]>im_max)img[k*imageX*imageY+j*imageX+i]=im_max;
						if(img[k*imageX*imageY+j*imageX+i]<im_min)img[k*imageX*imageY+j*imageX+i]=im_min;
					}
					else img[k*imageX*imageY+j*imageX+i]=0;
			}
}
void img_add(float* out,float* in1,float* in2,float tmp,long size)
{
	long i;
	for(i=0;i<size;i++)
		out[i]=in1[i]+tmp*in2[i];
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
void zhuanzhi(float* img1,float * img2)
{
	int k,i,j;
	for(k=0;k<ztop-zbot;k++)
		for(j=0;j<imageY;j++)
			for(i=0;i<imageX;i++)
				img1[k*imageY*imageX+j*imageX+i]=img2[k*imageY*imageX+(imageY-1-j)*imageX+i];
}