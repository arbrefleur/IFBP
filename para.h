#define MAX_THREADNUM_PERBLOCK 1024
#define  BLOCKSIZE 32
#define  BLOCKNUM 65535
#define pi 3.1415926535  
#define FOD   2000      //射线焦点到物体中心的距离  3300//
#define ODD   0  //物体中心到探测器的距离1536//
#define  DIS 2000
#define  DIST0 2000
#define DetectZ 400  //探测器的高度768//
#define DetectX 400 //探测器的宽度960//
#define imageY  400  //重建图像长度250//
#define imageX  400  //重建图像宽度250//
#define imageN  400   //重建的图像大小250//
#define  CX 0
#define CZ 0
#define NumCount 10  //迭代次数
#define lmta   0.2   //迭代因子
#define  frameN 360
#define  N0  frameN/4
#define error_SX (0)
#define error_SY (0)
#define error_DX (0)
#define error_DY (0)
#define zbot (-20)   //重建图像最下层29//
#define ztop  20   //重建图像最上层30//

#define Angle 105//逆时针倾斜角度15//
#define alpha 15
#define  z_dir 1
/*
#define MAX_THREADNUM_PERBLOCK 1024
#define  BLOCKSIZE 32
#define  BLOCKNUM 65535
#define pi 3.1415926535  
#define FOD   2401      //射线焦点到物体中心的距离  3300//
#define ODD   768  //物体中心到探测器的距离1536//
#define  DIS 3106
#define  DIST0 3106
#define DetectZ 384//400  //探测器的高度768//
#define DetectX 480//400 //探测器的宽度960//
#define imageY  300//400  //重建图像长度250//
#define imageX  300//400  //重建图像宽度250//
#define imageN  300//400   //重建的图像大小250//
//#define  CX 15.5//0
#define  CX 14.5//0
#define CZ 0
#define NumCount 10  //迭代次数
#define lmta   0.2   //迭代因子
#define  frameN 360
#define  N0  frameN/4

#define zbot (0)// (-20)   //重建图像最下层29//
#define ztop  30   //重建图像最上层30//

//#define Angle 105//逆时针倾斜角度15//
//#define alpha 15
#define Angle 102.7//逆时针倾斜角度15//
#define alpha 12.7
#define  z_dir 0
*/