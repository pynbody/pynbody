#ifndef KD_HINCLUDED
#define KD_HINCLUDED

#define ROOT		1
#define LOWER(i)	(i<<1)
#define UPPER(i)	((i<<1)+1)
#define PARENT(i)	(i>>1)
#define SIBLING(i) 	((i&1)?i-1:i+1)
#define SETNEXT(i)\
{\
	while (i&1) i=i>>1;\
	++i;\
	}

#define DARK	1
#define GAS		2
#define STAR	4

typedef struct Particle {
	float r[3];
	float v[3];
	float fMass;
	int iOrder;
	int iMark;
	float fDensity;
	float vMean[3];
	float fVel2;
	float fDivv;
	float fSmooth;
	} PARTICLE;

typedef struct bndBound {
	float fMin[3];
	float fMax[3];
	} BND;

typedef struct kdNode {
	float fSplit;
	BND bnd;
	int iDim;
	int pLower;
	int pUpper;
	} KDN;

typedef struct kdContext {
	int nBucket;
	int nParticles;
	int nDark;
	int nGas;
	int nStar;
	int bDark;
	int bGas;
	int bStar;
	int nActive;
	float fTime;
	int nLevels;
	int nNodes;
	int nSplit;
	PARTICLE *p;
	KDN *kdNodes;
	int uSecond;
	int uMicro;
	} * KD;


#define INTERSECT(c,cp,fBall2,lx,ly,lz,x,y,z,sx,sy,sz)\
{\
	float INTRSCT_dx,INTRSCT_dy,INTRSCT_dz;\
	float INTRSCT_dx1,INTRSCT_dy1,INTRSCT_dz1,INTRSCT_fDist2;\
	INTRSCT_dx = c[cp].bnd.fMin[0]-x;\
	INTRSCT_dx1 = x-c[cp].bnd.fMax[0];\
	INTRSCT_dy = c[cp].bnd.fMin[1]-y;\
	INTRSCT_dy1 = y-c[cp].bnd.fMax[1];\
	INTRSCT_dz = c[cp].bnd.fMin[2]-z;\
	INTRSCT_dz1 = z-c[cp].bnd.fMax[2];\
	if (INTRSCT_dx > 0.0) {\
		INTRSCT_dx1 += lx;\
		if (INTRSCT_dx1 < INTRSCT_dx) {\
			INTRSCT_fDist2 = INTRSCT_dx1*INTRSCT_dx1;\
			sx = x+lx;\
			}\
		else {\
			INTRSCT_fDist2 = INTRSCT_dx*INTRSCT_dx;\
			sx = x;\
			}\
		if (INTRSCT_fDist2 > fBall2) goto GetNextCell;\
		}\
	else if (INTRSCT_dx1 > 0.0) {\
		INTRSCT_dx += lx;\
		if (INTRSCT_dx < INTRSCT_dx1) {\
			INTRSCT_fDist2 = INTRSCT_dx*INTRSCT_dx;\
			sx = x-lx;\
			}\
		else {\
			INTRSCT_fDist2 = INTRSCT_dx1*INTRSCT_dx1;\
			sx = x;\
			}\
		if (INTRSCT_fDist2 > fBall2) goto GetNextCell;\
		}\
	else {\
		INTRSCT_fDist2 = 0.0;\
		sx = x;\
		}\
	if (INTRSCT_dy > 0.0) {\
		INTRSCT_dy1 += ly;\
		if (INTRSCT_dy1 < INTRSCT_dy) {\
			INTRSCT_fDist2 += INTRSCT_dy1*INTRSCT_dy1;\
			sy = y+ly;\
			}\
		else {\
			INTRSCT_fDist2 += INTRSCT_dy*INTRSCT_dy;\
			sy = y;\
			}\
		if (INTRSCT_fDist2 > fBall2) goto GetNextCell;\
		}\
	else if (INTRSCT_dy1 > 0.0) {\
		INTRSCT_dy += ly;\
		if (INTRSCT_dy < INTRSCT_dy1) {\
			INTRSCT_fDist2 += INTRSCT_dy*INTRSCT_dy;\
			sy = y-ly;\
			}\
		else {\
			INTRSCT_fDist2 += INTRSCT_dy1*INTRSCT_dy1;\
			sy = y;\
			}\
		if (INTRSCT_fDist2 > fBall2) goto GetNextCell;\
		}\
	else {\
		sy = y;\
		}\
	if (INTRSCT_dz > 0.0) {\
		INTRSCT_dz1 += lz;\
		if (INTRSCT_dz1 < INTRSCT_dz) {\
			INTRSCT_fDist2 += INTRSCT_dz1*INTRSCT_dz1;\
			sz = z+lz;\
			}\
		else {\
			INTRSCT_fDist2 += INTRSCT_dz*INTRSCT_dz;\
			sz = z;\
			}\
		if (INTRSCT_fDist2 > fBall2) goto GetNextCell;\
		}\
	else if (INTRSCT_dz1 > 0.0) {\
		INTRSCT_dz += lz;\
		if (INTRSCT_dz < INTRSCT_dz1) {\
			INTRSCT_fDist2 += INTRSCT_dz*INTRSCT_dz;\
			sz = z-lz;\
			}\
		else {\
			INTRSCT_fDist2 += INTRSCT_dz1*INTRSCT_dz1;\
			sz = z;\
			}\
		if (INTRSCT_fDist2 > fBall2) goto GetNextCell;\
		}\
	else {\
		sz = z;\
		}\
	}


void kdTime(KD,int *,int *);
int kdInit(KD *,int);
int kdReadTipsy(KD,FILE *,int,int,int,int);
void kdInMark(KD,char *);
void kdBuildTree(KD);
void kdOrder(KD);
void kdFinish(KD);

#endif











