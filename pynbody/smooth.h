#ifndef SMOOTH_HINCLUDED
#define SMOOTH_HINCLUDED

#include <stdbool.h>
#include "kd.h"

#define RESMOOTH_SAFE  500

#define M_1_PI  0.31830988618379067154

typedef struct pqNode {
	float fKey;
	struct pqNode *pqLoser;
	struct pqNode *pqFromInt;
	struct pqNode *pqFromExt;
	struct pqNode *pqWinner;	/* Only used when building initial tree */
	int p;
	float ax;
	float ay;
	float az;
	} PQ;


typedef struct smContext {
	KD kd;
	int nSmooth;
	float fPeriod[3];
	PQ *pq;
	PQ *pqHead;
	float *pfBall2;
	char *iMark;
	int nListSize;
	float *fList;
	int *pList;

    int pin,pi,pNext;
    float ax,ay,az;
    bool warnings; // added by AP to keep track of whether a memory-overrun  warning has been issued
	} * SMX;


#define PQ_INIT(pq,n)\
{\
	int PQ_j;\
	if ((n) == 1) {\
		(pq)[0].pqFromInt = NULL;\
		(pq)[0].pqFromExt = NULL;\
		}\
	for (PQ_j=0;PQ_j<(n);++PQ_j) {\
		if (PQ_j < 2) (pq)[PQ_j].pqFromInt = NULL;\
		else (pq)[PQ_j].pqFromInt = &(pq)[PQ_j>>1];\
		(pq)[PQ_j].pqFromExt = &(pq)[(PQ_j+(n))>>1];\
		}\
	}


#define PQ_BUILD(pq,n,q)\
{\
	int PQ_i,PQ_j;PQ *PQ_t,*PQ_lt;\
	for (PQ_j=(n)-1;PQ_j>0;--PQ_j) {\
		PQ_i = (PQ_j<<1);\
		if (PQ_i < (n)) PQ_t = (pq)[PQ_i].pqWinner;\
		else PQ_t = &(pq)[PQ_i-(n)];\
		++PQ_i;\
		if (PQ_i < (n)) PQ_lt = (pq)[PQ_i].pqWinner;\
		else PQ_lt = &(pq)[PQ_i-(n)];\
		if (PQ_t->fKey < PQ_lt->fKey) {\
			(pq)[PQ_j].pqLoser = PQ_t;\
			(pq)[PQ_j].pqWinner = PQ_lt;\
			}\
		else {\
			(pq)[PQ_j].pqLoser = PQ_lt;\
			(pq)[PQ_j].pqWinner = PQ_t;\
			}\
		}\
        if ((n) == 1) (q) = (pq);\
	else (q) = (pq)[1].pqWinner;\
	}

#define PQ_REPLACE(q)\
{\
	PQ *PQ_t,*PQ_lt;\
	PQ_t = (q)->pqFromExt;\
	while (PQ_t) {\
		if (PQ_t->pqLoser->fKey > (q)->fKey) {\
			PQ_lt = PQ_t->pqLoser;\
			PQ_t->pqLoser = (q);\
			(q) = PQ_lt;\
			}\
		PQ_t = PQ_t->pqFromInt;\
		}\
	}

double M3(double);
double dM3(double);
double F3(double);
double dF3(double);
double K3(double);
double dK3(double);

int smInit(SMX *,KD,int,float *);
void smFinish(SMX);
void smBallSearch(SMX,float,float *);
int  smBallGather(SMX,float,float *);

int smSmoothStep(SMX smx,void (*fncSmooth)(SMX,int,int,int *,float *));
void smSmoothInitStep(SMX smx);
void smDensitySym(SMX,int,int,int *,float *);
void smMeanVel(SMX,int,int,int *,float *);
void smVelDisp(SMX,int,int,int *,float *);
void smMeanVelSym(SMX,int,int,int *,float *);
void smDivvSym(SMX,int,int,int *,float *);
void smVelDispSym(SMX,int,int,int *,float *);
void smVelDispNBSym(SMX,int,int,int *,float *);


#endif
