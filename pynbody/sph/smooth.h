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
	char *iMark;
	int nListSize;
	float *fList;
	int *pList;
	int nCurrent; // current particle index for distributed loops

#ifdef KDT_THREADING
	pthread_mutex_t *pMutex;

	int nLocals; // number of local copies if this is a global smooth context
	int nReady; // number of local copies that are "ready" for the next stage
	pthread_cond_t *pReady; // synchronizing condition

	struct smContext *smx_global;
#endif

    int pin,pi,pNext;
    float ax,ay,az;
    bool warnings; //  keep track of whether a memory-overrun  warning has been issued
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
void smInitPriorityQueue(SMX);
void smFinish(SMX);

template<typename T>
void smBallSearch(SMX,float,float *);

template<typename T>
int  smBallGather(SMX,float,float *);

template<typename T>
int smSmoothStep(SMX smx, int procid);

void smSmoothInitStep(SMX smx, int nProcs);

template<typename T>
void smDensitySym(SMX,int,int,int *,float *, bool);

template<typename T>
void smDensity(SMX,int,int,int *,float *, bool);

template<typename Tf, typename Tq>
void smMeanQtyND(SMX,int,int,int *,float *, bool);
template<typename Tf, typename Tq>
void smDispQtyND(SMX,int,int,int *,float *, bool);
template<typename Tf, typename Tq>
void smMeanQty1D(SMX,int,int,int *,float *, bool);
template<typename Tf, typename Tq>
void smDispQty1D(SMX,int,int,int *,float *, bool);
template<typename Tf, typename Tq>
void smDivQty(SMX,int,int,int *,float *, bool);
template<typename Tf, typename Tq>
void smCurlQty(SMX,int,int,int *,float *, bool);

bool smCheckFits(KD kd, float *fPeriod);

template<typename T>
T Wendland_kernel(SMX, T, int);

template<typename T>
T cubicSpline(SMX, T);

template<typename Tf>
Tf cubicSpline_gradient(Tf, Tf, Tf, Tf);

template<typename Tf>
Tf Wendland_gradient(Tf, Tf);

/*
void smMeanVel(SMX,int,int,int *,float *);
void smVelDisp(SMX,int,int,int *,float *);
void smMeanVelSym(SMX,int,int,int *,float *);
void smDivvSym(SMX,int,int,int *,float *);
void smVelDispSym(SMX,int,int,int *,float *);
void smVelDispNBSym(SMX,int,int,int *,float *);
*/

template<typename T>
void smDomainDecomposition(KD kd, int nprocs);

int smGetNext(SMX smx_local);

#ifdef KDT_THREADING
void smReset(SMX smx_local);
SMX smInitThreadLocalCopy(SMX smx_global);
void smFinishThreadLocalCopy(SMX smx_local);
#endif

#endif
