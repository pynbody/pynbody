#ifndef SMOOTH_HINCLUDED
#define SMOOTH_HINCLUDED

#include <functional>
#include <memory>
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
    std::unique_ptr<std::vector<size_t>> result;
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

inline int smBallGatherStoreResultInList(SMX smx, float fDist2, int particleIndex, int foundIndex) {
    // append particleIndex to particleList
    // PyObject *particleIndexPy = PyLong_FromLong(smx->kd->p[particleIndex].iOrder);
    // PyList_Append(smx->result, particleIndexPy);
    // tempar->push_back(smx->kd->p[particleIndex].iOrder);
    smx->result->push_back(smx->kd->p[particleIndex].iOrder);
    return particleIndex+1;
}

inline int smBallGatherStoreResultInSmx(SMX smx, float fDist2, int particleIndex, int foundIndex) {
    if(foundIndex>=smx->nListSize) {
        if(!smx->warnings) fprintf(stderr, "Smooth - particle cache too small for local density - results will be incorrect\n");
        smx->warnings=true;
        return foundIndex;
      }
    smx->fList[foundIndex] = fDist2;
    smx->pList[foundIndex] = particleIndex;
    return foundIndex+1;
}


template<typename T, int (*storeResultFunction)(SMX, float, int, int) >
int smBallGather(SMX smx, float fBall2, float *ri)
{
	KDN *c;
	PARTICLE *p;
	KD kd=smx->kd;
	int pj,nCnt,cp,nSplit;
	float dx,dy,dz,x,y,z,lx,ly,lz,sx,sy,sz,fDist2;

	c = smx->kd->kdNodes;
	p = smx->kd->p;
	nSplit = smx->kd->nSplit;
	lx = smx->fPeriod[0];
	ly = smx->fPeriod[1];
	lz = smx->fPeriod[2];
	x = ri[0];
	y = ri[1];
	z = ri[2];

	nCnt = 0;
	cp = ROOT;
	while (1) {
		INTERSECT(c,cp,fBall2,lx,ly,lz,x,y,z,sx,sy,sz);
		/*
		 ** We have an intersection to test.
		 */
		if (cp < nSplit) {
			cp = LOWER(cp);
			continue;
			}
		else {
			for (pj=c[cp].pLower;pj<=c[cp].pUpper;++pj) {
				dx = sx - GET2<T>(kd->pNumpyPos,p[pj].iOrder,0);
				dy = sy - GET2<T>(kd->pNumpyPos,p[pj].iOrder,1);
				dz = sz - GET2<T>(kd->pNumpyPos,p[pj].iOrder,2);
				fDist2 = dx*dx + dy*dy + dz*dz;
				if (fDist2 <= fBall2) {
				    nCnt = storeResultFunction(smx, fDist2, pj, nCnt);
				}
			}

        }
	GetNextCell:
	    // called by INTERSECT when a cell can be ignored, and finds the next cell to inspect
		SETNEXT(cp,ROOT);
		if (cp == ROOT) break;
	}
	assert(nCnt <= smx->nListSize);
	return(nCnt);
	}



int smBallGatherStoreResultInSmx(SMX smx, float, int, int);

int smBallGatherStoreResultInList(SMX smx, float, int, int);

void initParticleList(SMX smx);

PyObject *getReturnParticleList(SMX smx);


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
