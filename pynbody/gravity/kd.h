/*
 * Gravity code stolen from Zurich group's (Joachim Stadel, Doug Potter,
 * Jonathan Coles, et al) amazing gravity code pkdgrav2 and severly
 * bastardized and converted from parallel to serial for pynbody by
 * Greg Stinson with help from Jonathan, Tom Quinn, Rok Roskar and
 * Andrew Pontzen.
 */


#ifndef KD_HINCLUDED
#define KD_HINCLUDED

#include <stdint.h>
#include <string.h>

#define HAVE_CONFIG_H
#ifndef HAVE_CONFIG_H
#include "floattype.h"
#endif
#include "ilp.h"
#include "ilc.h"
#include "moments.h"
#include "config.h"

typedef uint_fast32_t local_t; /* Count of particles locally (per processor) */
typedef uint_fast64_t total_t; /* Count of particles globally (total number) */

static inline int d2i(double d)  {
    return (int)d;
}

static inline int64_t d2u64(double d) {
    return (uint64_t)d;
}

/*
** Handy type punning macro.
*/
#define UNION_CAST(x, sourceType, destType) \
	(((union {sourceType a; destType b;})x).b)
/*
** The following sort of definition should really be in a global
** configuration header file -- someday...
*/

#define CID_PARTICLE	0
#define CID_CELL	1
#define CID_GROUP	2
#define CID_RM		3
#define CID_BIN		4
#define CID_SHAPES	5
#define CID_PK          2
#define CID_PNG         2

/*
** These macros implement the index manipulation tricks we use for moving
** around in the top-tree. Note: these do NOT apply to the local-tree!
*/
#define LOWER(i)	(i<<1)
#define UPPER(i)	((i<<1)+1)
#define SIBLING(i) 	(i^1)
#define PARENT(i)	(i>>1)
#define SETNEXT(i)\
{\
	while (i&1) i=i>>1;\
	++i;\
	}

#define MAX_TIMERS		10

/*
** Memory models.  Each is a bit mask that indicates that additional fields should be
** added to the particle structure.
*/
#define KD_MODEL_VELOCITY     (1<<0)  /* Velocity Required */
#define KD_MODEL_ACCELERATION (1<<1)  /* Acceleration Required */
#define KD_MODEL_POTENTIAL    (1<<2)  /* Potential Required */
#define KD_MODEL_GROUPS       (1<<3)  /* Group profiling */
#define KD_MODEL_HERMITE      (1<<4)  /* Hermite integrator */
#define KD_MODEL_RELAXATION   (1<<5)  /* Trace relaxation */
#define KD_MODEL_MASS         (1<<6)  /* Mass for each particle */
#define KD_MODEL_SOFTENING    (1<<7)  /* Softening for each particle */
#define KD_MODEL_VELSMOOTH    (1<<8)  /* Velocity Smoothing */
#define KD_MODEL_SPH          (1<<9)  /* Sph Fields */
#define KD_MODEL_STAR         (1<<10) /* Star Fields */
#define KD_MODEL_PSMETRIC     (1<<11) /* Phase-space metric*/

#define KD_MODEL_NODE_MOMENT  (1<<24) /* Include moment in the tree */
#define KD_MODEL_NODE_ACCEL   (1<<25) /* mean accel on cell (for grav step) */
#define KD_MODEL_NODE_VEL     (1<<26) /* center of mass velocity for cells */
#define KD_MODEL_NODE_SPHBNDS (1<<27) /* Include 3 extra bounds in tree */

#define KD_MODEL_NODE_BND     (1<<28) /* Include normal bounds in tree */
#define KD_MODEL_NODE_BND6    (1<<29) /* Include phase-space bounds in tree */


#define KD_MAX_CLASSES 256
#define MAX_RUNG     63

/*
** Here we define some special reserved nodes. Node-0 is a sentinel or null node, node-1
** is here defined as the ROOT of the local tree (or top tree), node-2 is unused and
*/
#define ROOT		1
#define NRESERVED_NODES MAX_RUNG+1

#define IORDERBITS 42    
#define IORDERMAX ((((uint64_t) 1)<<IORDERBITS)-1)

typedef struct particle {
    /*-----Base-Particle-Data----*/
    uint64_t iOrder     :  IORDERBITS;
    float fMass;
    float r[3];
    float a[3];
    float fPot;
    float fSoft;
    } PARTICLE;

typedef struct pLite {
    float r[3];
    int i;
    } PLITE;

typedef struct bndBound {
    double fCenter[3];
    double fMax[3];
    double size;
    } BND;

#define BND_COMBINE(b,b1,b2)\
{\
	int BND_COMBINE_j;\
	for (BND_COMBINE_j=0;BND_COMBINE_j<3;++BND_COMBINE_j) {\
		FLOAT BND_COMBINE_t1,BND_COMBINE_t2,BND_COMBINE_max,BND_COMBINE_min;\
		BND_COMBINE_t1 = (b1).fCenter[BND_COMBINE_j] + (b1).fMax[BND_COMBINE_j];\
		BND_COMBINE_t2 = (b2).fCenter[BND_COMBINE_j] + (b2).fMax[BND_COMBINE_j];\
		BND_COMBINE_max = (BND_COMBINE_t1 > BND_COMBINE_t2)?BND_COMBINE_t1:BND_COMBINE_t2;\
		BND_COMBINE_t1 = (b1).fCenter[BND_COMBINE_j] - (b1).fMax[BND_COMBINE_j];\
		BND_COMBINE_t2 = (b2).fCenter[BND_COMBINE_j] - (b2).fMax[BND_COMBINE_j];\
		BND_COMBINE_min = (BND_COMBINE_t1 < BND_COMBINE_t2)?BND_COMBINE_t1:BND_COMBINE_t2;\
		(b).fCenter[BND_COMBINE_j] = 0.5*(BND_COMBINE_max + BND_COMBINE_min);\
		(b).fMax[BND_COMBINE_j] = 0.5*(BND_COMBINE_max - BND_COMBINE_min);\
		}\
	}


typedef struct {
    double *fCenter;
    double *fMax;
    double *size;
    } pBND;

#define MINDIST(bnd,pos,min2) {\
    double BND_dMin;\
    int BND_j;\
    (min2) = 0;					\
    for (BND_j=0;BND_j<3;++BND_j) {\
	BND_dMin = fabs((bnd).fCenter[BND_j] - (pos)[BND_j]) - (bnd).fMax[BND_j]; \
	if (BND_dMin > 0) (min2) += BND_dMin*BND_dMin;			\
	}\
    }

/*
** General partition macro
** LT,LE: Compare less-than/less-than or equal
** ii,dj: Increment i and decrement j
** SWAP: Swap the i'th and j'th element
** LOWER,UPPER: comparison predicates
** e.g.,
** PARTICLE *pi = kdParticle(kd,i);
** PARTICLE *pj = kdParticle(kd,j);
**    PARTITION(pi<pj,pi<=pj,
**              pi=kdParticle(kd,++i),pj=kdParticle(kd,--j),
**              kdSwapParticle(kd,pi,pj),
**	        pi->r[d] >= fSplit,pj->r[d] < fSplit);
** When finished, the 'I' variable points to the first element of
** the upper partition (or one past the end).
** NOTE: Because this function supports tiled data structures,
**       the LT, followed by "if (LE)" needs to remain this way.
*/
#define PARTITION(LT,LE,INCI,DECJ,SWAP,LOWER,UPPER)	\
    {							\
    while ((LT) && (LOWER)) { INCI; }			\
    if ((LE) && (LOWER)) { INCI; }			\
    else {						\
	while ((LT) && (UPPER)) { DECJ; }		\
	while (LT) {					\
		    { SWAP; }				\
		    do { DECJ; } while (UPPER);	\
		    do { INCI; } while (LOWER);		\
	    }						\
	}						\
    }

#define NEW_STACK(S,inc) \
    int S ## __s=0, S ## __ns=inc, S ## __inc=inc; \
    do { S = malloc((S ## __ns)*sizeof(*S)); assert(S != NULL); } while (0)
#define FREE_STACK(S) do { free(S); } while (0)
#define PUSH(S,v) do { S[(S ## __s)++] = (v); } while (0)
#define POP(S) (S[--(S ## __s)])
#define STACK_EMPTY(S) (S ## __s == 0)
#define EXTEND_STACK(S) do { \
    if ( (S ## __s)+1 >= (S ## __ns) ) { \
        assert( (S ## __s)+1 == (S ## __ns) ); \
        (S ## __ns) += (S ## __inc); \
        S = realloc(S,(S ## __ns)*sizeof(*S)); \
        } \
} while (0)
#define CLEAR_STACK(S) do { S ## __s=0; } while (0)


typedef struct kdNode {
    double r[3];
    int iLower;
    int iParent;
    int pLower;
    int pUpper;
    float bMax;
    float fSoft2;
    } KDN;

#define NMAX_OPENCALC	1000

#define MAXSIDE(fMax,b) {\
    if ((fMax)[0] > (fMax)[1]) {\
	if ((fMax)[0] > (fMax)[2]) b = 2.0*(fMax)[0];\
	else b = 2.0*(fMax)[2];\
	}\
    else {\
	if ((fMax)[1] > (fMax)[2]) b = 2.0*(fMax)[1];\
	else b = 2.0*(fMax)[2];\
	}\
    }

#define MINSIDE(fMax,b) {\
    if ((fMax)[0] < (fMax)[1]) {\
	if ((fMax)[0] < (fMax)[2]) b = 2.0*(fMax)[0];\
	else b = 2.0*(fMax)[2];\
	}\
    else {\
	if ((fMax)[1] < (fMax)[2]) b = 2.0*(fMax)[1];\
	else b = 2.0*(fMax)[2];\
	}\
    }

#define CALCAXR(fMax,axr) {					\
    if ((fMax)[0] < (fMax)[1]) {				\
	if ((fMax)[1] < (fMax)[2]) {				\
	    if ((fMax)[0] > 0) axr = (fMax)[2]/(fMax)[0];	\
	    else axr = 1e6;					\
	}							\
	else if ((fMax)[0] < (fMax)[2]) {			\
	    if ((fMax)[0] > 0) axr = (fMax)[1]/(fMax)[0];	\
	    else axr = 1e6;					\
	}							\
	else if ((fMax)[2] > 0) axr = (fMax)[1]/(fMax)[2];	\
	else axr = 1e6;						\
    }								\
    else if ((fMax)[0] < (fMax)[2]) {				\
	if ((fMax)[1] > 0) axr = (fMax)[2]/(fMax)[1];		\
	else axr = 1e6;						\
    }								\
    else if ((fMax)[1] < (fMax)[2]) {				\
	if ((fMax)[1] > 0) axr = (fMax)[0]/(fMax)[1];		\
	else axr = 1e6;						\
    }								\
    else if ((fMax)[2] > 0) axr = (fMax)[0]/(fMax)[2];		\
    else axr = 1e6;						\
}


#define CALCOPEN(kdn,minside) {					\
        FLOAT CALCOPEN_d2 = 0;						\
	FLOAT CALCOPEN_b;						\
        int CALCOPEN_j;							\
	pBND CALCOPEN_bnd;						\
	kdNodeBnd(kd, kdn, &CALCOPEN_bnd);\
        for (CALCOPEN_j=0;CALCOPEN_j<3;++CALCOPEN_j) {                  \
            FLOAT CALCOPEN_d = fabs(CALCOPEN_bnd.fCenter[CALCOPEN_j] - (kdn)->r[CALCOPEN_j]) + \
                CALCOPEN_bnd.fMax[CALCOPEN_j];                          \
            CALCOPEN_d2 += CALCOPEN_d*CALCOPEN_d;                       \
            }								\
	MAXSIDE(CALCOPEN_bnd.fMax,CALCOPEN_b);				\
	if (CALCOPEN_b < minside) CALCOPEN_b = minside;			\
	if (CALCOPEN_b*CALCOPEN_b < CALCOPEN_d2) CALCOPEN_b = sqrt(CALCOPEN_d2); \
	(kdn)->bMax = CALCOPEN_b;					\
	}

/*
** Components required for tree walking.
*/
typedef struct CheckElt {
    int iCell;
    int id;
    double cOpen;
    double rOffset[3];
    } CELT;

typedef struct CheckStack {
    int nPart;
    int nCell;
    int nCheck;
    CELT *Check;
    FLOCR L;
    float fWeight;
    } CSTACK;

typedef struct ewaldTable {
    double hx,hy,hz;
    double hCfac,hSfac;
    } EWT;

struct EwaldVariables {
    double fEwCut2,fInner2,alpha,alpha2,k1,ka;
    double Q4xx,Q4xy,Q4xz,Q4yy,Q4yz,Q4zz,Q4,Q3x,Q3y,Q3z,Q2;
    EWT *ewt;
    int nMaxEwhLoop;
    int nEwhLoop;
    int nReps,nEwReps;
    };

typedef struct kdContext {
    double dTheta2;
    int nStore;
    int nRejects;
    int nTreeBitsLo;
    int nTreeBitsHi;
    int iTreeMask;
    int nTreeTiles;
    int nMaxNodes;
    uint64_t nDark;
    uint64_t nGas;
    uint64_t nStar;
    double fPeriod[3];
    char *kdTopPRIVATE; /* Because this is a variable size, we use a char pointer, not a KDN pointer! */
    char **kdNodeListPRIVATE; /* BEWARE: also char instead of KDN */
    int iTopRoot;
    int nNodes;
    int nNodesFull;     /* number of nodes in the full tree (including very active particles) */
    BND bnd;
    size_t iTreeNodeSize;
    size_t iParticleSize;
    PARTICLE *pStorePRIVATE;
    PARTICLE *pTempPRIVATE;
    float fSoftFix;
    float fSoftFac;
    float fSoftMax;
    int nClasses;
    PLITE *pLite;

    /*
    ** Advanced memory models - Tree Nodes
    */
    int oNodeMom; /* an FMOMR */
    int oNodeBnd;
    int oNodeBnd6;

    /*
    ** Tree walk variables.
    */
    int nMaxStack;
    CSTACK *S;
    ILP *ilp;
    ILC *ilc;
    CELT *Check;
    int nMaxPart, nMaxCell;
    int nMaxCheck;

    /*
    ** Opening angle table for mass weighting.
    */
#ifdef USE_DEHNEN_THETA
    float *fCritTheta;
    float *fCritMass;
    int nCritBins;
    float dCritLogDelta;
    float dCritThetaMin;
    float dCritThetaMax;
#else
    float fiCritTheta;
#endif

    /*
    ** New activation methods
    */
    uint8_t uMinRungActive;
    uint8_t uMaxRungActive;
    uint8_t uRungVeryActive;    /* NOTE: The first very active particle is at iRungVeryActive + 1 */

    /*
    ** Ewald summation setup.
    */
    MOMC momRoot;
    struct EwaldVariables ew;

    /*
    ** Timers stuff.
    */
    struct timer {
	double sec;
	double stamp;
	double system_sec;
	double system_stamp;
	double wallclock_sec;
	double wallclock_stamp;
	int iActive;
	} ti[MAX_TIMERS];
    } * KD;

static inline void kdMinMax( float *dVal, double *dMin, double *dMax ) {
    dMin[0] = dVal[0] < dMin[0] ? dVal[0] : dMin[0];
    dMin[1] = dVal[1] < dMin[1] ? dVal[1] : dMin[1];
    dMin[2] = dVal[2] < dMin[2] ? dVal[2] : dMin[2];
    dMax[0] = dVal[0] > dMax[0] ? dVal[0] : dMax[0];
    dMax[1] = dVal[1] > dMax[1] ? dVal[1] : dMax[1];
    dMax[2] = dVal[2] > dMax[2] ? dVal[2] : dMax[2];
    }

static inline void kdMinMax6( double *dVal0, double *dVal1, double *dMin, double *dMax ) {
    dMin[0] = dVal0[0] < dMin[0] ? dVal0[0] : dMin[0];
    dMin[1] = dVal0[1] < dMin[1] ? dVal0[1] : dMin[1];
    dMin[2] = dVal0[2] < dMin[2] ? dVal0[2] : dMin[2];
    dMin[3] = dVal1[0] < dMin[3] ? dVal1[0] : dMin[3];
    dMin[4] = dVal1[1] < dMin[4] ? dVal1[1] : dMin[4];
    dMin[5] = dVal1[2] < dMin[5] ? dVal1[2] : dMin[5];

    dMax[0] = dVal0[0] > dMax[0] ? dVal0[0] : dMax[0];
    dMax[1] = dVal0[1] > dMax[1] ? dVal0[1] : dMax[1];
    dMax[2] = dVal0[2] > dMax[2] ? dVal0[2] : dMax[2];
    dMax[3] = dVal1[0] > dMax[3] ? dVal1[0] : dMax[3];
    dMax[4] = dVal1[1] > dMax[4] ? dVal1[1] : dMax[4];
    dMax[5] = dVal1[2] > dMax[5] ? dVal1[2] : dMax[5];
    }

/*
** A tree node is of variable size.  The following routines are used to
** access individual fields.
*/
static inline KDN *kdTreeBase( KD kd ) {
    return (KDN *)kd->kdNodeListPRIVATE;
    }
static inline size_t kdNodeSize( KD kd ) {
    return kd->iTreeNodeSize;
    }
static inline size_t kdMaxNodeSize() {
    return sizeof(KDN) +  7*sizeof(double) + sizeof(FMOMR) + 6*sizeof(double);
    }
static inline void kdCopyNode(KD kd, KDN *a, KDN *b) {
    memcpy(a,b,kdNodeSize(kd));
    }
static inline void *kdNodeField( KDN *n, int iOffset ) {
    char *v = (char *)n;
    /*assert(iOffset);*/ /* Remove this for better performance */
    return (void *)(v + iOffset);
    }
static inline FMOMR *kdNodeMom(KD kd,KDN *n) {
    return kdNodeField(n,kd->oNodeMom);
    }

static inline void kdNodeBnd( KD kd, KDN *n, pBND *bnd ) {
    const int o = kd->oNodeBnd;
    const int e = 3*sizeof(double)*(1+(kd->oNodeBnd6!=0));
    bnd->fCenter = kdNodeField(n,o);
    bnd->fMax = kdNodeField(n,o+e);
    bnd->size = kdNodeField(n,o+e+e);
    }

static inline KDN *kdNode(KD kd,KDN *pBase,int iNode) {
    return (KDN *)&((char *)pBase)[kd->iTreeNodeSize*iNode];
    }
int kdNodes(KD kd);
void kdExtendTree(KD kd);
static inline KDN *kdTreeNode(KD kd,int iNode) {
    return (KDN *)&kd->kdNodeListPRIVATE[(iNode>>kd->nTreeBitsLo)][kd->iTreeNodeSize*(iNode&kd->iTreeMask)];
    }
void *kdTreeNodeGetElement(void *vData,int i,int iDataSize);
static inline KDN *kdTopNode(KD kd,int iNode) {
    return (KDN *)&kd->kdTopPRIVATE[kd->iTreeNodeSize*iNode];
    }

/*
** The size of a particle is variable based on the memory model.
** The following three routines must be used instead of accessing pStore
** directly.  kdParticle will return a pointer to the i'th particle.
** The Size and Base functions are intended for cache routines; no other
** code should care about sizes of the particle structure.
*/
static inline PARTICLE *kdParticleBase( KD kd ) {
    return kd->pStorePRIVATE;
    }
static inline size_t kdParticleSize( KD kd ) {
    return kd->iParticleSize;
    }
static inline size_t kdParticleMemory(KD kd) {
    return (kd->iParticleSize) * (kd->nStore+1);
    }
static inline PARTICLE *kdParticle(KD kd, int i) {
    return &(kd->pStorePRIVATE[i]);
    }
static inline void kdSaveParticle(KD kd, PARTICLE *a) {
    memcpy(kd->pTempPRIVATE,a,kdParticleSize(kd));
    }
static inline void kdLoadParticle(KD kd, PARTICLE *a) {
    memcpy(a,kd->pTempPRIVATE,kdParticleSize(kd));
    }
static inline void kdCopyParticle(KD kd, PARTICLE *a, PARTICLE *b) {
    memcpy(a,b,kdParticleSize(kd));
    }
static inline void kdSwapParticle(KD kd, PARTICLE *a, PARTICLE *b) {
    kdSaveParticle(kd,a);
    kdCopyParticle(kd,a,b);
    kdLoadParticle(kd,b);
    }

/*
** From tree.c:
*/
void kdTreeBuild(KD kd,int nBucket);
void kdCombineCells1(KD,KDN *kdn,KDN *p1,KDN *p2);
void kdCombineCells2(KD,KDN *kdn,KDN *p1,KDN *p2);
void kdDistribCells(KD,int,KDN *);
void kdCalcRoot(KD,MOMC *);
void kdDistribRoot(KD,MOMC *);

void kdGravInteract(KD kd,KDN *pBucket, ILP *ilp,int nPart,
		   ILC *ilc,int nCell, int bEwald, PARTICLE *p);
void kdGravWalk(KD kd, int nReps,int bEwald, PARTICLE *testParticles, int nPos);
double kdTime(void);

/*
** From kd.c:
*/
double kdGetTimer(KD,int);
double kdGetSystemTimer(KD,int);
double kdGetWallClockTimer(KD,int);
void kdClearTimer(KD,int);
void kdStartTimer(KD,int);
void kdStopTimer(KD,int);
void kdInitialize( KD kd,int nStore,int nBucket,double dTheta,int nTreeBitsLo,
		   int nTreeBitsHi, float *fPeriod);
KDN * kdNodeInit(KD kd, PARTICLE *p);
void kdFinish(KD);
size_t kdClCount(KD kd);
size_t kdClMemory(KD kd);
size_t kdIlcMemory(KD kd);
size_t kdIlpMemory(KD kd);
size_t kdTreeMemory(KD kd);

void kdSetSoft(KD kd,double dSoft);
void kdSetCrit(KD kd,double dCrit);
void kdCalcBound(KD,BND *);
void kdEnforcePeriodic(KD,BND *);
void kdPhysicalSoft(KD kd,double dSoftMax,double dFac,int bSoftMaxMul);

void kdBucketWeight(KD kd,int iBucket,FLOAT fWeight);
int kdWeight(KD,int,FLOAT,int,int,int,int *,int *,FLOAT *,FLOAT *);
double kdTotalMass(KD kd);
int kdLowerPart(KD,int,FLOAT,int,int);
int kdUpperPart(KD,int,FLOAT,int,int);
int kdWeightWrap(KD,int,FLOAT,FLOAT,int,int,int,int,int *,int *);
int kdLowerPartWrap(KD,int,FLOAT,FLOAT,int,int,int);
int kdUpperPartWrap(KD,int,FLOAT,FLOAT,int,int,int);
int kdLowerOrdPart(KD,uint64_t,int,int);
int kdUpperOrdPart(KD,uint64_t,int,int);
int kdActiveOrder(KD);

int kdColRejects(KD,int);
int kdColRejects_Old(KD,int,FLOAT,FLOAT,int);

int kdSwapRejects(KD,int);
int kdSwapSpace(KD);
int kdFreeStore(KD);
int kdLocal(KD);

int kdColOrdRejects(KD,uint64_t,int);
void kdLocalOrder(KD);

void kdCalcCOM(KD kd, double *dCenter, double dRadius,
		double *com, double *vcm, double *L,
		double *M, uint64_t *N);
void kdGridInitialize(KD kd, int n1, int n2, int n3, int a1, int s, int n);
void kdGridProject(KD kd);

static inline double softmassweight(double m1,double h12,double m2,double h22) {
    double tmp = h12*h22;
    if (m1 == 0.0) return(h22);
    if (m2 == 0.0) return(h12);
    if (tmp > 0.0) return((m1+m2)*tmp/(h22*m1+h12*m2));
    else return(0.0);
    }

static inline void vec_sub(double *r,const double *a,const double *b ) {
    int i;
    for (i=0; i<3; i++) r[i] = a[i] - b[i];
}

static inline void vec_add_const_mult(double *r,const double *a,double c,const double *b) {
    int i;
    for (i=0; i<3; i++) r[i] = a[i] + c * b[i];
}

static inline void matrix_vector_mult(double *b,double mat[3][3], const double *a) {
    int i,j ;
    for (i=0; i<3; i++) {
        b[i] = 0.0;
        for (j=0; j<3; j++) b[i] += mat[i][j] * a[j];
    }
}

static inline double dot_product(const double *a,const double *b) {
    int i;
    double r = 0.0;
    for(i=0; i<3; i++) r += a[i]*b[i];
    return r;
    }

static inline void cross_product(double *r,const double *a,const double *b) {
    r[0] = a[1] * b[2] - a[2] * b[1] ;
    r[1] = a[2] * b[0] - a[0] * b[2] ;
    r[2] = a[0] * b[1] - a[1] * b[0] ;
}

static inline void mat_transpose(double mat[3][3], double trans_mat[3][3]) {
    int i,j ;
    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            trans_mat[i][j] = mat[j][i];
	    }
	}
    }




#endif
