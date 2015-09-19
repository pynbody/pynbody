#ifndef ILP_H
#define ILP_H
#include <stdint.h>

#ifndef ILP_PART_PER_TILE
#define ILP_PART_PER_TILE 4096 /* 4096*24 ~ 100k */
#endif

#define ILP_ALIGN_BITS 2
#define ILP_ALIGN_SIZE (1<<ILP_ALIGN_BITS)
#define ILP_ALIGN_MASK (ILP_ALIGN_SIZE-1)

typedef struct ilPart {
    double m,x,y,z;
    double fourh2;
    uint64_t iOrder;
    } ILP;

#if 0
/*
** We use a union here so that the compiler can properly align the values.
*/
typedef union {
    float f[ILP_PART_PER_TILE];
    } ilpFloat;


typedef union {
    int64_t i[ILP_PART_PER_TILE]; /* Signed because negative marks softened cells */
    } ilpInt64;


typedef struct ilpTile {
    struct ilpTile *next;
    struct ilpTile *prev;
    uint32_t nMaxPart;          /* Maximum number of particles in this tile */
    uint32_t nPart;             /* Current number of particles */

    ilpFloat dx, dy, dz;        /* Offset from ilp->cx, cy, cz */
    ilpFloat d2;                /* Distance squared: calculated */
    ilpFloat m;                 /* Mass */
    ilpFloat fourh2;            /* Softening: calculated */
/* #ifdef HERMITE */
    ilpFloat vx, vy, vz;
/* #endif */
/* #if defined(SYMBA) || defined(PLANETS) */
    ilpInt64 iOrder;
/* #endif */
    } *ILPTILE;

typedef struct ilpContext {
    ILPTILE first;              /* first tile in the chain */
    ILPTILE tile;               /* Current tile in the chain */
    double cx, cy, cz;          /* Center coordinates */
    uint32_t nPrevious;         /* Particles in tiles prior to "tile" */
    } *ILP;

typedef struct {
    ILPTILE  tile;
    uint32_t nPart;
    uint32_t nPrevious;
    } ILPCHECKPT;

ILPTILE ilpExtend(ILP ilp);    /* Add tile and return new tile */
ILPTILE ilpClear(ILP ilp);     /* Go back to, and return first tile (empty) */
void ilpInitialize(ILP *ilp);
void ilpFinish(ILP ilp);
size_t ilpMemory(ILP ilp);

static inline void ilpCheckPt(ILP ilp,ILPCHECKPT *cp) {
    cp->tile = ilp->tile;
    cp->nPart = ilp->tile->nPart;
    cp->nPrevious = ilp->nPrevious;
    }

static inline void ilpRestore(ILP ilp,ILPCHECKPT *cp) {
    ilp->tile = cp->tile;
    ilp->nPrevious = cp->nPrevious;
    ilp->tile->nPart = cp->nPart;
    }

static inline uint32_t ilpCount(ILP ilp) {
    return ilp->nPrevious + ilp->tile->nPart;
    }

/* #if defined(SYMBA) || defined(PLANETS) */
#define ilpAppend_1(ilp,I) tile->iOrder.i[ILP_APPEND_i] = (I);
/* #else */
/* #define ilpAppend_1(ilp,I) */
/* #endif */

/* #if defined(HERMITE) */
#define ilpAppend_2(ilp,VX,VY,VZ)					\
    tile->vx.f[ILP_APPEND_i] = (VX);					\
    tile->vy.f[ILP_APPEND_i] = (VY);					\
    tile->vz.f[ILP_APPEND_i] = (VZ);
/* #else */
/* #define ilpAppend_2(ilp,VX,VY,VZ) */
/* #endif */


#define ilpAppend(ilp,X,Y,Z,M,S,I)				\
    {									\
    ILPTILE tile = (ilp)->tile;						\
    uint_fast32_t ILP_APPEND_i;						\
    if ( tile->nPart == tile->nMaxPart ) tile = ilpExtend((ilp));	\
    ILP_APPEND_i = tile->nPart;						\
    tile->dx.f[ILP_APPEND_i] = (ilp)->cx - (X);			\
    tile->dy.f[ILP_APPEND_i] = (ilp)->cy - (Y);			\
    tile->dz.f[ILP_APPEND_i] = (ilp)->cz - (Z);			\
    assert( (M) > 0.0 );						\
    tile->m.f[ILP_APPEND_i] = (M);					\
    tile->fourh2.f[ILP_APPEND_i] = (S);				\
    ilpAppend_1((ilp),I);						\
    ++tile->nPart;							\
    }

#define ILP_LOOP(ilp,ptile) for( ptile=(ilp)->first; ptile!=(ilp)->tile->next; ptile=ptile->next )



static inline void ilpCompute(ILP ilp, float fx, float fy, float fz ) {
    ILPTILE tile;
    uint32_t j;

    ILP_LOOP(ilp,tile) {
	for (j=0;j<tile->nPart;++j) {
	    tile->dx.f[j] += fx;
	    tile->dy.f[j] += fy;
	    tile->dz.f[j] += fz;
	    tile->d2.f[j] = tile->dx.f[j]*tile->dx.f[j]
			      + tile->dy.f[j]*tile->dy.f[j] + tile->dz.f[j]*tile->dz.f[j];
	    }
	}
    }

#endif
#endif
