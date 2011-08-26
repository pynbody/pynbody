#ifndef CL_H
#define CL_H
#include <stdint.h>

#ifndef CL_PART_PER_TILE
#define CL_PART_PER_TILE 1024 /* 1024*100 ~ 100k */
#endif

#define CL_ALIGN_BITS 2
#define CL_ALIGN_SIZE (1<<CL_ALIGN_BITS)
#define CL_ALIGN_MASK (CL_ALIGN_SIZE-1)


/*
** We use a union here so that the compiler can properly align the values.
*/
typedef union {
    float f[CL_PART_PER_TILE];
    } clFloat;

typedef union {
    int32_t i[CL_PART_PER_TILE];
    } clInt32;

typedef union {
    uint64_t i[CL_PART_PER_TILE];
    } clInt64;


typedef struct clTile {
    struct clTile *next;
    struct clTile *prev;
    uint32_t nMaxItems;          /* Maximum number of items in this tile */
    uint32_t nItems;             /* Current number of items */

    clInt32 iOpen;
    clInt32 iCell;
    clInt32 id;
    clInt32 iLower;
    clInt32 nc;
    clFloat cOpen;
    clFloat m;
    clFloat fourh2;
    clFloat x;
    clFloat y;
    clFloat z;
    clFloat xOffset;
    clFloat yOffset;
    clFloat zOffset;
    clFloat xCenter;
    clFloat yCenter;
    clFloat zCenter;
    clFloat xMax;
    clFloat yMax;
    clFloat zMax;
/*
** ILP has some extra data here that gets partitioned to find the closest
** particles. ILC doesn't need this (at least not yet).
*/
    } *CLTILE;

typedef struct clContext {
    CLTILE first;               /* first tile in the chain */
    CLTILE tile;                /* Current tile in the chain */
    double cx, cy, cz;          /* Center coordinates */
    uint32_t nPrevious;         /* Particles in tiles prior to "tile" */
    } *CL;

CLTILE clExtend(CL cl);         /* Add tile and return new tile */
CLTILE clClear(CL cl);          /* Go back to, and return first tile (empty) */
CLTILE clClone(CL cl, CL src);  /* Make "cl" the same as "src" */
void clInitialize(CL *cl);
void clFinish(CL cl);
size_t clMemory(CL cl);

static inline uint32_t clCount(CL cl) {
    return cl->nPrevious + cl->tile->nItems;
    }

static inline void clAppendAll(CL cl,int iCell, int id,int iLower,int nc,
    float cOpen,float m,float fourh2,float x, float y, float z,
    float xOffset,float yOffset,float zOffset,float xCenter,float yCenter,float zCenter,
    float xMax,float yMax,float zMax,int iOpen) {
    CLTILE tile = (cl)->tile;
    uint_fast32_t i;
    if ( tile->nItems == tile->nMaxItems ) tile = clExtend((cl));
    i = tile->nItems;
    tile->iOpen.i[i] = iOpen;
    tile->iCell.i[i] = (iCell);
    tile->id.i[i] = (id);
    tile->iLower.i[i] = (iLower);
    tile->nc.i[i] = (nc);
    tile->cOpen.f[i] = (cOpen);
    tile->m.f[i] = (m);
    tile->fourh2.f[i] = (fourh2);
    tile->x.f[i] = x;
    tile->y.f[i] = y;
    tile->z.f[i] = z;
    tile->xOffset.f[i] = xOffset;
    tile->yOffset.f[i] = yOffset;
    tile->zOffset.f[i] = zOffset;
    tile->xCenter.f[i] = xCenter;
    tile->yCenter.f[i] = yCenter;
    tile->zCenter.f[i] = zCenter;
    tile->xMax.f[i] = xMax;
    tile->yMax.f[i] = yMax;
    tile->zMax.f[i] = zMax;
    ++tile->nItems;
    }

#define clAppend(cl,iCell,id,iLower,nc,cOpen,m,fourh2,r,fOffset,fCenter,fMax)\
    clAppendAll(cl,iCell,id,iLower,nc,cOpen,m,fourh2,r[0],r[1],r[2],\
    fOffset[0],fOffset[1],fOffset[2],fCenter[0],fCenter[1],fCenter[2],\
    fMax[0],fMax[1],fMax[2],0)

static inline void clCopyItem(CLTILE A, int Ai, CLTILE B, int Bi) {
    A->iOpen.i[Ai]   = B->iOpen.i[Bi];
    A->iCell.i[Ai]   = B->iCell.i[Bi];
    A->id.i[Ai]      = B->id.i[Bi];
    A->iLower.i[Ai]  = B->iLower.i[Bi];
    A->nc.i[Ai]      = B->nc.i[Bi];
    A->cOpen.f[Ai]   = B->cOpen.f[Bi];
    A->m.f[Ai]       = B->m.f[Bi];
    A->fourh2.f[Ai]  = B->fourh2.f[Bi];
    A->x.f[Ai]       = B->x.f[Bi];
    A->y.f[Ai]       = B->y.f[Bi];
    A->z.f[Ai]       = B->z.f[Bi];
    A->xOffset.f[Ai] = B->xOffset.f[Bi];
    A->yOffset.f[Ai] = B->yOffset.f[Bi];
    A->zOffset.f[Ai] = B->zOffset.f[Bi];
    A->xCenter.f[Ai] = B->xCenter.f[Bi];
    A->yCenter.f[Ai] = B->yCenter.f[Bi];
    A->zCenter.f[Ai] = B->zCenter.f[Bi];
    A->xMax.f[Ai]    = B->xMax.f[Bi];
    A->yMax.f[Ai]    = B->yMax.f[Bi];
    A->zMax.f[Ai]    = B->zMax.f[Bi];
}

static inline void clAppendItem(CL cl, CLTILE B, int Bi) {
    clAppendAll(cl,B->iCell.i[Bi],B->id.i[Bi],B->iLower.i[Bi], B->nc.i[Bi], B->cOpen.f[Bi], B->m.f[Bi], B->fourh2.f[Bi],
	B->x.f[Bi], B->y.f[Bi], B->z.f[Bi],B->xOffset.f[Bi],B->yOffset.f[Bi],B->zOffset.f[Bi],
	B->xCenter.f[Bi],B->yCenter.f[Bi],B->zCenter.f[Bi],B->xMax.f[Bi],B->yMax.f[Bi],B->zMax.f[Bi],B->iOpen.i[Bi]);
    }

#define CL_LOOP(CL,CL_TILE) for( CL_TILE=(CL)->first; CL_TILE!=(CL)->tile->next; CL_TILE=CL_TILE->next )

#endif
