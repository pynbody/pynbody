#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "cl.h"

/* NOT THREAD SAFE -- FIX ME */
static CLTILE clFreeList = NULL;

size_t clMemory(CL cl) {
    size_t nBytes;
    CLTILE tile;
    if (cl==NULL) {
        nBytes = 0;
        for(tile=clFreeList;tile!=NULL;tile=tile->next)
	    nBytes += sizeof(struct clTile);
    }
    else {
        nBytes = sizeof(struct clContext);
        for(tile=cl->first;tile!=NULL;tile=tile->next)
	    nBytes += sizeof(struct clTile);
    }
    return nBytes;
    }

/*
** Private: Create a new tile
*/
static CLTILE newTile(CLTILE prev) {
    CLTILE tile;

    if (clFreeList) {
        tile = clFreeList;
	clFreeList = clFreeList->next;
    }
    else {
        tile = malloc(sizeof(struct clTile));
    }
    assert( tile != NULL );
    assert(CL_PART_PER_TILE%4 == 0 );

    tile->next = NULL;
    tile->prev = prev;
    tile->nMaxItems = CL_PART_PER_TILE;
    tile->nItems = 0;

    return tile;
    }

/*
** If the current tile is full (nItems == nMaxItems), then
** this function is called to get a new, empty tile.
*/
CLTILE clExtend(CL cl) {
    assert( cl != NULL );
    assert( cl->tile != NULL );
    assert( cl->first != NULL );
    assert( cl->tile->nItems == cl->tile->nMaxItems );

    cl->nPrevious += cl->tile->nItems;

    /* Use the next tile if it exists, or create a new one */
    if ( cl->tile->next != NULL ) {
	cl->tile = cl->tile->next;
	cl->tile->nItems = 0;
	}
    else {
	cl->tile = cl->tile->next = newTile(cl->tile);
	}

    return cl->tile;
    }

/*
** Empty the list of particles (go back to the first tile)
*/
CLTILE clClear(CL cl) {
    assert( cl != NULL );
    if (cl->tile!=cl->first) {
        cl->tile->next = clFreeList;
        clFreeList = cl->first->next;
        cl->first->next = NULL;
    }
    cl->tile = cl->first;
    cl->nPrevious = 0;
    assert( cl->tile != NULL );
    cl->tile->nItems = 0;
    return cl->tile;
    }

void clInitialize(CL *cl) {
    *cl = malloc(sizeof(struct clContext));
    assert( *cl != NULL );
    (*cl)->first = (*cl)->tile = newTile(NULL);
    (*cl)->nPrevious = 0;
    }

void clFinish(CL cl) {
    CLTILE tile, next;

    assert( cl != NULL );

    /* Free all allocated tiles first */
    for ( tile=cl->first; tile!=NULL; tile=next ) {
	next = tile->next;
	free(tile);
	}

    free(cl);
    }

CLTILE clClone(CL cl,CL src) {
    CLTILE tile, newtile;
    int n;

    clClear(cl);
    CL_LOOP(src,tile) {
	newtile = cl->tile;
	if (newtile->nItems) newtile=clExtend(cl);
	newtile->nMaxItems = tile->nMaxItems;
	n = newtile->nItems = tile->nItems;
	memcpy(&newtile->iCell.i,&tile->iCell.i,n*sizeof(tile->iCell.i[0]));
	memcpy(&newtile->id.i,&tile->id.i,n*sizeof(tile->id.i[0]));
	memcpy(&newtile->iLower.i,&tile->iLower.i,n*sizeof(tile->iLower.i[0]));
	memcpy(&newtile->nc.i,&tile->nc.i,n*sizeof(tile->nc.i[0]));
	memcpy(&newtile->cOpen.f,&tile->cOpen.f,n*sizeof(tile->cOpen.f[0]));
	memcpy(&newtile->m.f,&tile->m.f,n*sizeof(tile->m.f[0]));
	memcpy(&newtile->fourh2.f,&tile->fourh2.f,n*sizeof(tile->fourh2.f[0]));
	memcpy(&newtile->x.f,&tile->x.f,n*sizeof(tile->x.f[0]));
	memcpy(&newtile->y.f,&tile->y.f,n*sizeof(tile->y.f[0]));
	memcpy(&newtile->z.f,&tile->z.f,n*sizeof(tile->z.f[0]));
	memcpy(&newtile->xOffset.f,&tile->xOffset.f,n*sizeof(tile->xOffset.f[0]));
	memcpy(&newtile->yOffset.f,&tile->yOffset.f,n*sizeof(tile->yOffset.f[0]));
	memcpy(&newtile->zOffset.f,&tile->zOffset.f,n*sizeof(tile->zOffset.f[0]));
	memcpy(&newtile->xCenter.f,&tile->xCenter.f,n*sizeof(tile->xCenter.f[0]));
	memcpy(&newtile->yCenter.f,&tile->yCenter.f,n*sizeof(tile->yCenter.f[0]));
	memcpy(&newtile->zCenter.f,&tile->zCenter.f,n*sizeof(tile->zCenter.f[0]));
	memcpy(&newtile->xMax.f,&tile->xMax.f,n*sizeof(tile->xMax.f[0]));
	memcpy(&newtile->yMax.f,&tile->yMax.f,n*sizeof(tile->yMax.f[0]));
	memcpy(&newtile->zMax.f,&tile->zMax.f,n*sizeof(tile->zMax.f[0]));
    }
    return cl->tile;
}
