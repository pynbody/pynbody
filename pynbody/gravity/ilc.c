#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include "ilc.h"

size_t ilcMemory(ILC ilc) {
    size_t nBytes = sizeof(struct ilcContext);
    ILCTILE tile;
    for(tile=ilc->first;tile!=NULL;tile=tile->next)
	nBytes += sizeof(struct ilcTile);
    return nBytes;
    }

/*
** Private: Create a new tile
*/
static ILCTILE newTile(ILCTILE prev) {
    ILCTILE tile = malloc(sizeof(struct ilcTile));
    int i;

    assert( tile != NULL );
    assert(ILC_PART_PER_TILE%4 == 0 );

    tile->next = NULL;
    tile->prev = prev;
    tile->nMaxCell = ILC_PART_PER_TILE;
    tile->nCell = 0;

    for(i=0; i<tile->nMaxCell; ++i) {
	tile->dx.f[i] = tile->dy.f[i] = tile->dz.f[i] = tile->d2.f[i] = 1.0;
#ifdef HERMITE
	tile->vx.f[i] = tile->vy.f[i] = tile->vz.f[i] = 1.0;
#endif
	tile->xxxx.f[i] = tile->xxxy.f[i] = tile->xxxz.f[i] = tile->xxyz.f[i] = 
	    tile->xxyy.f[i] = tile->yyyz.f[i] = tile->xyyz.f[i] = tile->xyyy.f[i] = tile->yyyy.f[i] = 0.0f;
	tile->xxx.f[i] = tile->xyy.f[i] = tile->xxy.f[i] = tile->yyy.f[i] = 
	    tile->xxz.f[i] = tile->yyz.f[i] = tile->xyz.f[i] = 0.0f;
	tile->xx.f[i] = tile->xy.f[i] = tile->xz.f[i] = tile->yy.f[i] = tile->yz.f[i] = 0.0f;
	tile->m.f[i] = tile->u.f[i] = 0.0f;
	}
    return tile;
    }

/*
** If the current tile is full (nCell == nMaxCell), then
** this function is called to get a new, empty tile.
*/
ILCTILE ilcExtend(ILC ilc) {
    assert( ilc != NULL );
    assert( ilc->tile != NULL );
    assert( ilc->first != NULL );
    assert( ilc->tile->nCell == ilc->tile->nMaxCell );

    ilc->nPrevious += ilc->tile->nCell;

    /* Use the next tile if it exists, or create a new one */
    if ( ilc->tile->next != NULL ) {
	ilc->tile = ilc->tile->next;
	ilc->tile->nCell = 0;
	}
    else {
	ilc->tile = ilc->tile->next = newTile(ilc->tile);
	}

    return ilc->tile;
    }

/*
** Empty the list of particles (go back to the first tile)
*/
ILCTILE ilcClear(ILC ilc) {
    assert( ilc != NULL );
    ilc->tile = ilc->first;
    ilc->nPrevious = 0;
    assert( ilc->tile != NULL );
    ilc->tile->nCell = 0;
    ilc->cx = ilc->cy = ilc->cz = 0.0;
    return ilc->tile;
    }

void ilcInitialize(ILC *ilc) {
    *ilc = malloc(sizeof(struct ilcContext));
    assert( *ilc != NULL );
    (*ilc)->first = (*ilc)->tile = newTile(NULL);
    (*ilc)->nPrevious = 0;
    }

void ilcFinish(ILC ilc) {
    ILCTILE tile, next;

    assert( ilc != NULL );

    /* Free all allocated tiles first */
    for ( tile=ilc->first; tile!=NULL; tile=next ) {
	next = tile->next;
	free(tile);
	}

    free(ilc);
    }
