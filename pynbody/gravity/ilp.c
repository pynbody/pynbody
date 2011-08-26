#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "ilp.h"
#include "kd.h" /* for PARTITION macro */

size_t ilpMemory(ILP ilp) {
    size_t nBytes = sizeof(struct ilpContext);
    ILPTILE tile;
    for(tile=ilp->first;tile!=NULL;tile=tile->next)
	nBytes += sizeof(struct ilpTile);
    return nBytes;
    }

/*
** Private: Create a new tile
*/
static ILPTILE newTile(ILPTILE prev) {
    ILPTILE tile = malloc(sizeof(struct ilpTile));
    int i;

    assert( tile != NULL );
    assert(ILP_PART_PER_TILE%4 == 0 );

    tile->next = NULL;
    tile->prev = prev;
    tile->nMaxPart = ILP_PART_PER_TILE;
    tile->nPart = 0;

    /*
    ** We need valid data for the SIMD PP code. This is probably the best way.
    */
    for(i=0; i<tile->nMaxPart; ++i) {
	tile->dx.f[i] = tile->dy.f[i] = tile->dz.f[i] = tile->d2.f[i] = 1.0f;
	tile->vx.f[i] = tile->vy.f[i] = tile->vz.f[i] = tile->fourh2.f[i] = 1.0f;
	tile->m.f[i] = 0.0;
	tile->iOrder.i[i] = 0;
        }
    return tile;
    }

/*
** If the current tile is full (nPart == nMaxPart), then
** this function is called to get a new, empty tile.
*/
ILPTILE ilpExtend(ILP ilp) {
    assert( ilp != NULL );
    assert( ilp->tile != NULL );
    assert( ilp->first != NULL );
    assert( ilp->tile->nPart == ilp->tile->nMaxPart );

    ilp->nPrevious += ilp->tile->nPart;

    /* Use the next tile if it exists, or create a new one */
    if ( ilp->tile->next != NULL ) {
	ilp->tile = ilp->tile->next;
	ilp->tile->nPart = 0;
	}
    else {
	ilp->tile = ilp->tile->next = newTile(ilp->tile);
	}

    return ilp->tile;
    }

/*
** Empty the list of particles (go back to the first tile)
*/
ILPTILE ilpClear(ILP ilp) {
    assert( ilp != NULL );
    ilp->tile = ilp->first;
    ilp->nPrevious = 0;
    assert( ilp->tile != NULL );
    ilp->tile->nPart = 0;
    ilp->cx = ilp->cy = ilp->cz = 0.0;
    return ilp->tile;
    }

void ilpInitialize(ILP *ilp) {
    *ilp = malloc(sizeof(struct ilpContext));
    assert( *ilp != NULL );
    (*ilp)->first = (*ilp)->tile = newTile(NULL);
    (*ilp)->nPrevious = 0;
    }

void ilpFinish(ILP ilp) {
    ILPTILE tile, next;

    assert( ilp != NULL );

    /* Free all allocated tiles first */
    for ( tile=ilp->first; tile!=NULL; tile=next ) {
	next = tile->next;
	free(tile);
	}

    free(ilp);
    }
