/*
 * Gravity code stolen from Zurich group's (Joachim Stadel, Doug Potter,
 * Jonathan Coles, et al) amazing gravity code pkdgrav2 and severly
 * bastardized and converted from parallel to serial for pynbody by
 * Greg Stinson with help from Jonathan, Tom Quinn, Rok Roskar and
 * Andrew Pontzen.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#if defined(__APPLE__)
#undef HAVE_MALLOC_H
#endif
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
#include <math.h>
#include <time.h>
#include <assert.h>
#include "kd.h"
#ifndef HAVE_CONFIG_H
#include "floattype.h"
#endif

#define WALK_MINMULTIPOLE	4

/*
** Returns total number of active particles for which gravity was calculated.
*/
void kdGravWalk(KD kd, int nReps,int bEwald, PARTICLE *testParticles, int nPos){
    PARTICLE *p;
    PARTICLE q;
    KDN *kdn;
    KDN *kdc;
    pBND kbnd, cbnd;
    FLOAT dMin,min2;
    FLOAT rCheck[3];
    FLOAT rOffset[3];
    int ix,iy,iz,bRep;
    int nMaxInitCheck,nCheck, iCheckCell;
    int i,ii,j,n,pj,nTotActive;
    int iOpen, iPos;
    int nPart;
    int nCell;
    double start_time;

    assert(kd->oNodeMom);
    /*
    ** Allocate initial interaction lists.
    */
    nMaxInitCheck = 2*nReps+1;
    nMaxInitCheck = nMaxInitCheck*nMaxInitCheck*nMaxInitCheck;	/* all replicas */
    assert(nMaxInitCheck < kd->nMaxCheck);
    ii=0;
    for (iPos = 0; iPos < nPos; iPos++) {
	start_time = kdTime();
	q = testParticles[iPos];
	kdn = kdNodeInit(kd, &q);
	free(kd->ilp);
	kd->ilp = malloc(kd->nMaxPart*sizeof(ILP));
	assert(kd->ilp != NULL);
	nPart = 0;
	free(kd->ilc);
	kd->ilc = malloc(kd->nMaxCell*sizeof(ILC));
	assert(kd->ilc != NULL);
	nCell = 0;

	nCheck = 0;
	/*
	** First we add any replicas of the entire box
	** to the Checklist.
	*/
	for (ix=-nReps;ix<=nReps;++ix) {
	    rOffset[0] = ix*kd->fPeriod[0];
	    for (iy=-nReps;iy<=nReps;++iy) {
		rOffset[1] = iy*kd->fPeriod[1];
		for (iz=-nReps;iz<=nReps;++iz) {
		    rOffset[2] = iz*kd->fPeriod[2];
		    /*
		      bRep = ix || iy || iz;
		      if (bRep) {
		    */
		    kd->Check[nCheck].iCell = ROOT;
		    for (j=0;j<3;++j) kd->Check[nCheck].rOffset[j] = rOffset[j];
		    ++nCheck;
		    /*
		      }
		    */
		    }
		}
	    }
	while (1) {
	    /*
	    ** Process the Checklist.
	    ii = 0;
	    */
	    for (i=0;i<nCheck;++i) {
		kdc = kdTreeNode(kd,kd->Check[i].iCell);
		n = kdc->pUpper - kdc->pLower + 1;
		/*
		printf("node %d, upper: %d, lower: %d, n:
		%d\n",kd->Check[i].iCell,kdc->pUpper,
		kdc->pLower,kdc->pUpper - kdc->pLower + 1);
		*/
		kdNodeBnd(kd, kdc, &cbnd);
		/*printf("%d rCheck",kd->Check[i].iCell);*/
		for (j=0;j<3;++j) {
		    rCheck[j] = cbnd.fCenter[j] + kd->Check[i].rOffset[j];
		    /*rCheck[j] = kdc->r[j] + kd->Check[i].rOffset[j];*/
		    /*printf("[%d]: %g ",j,rCheck[j]);*/
		    }
		/*printf("\n");*/
		
		kdNodeBnd(kd, kdn, &kbnd);
		/*
		** If this cell is a bucket we have to either open the checkcell
		** and get the particles, or accept the multipole. For this
		** reason we only need to calculate min2.
		*/
		min2 = 0;
		for (j=0;j<3;++j) {
		    dMin = fabs(rCheck[j] - kbnd.fCenter[j]);
		    dMin -= kbnd.fMax[j];
		    if (dMin > 0) min2 += dMin*dMin;
		    }
		/*
		** By default we open the cell!
		*/
		printf("min2: %g fOpen: %g\n",6.8e4*sqrt(min2),6.8e4*kdc->bMax);
		if (min2 > kdc->bMax*kdc->bMax/kd->dTheta2) {

#if !defined(SOFTLINEAR) && !defined(SOFTSQUARE)
		    if (min2 > 4*kdc->fSoft2) {
			printf("Open cell?  n: %d",n);
			if (n >= WALK_MINMULTIPOLE) iOpen = -1;
			else iOpen = 1;
			}
		    else iOpen = -2; /* means we treat this cell as a softened monopole */
#endif
		    }
		else {
		    iOpen = 1;
		    }
		/*
		  printf("   i:%6d iCheck:%6d iOpen:%2d\n",i,kd->Check[i].iCell,iOpen);
		*/
		if (iOpen > 0) {
		    /*
		    ** Contained! (or intersected in the case of reaching the bucket)
		    */
		    iCheckCell = kdc->iLower;
		    if (iCheckCell) {
			/*
			** Open the cell.
			** Here we ASSUME that the children of
			** kdc are all in sequential memory order!
			** (the new tree build assures this)
			** We could do a prefetch here for non-local
			** cells.
			*/
			if (nCheck + 2 > kd->nMaxCheck) {
			    kd->nMaxCheck += 1000;
			    kd->Check = realloc(kd->Check,kd->nMaxCheck*sizeof(CELT));
			    assert(kd->Check != NULL);
			    }
			kd->Check[nCheck] = kd->Check[i];
			kd->Check[nCheck+1] = kd->Check[i];
			kd->Check[nCheck].iCell = iCheckCell;
			kd->Check[nCheck+1].iCell = iCheckCell+1;
			nCheck += 2;
			}
		    else {
			/*
			** Now I am trying to open a bucket, which means I place particles on the kd->ilp
			** interaction list. I also have to make sure that I place the
			** particles in time synchronous positions.
			*/
			/*
			** Local Bucket Interaction.
			** Interact += Pacticles(kdc);
			*/
			if (nPart + n > kd->nMaxPart) {
			    kd->nMaxPart += 500 + n;
			    kd->ilp = realloc(kd->ilp,kd->nMaxPart*sizeof(ILP));
			    assert(kd->ilp != NULL);
			    }
			for (pj=kdc->pLower;pj<=kdc->pUpper;++pj) {
			    p = kdParticle(kd,pj);
			    /*
			    ilpAppend(kd->ilp,
				      p->r[0] + kd->Check[i].rOffset[0],
				      p->r[1] + kd->Check[i].rOffset[1],
				      p->r[2] + kd->Check[i].rOffset[2],
				      p->fMass,4*p->fSoft*p->fSoft,p->iOrder);
			    */
			    kd->ilp[nPart].iOrder = p->iOrder;
			    kd->ilp[nPart].m = p->fMass;
			    kd->ilp[nPart].x = p->r[0] + kd->Check[i].rOffset[0];
			    kd->ilp[nPart].y = p->r[1] + kd->Check[i].rOffset[1];
			    kd->ilp[nPart].z = p->r[2] + kd->Check[i].rOffset[2];
			    kd->ilp[nPart].fourh2 = 4*p->fSoft*p->fSoft;
			    ++nPart;
			    }
			}  /* end of opening a bucket */
		    }
		else if (iOpen == -1) {
		    /*
		    ** No intersection, accept multipole!
		    ** Interact += Moment(kdc);
		    */
		    if (nCell == kd->nMaxCell) {
			kd->nMaxCell += 500;
			kd->ilc = realloc(kd->ilc,kd->nMaxCell*sizeof(ILC));
			assert(kd->ilc != NULL);
			}
		    kd->ilc[nCell].x = rCheck[0];
		    kd->ilc[nCell].y = rCheck[1];
		    kd->ilc[nCell].z = rCheck[2];
		    kd->ilc[nCell].mom = *kdNodeMom(kd,kdc);
		    /*
		    ilcAppend(kd->ilc,rCheck[0],rCheck[1],rCheck[2],
			      kdNodeMom(kd,kdc),kdc->bMax);
		    */
		    ++nCell;
		    }
		else if (iOpen == -2) {
		    /*
		    ** We accept this multipole from the opening criterion, but it is a softened
		    ** interaction, so we need to treat is as a softened monopole by putting it
		    ** on the particle interaction list.
		    */
		    if (nPart == kd->nMaxPart) {
			kd->nMaxPart += 500;
			kd->ilp = realloc(kd->ilp,kd->nMaxPart*sizeof(ILP));
			assert(kd->ilp != NULL);
			}
		    /*
		    ilpAppend(kd->ilp,rCheck[0],rCheck[1],rCheck[2],
			      kdNodeMom(kd,kdc)->m,4*kdc->fSoft2,-1); 
		    */
		    kd->ilp[nPart].iOrder = -1; /* set iOrder to negative value for time step criterion */
		    kd->ilp[nPart].m = kdNodeMom(kd,kdc)->m;
		    kd->ilp[nPart].x = rCheck[0];
		    kd->ilp[nPart].y = rCheck[1];
		    kd->ilp[nPart].z = rCheck[2];
		    kd->ilp[nPart].fourh2 = 4*kdc->fSoft2;
		    /* set iOrder to negative value for time step criterion */
		    ++nPart;
		    }
		else {
		    kd->Check[ii++] = kd->Check[i];
		    }
		}
	    printf("checklist length: %d\n",nCheck);
	    nCheck = ii;
	    /*
	    ** Done processing of the Checklist.
	    */
	    if (!kdn->iLower) break;
	    }
	/*
	** Now the interaction list should be complete and the
	** Checklist should be empty! Calculate gravity on this
	** Bucket!
	*/
	assert(nCheck == 0);
	/*
	** We no longer add *this bucket to any interaction list, this is now done with an
	** N(N-1)/2 loop in kdBucketInteract().
	*/
	kdc = kdn;
	/*
	** Now calculate gravity on this bucket!
	*/
	printf("iPos: %d, tree walked: %g s\n",iPos,kdTime()-start_time);
	start_time=kdTime();
	kdGravInteract(kd,kdc,kd->ilp,nPart,kd->ilc,nCell, bEwald,&q);
	testParticles[iPos].fPot = q.fPot;
	for (j=0; j<3; j++) testParticles[iPos].a[j] = q.a[j];
	printf("gravity calculated: %g s\n",kdTime()-start_time);
	free(kdn);
	}
    }
