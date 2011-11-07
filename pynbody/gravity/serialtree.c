/*
 * Gravity code stolen from Zurich group's (Joachim Stadel, Doug Potter,
 * Jonathan Coles, et al) amazing gravity code pkdgrav2 and severly
 * bastardized and converted from parallel to serial for pynbody by
 * Greg Stinson with help from Jonathan, Tom Quinn, Rok Roskar and
 * Andrew Pontzen.
 */

#define HAVE_CONFIG_H
#ifdef HAVE_CONFIG_H
#include "config.h"
#else
#include "floattype.h"
#endif

#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#if defined(__APPLE__)
#undef HAVE_MALLOC_H
#endif
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
//#if defined(__APPLE__)
#include <stdio.h>
//#endif
#include <assert.h>
#include "kd.h"
#include "moments.h"

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

double kdTime(void) {
	struct timeval tv;

	gettimeofday(&tv,NULL);
	return tv.tv_sec + tv.tv_usec*1e-6;
	}
/* Add a NODE structure: assume double alignment */
static int kdNodeAddStruct(KD kd,int n) {
    int iOffset = kd->iTreeNodeSize;
    assert( kd->kdNodeListPRIVATE == NULL );
    assert( (iOffset & (sizeof(double)-1)) == 0 );
    kd->iTreeNodeSize += n;
    return iOffset;
    }
/* Add n doubles to the node structure */
static int kdNodeAddDouble(KD kd,int n) {
    int iOffset = kd->iTreeNodeSize;
    assert( kd->kdNodeListPRIVATE == NULL );
    assert( (iOffset & (sizeof(double)-1)) == 0 );
    kd->iTreeNodeSize += sizeof(double) * n;
    return iOffset;
    }

void kdInitialize( KD kd,int nStore,int nBucket,double dTheta,int nTreeBitsLo,
		   int nTreeBitsHi, float *fPeriod) {
    int j,ism;
    uint64_t mMemoryModel = 0;

#define RANDOM_SEED 1
    srand(RANDOM_SEED);

    /*kd = (KD )malloc(sizeof(struct kdContext));*/
    assert(kd != NULL);
    kd->kdNodeListPRIVATE = NULL;
    kd->pStorePRIVATE = NULL;
    kd->nStore = nStore;
    kd->nRejects = 0;
    for (j=0;j<3;++j) {
	kd->fPeriod[j] = fPeriod[j];
	}

    kd->dTheta2 = dTheta*dTheta;
    
    /*
    ** Calculate the amount of memory (size) of each particle.  This is the
    ** size of a base particle (PARTICLE), plus any extra fields as defined
    ** by the current memory model.  Fields need to be added in order of
    ** descending size (i.e., doubles & int64 and then float & int32)
    */
    kd->iParticleSize = sizeof(PARTICLE);
    kd->iTreeNodeSize = sizeof(KDN);

    /*
    ** Tree node memory models
    */
    mMemoryModel |= KD_MODEL_NODE_MOMENT;
    mMemoryModel |= KD_MODEL_NODE_BND;
    if ( mMemoryModel & KD_MODEL_NODE_BND ) {
        if ( mMemoryModel & KD_MODEL_NODE_BND6 ) {
            kd->oNodeBnd6 =
	    kd->oNodeBnd  = kdNodeAddDouble(kd,6+6+1);
        }
        else {
	    kd->oNodeBnd6 = 0;
	    kd->oNodeBnd  = kdNodeAddDouble(kd,3+3+1);
        }
    }
    else {
	kd->oNodeBnd  = 0;
	kd->oNodeBnd6 = 0;
    }

    if ( mMemoryModel & KD_MODEL_NODE_MOMENT )
	kd->oNodeMom = kdNodeAddStruct(kd,sizeof(FMOMR));
    else
	kd->oNodeMom = 0;

    /*
    ** N.B.: Update kdMaxNodeSize in kd.h if you add fields.  We need to
    **       know the size of a node when setting up the pst.
    */
    assert(kdNodeSize(kd) > 0);
    assert(kdNodeSize(kd)<=kdMaxNodeSize());

    /*
    ** Allocate the main particle store.
    **
    ** We need one EXTRA storage location at the very end to use for
    ** calculating acceleration on arbitrary positions in space, for example
    ** determining the force on the sun. The easiest way to do this is to
    ** allocate one hidden particle, which won't interfere with the rest of
    ** the code (hopefully). kd->pStore[kd->nStore] is this particle.
    **
    ** We also allocate a temporary particle used for swapping.  We need to do
    ** this now because the outside world can no longer know the size of a
    ** particle.
    */
    kd->iParticleSize = (kd->iParticleSize + sizeof(double) - 1 ) & ~(sizeof(double)-1);
    kd->pTempPRIVATE = malloc(kdParticleSize(kd));
    assert(kd->pTempPRIVATE != NULL);

    kd->fSoftFix = -1.0;
    kd->fSoftFac = 1.0;
    kd->fSoftMax = HUGE;
    /*
    ** Now we setup the node storage for the tree.  We allocate
    ** a single "tile" for the tree.  If this is not sufficient, then additional
    ** tiles are allocated dynamically.  The default parameters allow for 2^32
    ** nodes total which is the integer limit anyway.
    */
    kd->iTreeNodeSize = (kd->iTreeNodeSize + sizeof(double) - 1 ) & ~(sizeof(double)-1);
    kd->nTreeBitsLo = nTreeBitsLo;
    kd->nTreeBitsHi = nTreeBitsHi;
    kd->iTreeMask = (1<<kd->nTreeBitsLo) - 1;
    kd->kdNodeListPRIVATE = malloc((1<<kd->nTreeBitsHi)*sizeof(KDN *));
    assert(kd->kdNodeListPRIVATE != NULL);
    kd->kdNodeListPRIVATE[0] = malloc((1<<kd->nTreeBitsLo)*kd->iTreeNodeSize);
    assert(kd->kdNodeListPRIVATE[0] != NULL);
    kd->nTreeTiles = 1;
    kd->nMaxNodes = (1<<kd->nTreeBitsLo) * kd->nTreeTiles;
    /*
    ** pLite particles are also allocated and are quicker when sorting particle
    ** type operations such as tree building and domain decomposition are being
    ** performed.
    */
    kd->pLite = malloc((nStore+1)*sizeof(PLITE));
    assert(pkd->pLite != NULL);
    kd->nNodes = 0;
    kd->kdTopPRIVATE = NULL;
    /*
    ** Ewald stuff!
    */
    kd->ew.nMaxEwhLoop = 100;
    kd->ew.ewt = malloc(kd->ew.nMaxEwhLoop*sizeof(EWT));
    assert(kd->ew.ewt != NULL);
    /*
    ** Tree walk stuff.
    */
    kd->nMaxPart = 10000;
    kd->ilp = malloc(kd->nMaxPart*sizeof(ILP));
    assert(kd->ilp != NULL);
    kd->nMaxCell = 1000;
    kd->ilc = malloc(kd->nMaxCell*sizeof(ILC));
    assert(kd->ilc != NULL);
    /*
    ** Allocate Checklist.
    */
    kd->nMaxCheck = 10000;
    kd->Check = malloc(kd->nMaxCheck*sizeof(CELT));
    assert(kd->Check != NULL);
    /*
    ** Allocate the stack.
    */
    kd->nMaxStack = 30;
    kd->S = malloc(kd->nMaxStack*sizeof(CSTACK));
    assert(kd->S != NULL);
    for (ism=0;ism<kd->nMaxStack;++ism) {
	kd->S[ism].Check = malloc(kd->nMaxCheck*sizeof(CELT));
	assert(kd->S[ism].Check != NULL);
	}

#ifdef USE_DEHNEN_THETA
    kd->fCritTheta = NULL;
    kd->fCritMass = NULL;
    kd->nCritBins = 0;
    kd->dCritThetaMin = 0.0;
#endif

    assert(kdNodeSize(kd) > 0);
    }

void InitializeParticles(KD kd, BND *pbnd) {
    PLITE *pLite = kd->pLite;
    PARTICLE *p;
    KDN *pNode;
    pBND bnd;
    int i,j;

    /*
    ** Initialize the temporary particles.
    */
    for (i=0;i<kd->nStore;++i) {
	p = kdParticle(kd,i);
	for (j=0;j<3;++j) pLite[i].r[j] = p->r[j];
	pLite[i].i = i;
	}
    /*
    **It is only forseen that there are 4 reserved nodes at present 0-NULL, 1-ROOT, 2-UNUSED, 3-VAROOT.
    */
    kd->nNodes = NRESERVED_NODES;

    /*
    ** Set up the root node.
    */
    pNode = kdTreeNode(kd,ROOT);
    pNode->iLower = 0;
    pNode->iParent = 0;
    pNode->pLower = 0;
    pNode->pUpper = kd->nStore - 1;
    kdNodeBnd(kd, pNode, &bnd);
    for (j=0;j<3;++j) {
	bnd.fCenter[j] = pbnd->fCenter[j];
	bnd.fMax[j] = pbnd->fMax[j];
	}
    }

KDN * kdNodeInit(KD kd, PARTICLE *p) {
    KDN *pNode;
    pBND bnd;
    int j;

    pNode = (KDN *)malloc(kd->iTreeNodeSize);
    pNode->iLower = 0;
    pNode->iParent = 0;
    pNode->pLower = 0;
    pNode->pUpper = 1;
    kdNodeBnd(kd, pNode, &bnd);
    for (j=0;j<3;++j) {
	pNode->r[j] = p->r[j];
	bnd.fCenter[j] = p->r[j];
	bnd.fMax[j] = 0;
	}

    CALCOPEN(pNode, 0)
    return pNode;
    }

/* Extend the tree by adding more nodes */
void kdExtendTree(KD kd) {
    if ( kd->nTreeTiles >= (1<<kd->nTreeBitsHi) ) {
	fprintf(stderr, "ERROR: insufficent nodes available in tree build"
	    "-- Increase nTreeBitsLo and/or nTreeBitsHi\n"
	    "nTreeBitsLo=%d nTreeBitsHi=%d\n",
	    kd->nTreeBitsLo, kd->nTreeBitsHi);
	assert( kd->nTreeTiles < (1<<kd->nTreeBitsHi) );
	}
    kd->kdNodeListPRIVATE[kd->nTreeTiles] = malloc((1<<kd->nTreeBitsLo)*kd->iTreeNodeSize);
    assert(kd->kdNodeListPRIVATE[kd->nTreeTiles] != NULL);
    ++kd->nTreeTiles;
    kd->nMaxNodes = (1<<kd->nTreeBitsLo) * kd->nTreeTiles;
    }

#define MIN_SRATIO    0.05

/*
** M is the bucket size.
** This function assumes that the root node is correctly set up (particularly the bounds).
*/
#define TEMP_S_INCREASE 100
void BuildTemp(KD kd,int iNode,int M) {
    PLITE *p = kd->pLite;
    KDN *pNode = kdTreeNode(kd,iNode);
    pBND bnd,lbnd,rbnd;
    KDN *pLeft, *pRight;
    PLITE t;
    FLOAT fSplit;
    FLOAT ls;
    int *S;		/* this is the stack */
    int s,ns;
    int iLeft,iRight;
    int d,i,j;
    int nr,nl;
    int lc,rc;
    int nBucket = 0;

    kdNodeBnd(kd,pNode,&bnd);

    /*
    ** Allocate stack!
    */
    ns = TEMP_S_INCREASE;
    s = 0;
    S = malloc(ns*sizeof(int));
    assert(S != NULL);

    if (pNode->pUpper - pNode->pLower + 1 <= M){
	printf("upper:%d lower:%d u-l: %d M:%d\n",pNode->pUpper,pNode->pLower,pNode->pUpper-pNode->pLower, M);
	goto DonePart;
	}
    assert( bnd.fMax[0] > 0.0 ||
	    bnd.fMax[1] > 0.0 ||
	    bnd.fMax[2] > 0.0 );
    while (1) {
	/*
	** Begin new stage!
	** Calculate the appropriate fSplit.
	** Pick longest dimension and split it in half.
	*/
	if (bnd.fMax[0] < bnd.fMax[1]) {
	    if (bnd.fMax[1] < bnd.fMax[2]) d = 2;
	    else d = 1;
	    }
	else if (bnd.fMax[0] < bnd.fMax[2]) d = 2;
	else d = 0;
	fSplit = bnd.fCenter[d];
	/*
	** Now start the partitioning of the particles about
	** fSplit on dimension given by d.
	*/
	i = pNode->pLower;
	j = pNode->pUpper;
	while (i <= j) {
	    if (p[i].r[d] < fSplit) ++i;
	    else break;
	    }
	while (i <= j) {
	    if (fSplit < p[j].r[d]) --j;
	    else break;
	    }
	if (i < j) {
	    t = p[i];
	    p[i] = p[j];
	    p[j] = t;
	    while (1) {
		while (p[++i].r[d] < fSplit);
		while (fSplit < p[--j].r[d]);
		if (i < j) {
		    t = p[i];
		    p[i] = p[j];
		    p[j] = t;
		    }
		else break;
		}
	    }

	nl = i - pNode->pLower;
	nr = pNode->pUpper - i + 1;
	if (nl > 0 && nr > 0) {
	    /*
	    ** Allocate 2 new tree nodes making sure that we have
	    ** allocated enough storage.
	    */
	    if ( kd->nNodes+2 > kd->nMaxNodes ) {
		kdExtendTree(kd);
		}
	    iLeft = kd->nNodes++;
	    pLeft = kdTreeNode(kd,iLeft);
	    pLeft->iParent = iNode;
	    pLeft->pLower = pNode->pLower;
	    pLeft->pUpper = i-1;
	    iRight = kd->nNodes++;
	    pRight = kdTreeNode(kd,iRight);
	    assert(iRight & 1);
	    pRight->iParent = iNode;
	    pRight->pLower = i;
	    pRight->pUpper = pNode->pUpper;
	    pNode->iLower = iLeft;

            kdNodeBnd(kd, pLeft, &lbnd);
            kdNodeBnd(kd, pRight, &rbnd);

	    /*
	    ** Now deal with the bounds.
	    */
	    for (j=0;j<3;++j) {
		if (j == d) {
		    rbnd.fMax[j] = lbnd.fMax[j] = 0.5*bnd.fMax[j];
		    lbnd.fCenter[j] = bnd.fCenter[j] - lbnd.fMax[j];
		    rbnd.fCenter[j] = bnd.fCenter[j] + rbnd.fMax[j];
		    }
		else {
		    lbnd.fCenter[j] = bnd.fCenter[j];
		    lbnd.fMax[j] = bnd.fMax[j];
		    rbnd.fCenter[j] = bnd.fCenter[j];
		    rbnd.fMax[j] = bnd.fMax[j];
		    }
		}
	    /*
	    ** Now figure out which subfile to process next.
	    */
	    lc = (nl > M); /* this condition means the left child is not a bucket */
	    rc = (nr > M);
	    if (rc && lc) {
		/* Allocate more stack if required */
		if ( s+1 >= ns ) {
		    assert( s+1 == ns );
		    ns += TEMP_S_INCREASE;
		    S = realloc(S,ns*sizeof(int));
		    }
		if (nr > nl) {
		    S[s++] = iRight;	/* push tr */
		    iNode = iLeft;		/* process lower subfile */
		    }
		else {
		    S[s++] = iLeft;	/* push tl */
		    iNode = iRight;		/* process upper subfile */
		    }
		}
	    else if (lc) {
		/*
		** Right must be a bucket in this case!
		*/
		iNode = iLeft;   /* process lower subfile */
		pRight->iLower = 0;
		++nBucket;
		}
	    else if (rc) {
		/*
		** Left must be a bucket in this case!
		*/
		iNode = iRight;   /* process upper subfile */
		pLeft->iLower = 0;
		++nBucket;
		}
	    else {
		/*
		** Both are buckets (we need to pop from the stack to get the next subfile.
		*/
		pLeft->iLower = 0;
		++nBucket;
		pRight->iLower = 0;
		++nBucket;
		if (s) iNode = S[--s];		/* pop tn */
		else break;
		}
	    }
	else {
	    /*
	    ** No nodes allocated, Change the bounds if needed!
	    */
	    if (d >= 0 && d < 3) bnd.fMax[d] *= 0.5;
	    if (nl > 0) {
		if (d >= 0 && d < 3) bnd.fCenter[d] -= bnd.fMax[d];
		MAXSIDE(bnd.fMax,ls);
		lc = (nl > M); /* this condition means the node is not a bucket */
		if (!lc) {
		    pNode->iLower = 0;
		    ++nBucket;
		    if (s) iNode = S[--s];		/* pop tn */
		    else break;
		    }
		}
	    else {
		if (d >= 0 && d < 3) bnd.fCenter[d] += bnd.fMax[d];
		rc = (nr > M);
		if (!rc) {
		    pNode->iLower = 0;
		    ++nBucket;
		    if (s) iNode = S[--s];		/* pop tn */
		    else break;
		    }
		}
	    }
	pNode = kdTreeNode(kd,iNode);
        kdNodeBnd(kd, pNode, &bnd);
	}
DonePart:
    free(S);
    }

/*
** If this is called with iStart being the index of the first very active particle
** then it reshuffles only the very actives. This is again a bit ugly, but will
** do for now.
*/
void ShuffleParticles(KD kd,int iStart) {
    PARTICLE *p, *pNew, *pNewer;
    int i,iNew,iNewer,iTemp;

    /*
    ** Now we move the particles in one go using the temporary
    ** particles which have been shuffled.
    */
    iTemp = iStart;
    while (1) {
	p = &(kd->pStorePRIVATE[iTemp]);
	kdSaveParticle(kd,p);
	i = iTemp;
	iNew = kd->pStorePRIVATE[i].iOrder;
	while (iNew != iTemp) {
	    pNew = kdParticle(kd,iNew);
	    iNewer = kd->pStorePRIVATE[iNew].iOrder;
	    pNewer = kdParticle(kd,iNewer);
	    /* Particles are being shuffled here in a non-linear order.
	    ** Being smart humans, we can tell the CPU where the next chunk
	    ** of data can be found.  The limit is 8 outstanding prefetches
	    ** (according to the Opteron Guide).
	    */
	    /*
#if defined(__GNUC__) || defined(__INTEL_COMPILER)
	    __builtin_prefetch((char *)(pNewer)+0,1,0);
#ifndef __ALTIVEC__
	    __builtin_prefetch((char *)(pNewer)+64,1,0);
#endif
#endif
	    */
	    kdCopyParticle(kd,p,pNew);
	    kd->pStorePRIVATE[i].iOrder = 0;
	    i = iNew;
	    p = kdParticle(kd,i);
	    iNew = kd->pStorePRIVATE[i].iOrder;
	    }
	kdLoadParticle(kd,p);
	kd->pStorePRIVATE[i].iOrder = 0;
	while (!kd->pStorePRIVATE[iTemp].iOrder) {
	    if (++iTemp == kd->nStore) return;
	    }
	}
    }

void Create(KD kd,int iNode) {
    PARTICLE *p;
    KDN *kdn,*kdl,*kdu;
    FMOMR mom;
    pBND bnd;
    FLOAT m,fMass,fSoft,x,y,z,ft,d2,d2Max,dih2,bmin,b;
    int j, pj,d,nDepth;
    const int nMaxStackIncrease = 1;
    int bSoftZero = 0;

    nDepth = 1;
    while (1) {
	while (kdTreeNode(kd,iNode)->iLower) {
	    /*
	    printf("%2d:%d\n",nDepth,iNode);
	    */
	    iNode = kdTreeNode(kd,iNode)->iLower;
	    ++nDepth;
	    /*
	    ** Is this the deepest in the tree so far? We might need to have more stack
	    ** elements for the tree walk!
	    ** nMaxStack == nDepth guarantees that there is at least one deeper
	    ** stack entry available than what is needed to walk the tree.
	    */
	    if (nDepth > kd->nMaxStack) {
		kd->S = realloc(kd->S,(kd->nMaxStack+nMaxStackIncrease)*sizeof(CSTACK));
		assert(kd->S != NULL);
		kd->nMaxStack += nMaxStackIncrease;
		}
	    }
	/*
	printf("%2d:%d\n",nDepth,iNode);
	*/
	/*
	** Now calculate all bucket quantities!
	** This includes M,CoM,Moments and special
	** bounds and iMaxRung.
	*/
	kdn = kdTreeNode(kd,iNode);
        kdNodeBnd(kd, kdn, &bnd);
	/*
	** Before squeezing the bounds, calculate a minimum b value based on the splitting bounds alone.
	** This gives us a better feel for the "size" of a bucket with only a single particle.
	*/
	MINSIDE(bnd.fMax,bmin);
	*bnd.size = 2.0*(bnd.fMax[0]+bnd.fMax[1]+bnd.fMax[2])/3.0;
	/*
	** Now shrink wrap the bucket bounds.
	*/
	pj = kdn->pLower;
	p = kdParticle(kd,pj);
	for (d=0;d<3;++d) {
	    ft = p->r[d];
	    bnd.fCenter[d] = ft;
	    bnd.fMax[d] = ft;
	    }
	for (++pj;pj<=kdn->pUpper;++pj) {
	    p = kdParticle(kd,pj);
	    for (d=0;d<3;++d) {
		ft = p->r[d];
		if (ft < bnd.fCenter[d])
		    bnd.fCenter[d] = ft;
		else if (ft > bnd.fMax[d])
		    bnd.fMax[d] = ft;
		}
	    }
	for (d=0;d<3;++d) {
	    ft = bnd.fCenter[d];
	    bnd.fCenter[d] = 0.5*(bnd.fMax[d] + ft);
	    bnd.fMax[d] = 0.5*(bnd.fMax[d] - ft);
	    }
	pj = kdn->pLower;
	p = kdParticle(kd,pj);
	m = p->fMass;
	fSoft = p->fSoft;
	fMass = m;
	if(fSoft == 0.0) {
	    dih2 = 0.0;
	    bSoftZero = 1;
	    }
	else dih2 = m/(fSoft*fSoft);

	if(bSoftZero)
	    kdn->fSoft2 = 0.0;
	else {
#if defined(TEST_SOFTENING)
	    kdn->fSoft2 = dih2*dih2;
#else
	    kdn->fSoft2 = 1/(dih2*m);
#endif
	    }

	//	d2Max = bmin*bmin;
	d2Max = 0.0;
	for (pj=kdn->pLower;pj<=kdn->pUpper;++pj) {
	    p = kdParticle(kd,pj);
	    x = p->r[0] - kdn->r[0];
	    y = p->r[1] - kdn->r[1];
	    z = p->r[2] - kdn->r[2];
	    d2 = x*x + y*y + z*z;
	    /*
	    ** Update bounding ball and softened bounding ball.
	    */
	    d2Max = (d2 > d2Max)?d2:d2Max;
	    }
        MAXSIDE(bnd.fMax,b);
        if (b < bmin) b = bmin;
	kdn->bMax = b;
	/*
	** Now calculate the reduced multipole moment.
	** Note that we use the cell's openening radius as the scaling factor!
	*/
	if (kd->oNodeMom) {
	    momClearFmomr(kdNodeMom(kd,kdn));
	    for (pj=kdn->pLower;pj<=kdn->pUpper;++pj) {
		p = kdParticle(kd,pj);
		x = p->r[0] - kdn->r[0];
		y = p->r[1] - kdn->r[1];
		z = p->r[2] - kdn->r[2];
		momMakeFmomr(&mom,p->fMass,kdn->bMax,x,y,z);
		momAddFmomr(kdNodeMom(kd,kdn),&mom);
	    }
	}
	/*
	** Finished with the bucket, move onto the next one,
	** or to the parent.
	*/
	while (iNode & 1) {
	    iNode = kdTreeNode(kd,iNode)->iParent;
	    --nDepth;
	    if (!iNode) {
		assert(nDepth == 0);
		return;	/* exit point!!! */
		}
	    /*
	    ** Now combine quantities from each of the children (2) of
	    ** this cell to form the quantities for this cell.
	    ** First find the CoM, just like for the bucket.
	    */
	    kdn = kdTreeNode(kd,iNode);
            kdNodeBnd(kd, kdn, &bnd);
	    /*
	    ** Before squeezing the bounds, calculate a minimum b value based on the splitting bounds alone.
	    ** This gives us a better feel for the "size" of a bucket with only a single particle.
	    */
	    MINSIDE(bnd.fMax,bmin);
	    *bnd.size = 2.0*(bnd.fMax[0]+bnd.fMax[1]+bnd.fMax[2])/3.0;
	    pj = kdn->pLower;
	    kdl = kdTreeNode(kd,kdn->iLower);
	    kdu = kdTreeNode(kd,kdn->iLower + 1);
	    kdCombineCells1(kd,kdn,kdl,kdu);
	    if (kdn->pUpper - pj < NMAX_OPENCALC) {
		p = kdParticle(kd,pj);
		x = p->r[0] - kdn->r[0];
		y = p->r[1] - kdn->r[1];
		z = p->r[2] - kdn->r[2];
		d2Max = x*x + y*y + z*z;
		for (++pj;pj<=kdn->pUpper;++pj) {
		    p = kdParticle(kd,pj);
		    x = p->r[0] - kdn->r[0];
		    y = p->r[1] - kdn->r[1];
		    z = p->r[2] - kdn->r[2];
		    d2 = x*x + y*y + z*z;
		    d2Max = (d2 > d2Max)?d2:d2Max;
		    }
		/*
		** Now determine the opening radius for gravity.
		*/
		for(j=0; j< 3; j++) printf(" %g ", bnd.fCenter[j]*68000);
		MAXSIDE(bnd.fMax,b);
		printf("maxside: %g",b*68000);
		if (b < bmin) b = bmin;
		if (d2Max>b) b = d2Max;
		kdn->bMax = b;
		printf(", bmin: %g, d2Max: %g, bMax 1: %g ",bmin*68000, d2Max*68000, kdn->bMax*68000);
		printf("\n");
		}
	    else {
		CALCOPEN(kdn,bmin);  /* set bMax */
		printf("bMax 2: %g ",kdn->bMax*68000);
		for(j=0; j< 3; j++) printf(" %g ", bnd.fCenter[j]*68000);
		printf("\n");
	    }
	    kdCombineCells2(kd,kdn,kdl,kdu);
	    }
	++iNode;
	}
    }


void kdCombineCells1(KD kd,KDN *kdn,KDN *p1,KDN *p2) {
    FLOAT m1,m2,ifMass;
    int j;
    pBND bnd, p1bnd, p2bnd;

    kdNodeBnd(kd, kdn, &bnd);
    kdNodeBnd(kd, p1, &p1bnd);
    kdNodeBnd(kd, p2, &p2bnd);

    if (kd->oNodeMom) {
	m1 = kdNodeMom(kd,p1)->m;
	m2 = kdNodeMom(kd,p2)->m;
	ifMass = 1/(m1 + m2);
	/*
	** In the case where a cell has all its particles source inactive mom.m == 0, which is ok, but we
	** still need a reasonable center in order to define opening balls in the tree code.
	*/
	if ( m1==0.0 || m2 == 0.0 ) {
	    ifMass = 1.0;
	    m1 = m2 = 0.5;
	    }
	}
    else {
	ifMass = 1.0;
	m1 = m2 = 0.5;
	}
    for (j=0;j<3;++j) kdn->r[j] = ifMass*(m1*p1->r[j] + m2*p2->r[j]);
    if(p1->fSoft2 == 0.0 || p2->fSoft2 == 0.0)
	kdn->fSoft2 = 0.0;
    else
#if defined(TEST_SOFTENING)
	kdn->fSoft2 = p1->fSoft2 > p2->fSoft2 ? p1->fSoft2 : p2->fSoft2;
#else
    	kdn->fSoft2 = 1.0/(ifMass*(m1/p1->fSoft2 + m2/p2->fSoft2));
#endif
    BND_COMBINE(bnd,p1bnd,p2bnd);
    }


void kdCombineCells2(KD kd,KDN *kdn,KDN *p1,KDN *p2) {
    FMOMR mom;
    float x,y,z;

    /*
    ** Now calculate the reduced multipole moment.
    ** Shift the multipoles of each of the children
    ** to the CoM of this cell and add them up.
    */
    if (kd->oNodeMom) {
	*kdNodeMom(kd,kdn) = *kdNodeMom(kd,p1);
	x = p1->r[0] - kdn->r[0];
	y = p1->r[1] - kdn->r[1];
	z = p1->r[2] - kdn->r[2];
	momShiftFmomr(kdNodeMom(kd,kdn),p1->bMax,x,y,z);

	momRescaleFmomr(kdNodeMom(kd,kdn),kdn->bMax,p1->bMax);

	mom = *kdNodeMom(kd,p2);
	x = p2->r[0] - kdn->r[0];
	y = p2->r[1] - kdn->r[1];
	z = p2->r[2] - kdn->r[2];
	momShiftFmomr(&mom,p2->bMax,x,y,z);

	momScaledAddFmomr(kdNodeMom(kd,kdn),kdn->bMax,&mom,p2->bMax);

	}
}

void kdCalcBound(KD kd,BND *pbnd) {
    double dMin[3],dMax[3];
    PARTICLE *p;
    int i = 0;
    int j;

    assert(kd->nStore > 0);
    p = kdParticle(kd,i);
    for (j=0;j<3;++j) {
	dMin[j] = p->r[j];
	dMax[j] = p->r[j];
	}
    for (++i;i<kd->nStore;++i) {
	p = kdParticle(kd,i);
	kdMinMax(p->r,dMin,dMax);
	}
    for (j=0;j<3;++j) {
	pbnd->fCenter[j] = kd->bnd.fCenter[j] = 0.5*(dMin[j] + dMax[j]);
	pbnd->fMax[j] = kd->bnd.fMax[j] = 0.5*(dMax[j] - dMin[j]);
	}
    }

void kdTreeBuild(KD kd,int nBucket) {
    int iStart;
    BND bnd;

    kdCalcBound(kd, &bnd);
    InitializeParticles(kd,&kd->bnd);

    BuildTemp(kd,ROOT,nBucket);
    kd->nNodesFull = kd->nNodes;

    iStart = 0;
    ShuffleParticles(kd,iStart);
    Create(kd,ROOT);
    }


/*
** Hopefully we can bypass this step once we figure out how to do the
** Multipole Ewald with reduced multipoles.
*/
void kdCalcRoot(KD kd,MOMC *pmom) {
    PARTICLE *p;
    FLOAT xr = kdTopNode(kd,ROOT)->r[0];
    FLOAT yr = kdTopNode(kd,ROOT)->r[1];
    FLOAT zr = kdTopNode(kd,ROOT)->r[2];
    FLOAT x,y,z;
    FLOAT fMass;
    MOMC mc;
    int i = 0;

    p = kdParticle(kd,i);
    x = p->r[0] - xr;
    y = p->r[1] - yr;
    z = p->r[2] - zr;
    fMass = p->fMass;
    momMakeMomc(pmom,fMass,x,y,z);
    for (++i;i<kd->nStore;++i) {
	p = kdParticle(kd,i);
	fMass = p->fMass;
	x = p->r[0] - xr;
	y = p->r[1] - yr;
	z = p->r[2] - zr;
	momMakeMomc(&mc,fMass,x,y,z);
	momAddMomc(pmom,&mc);
	}
    }


void kdDistribRoot(KD kd,MOMC *pmom) {
    kd->momRoot = *pmom;
    }


void kdFinish(KD kd) {
    int i;

    if (kd->kdNodeListPRIVATE) {
	/*
	** Close caching space and free up nodes.
	*/
	for( i=0; i<kd->nTreeTiles; i++)
	    free(kd->kdNodeListPRIVATE[i]);
	free(kd->kdNodeListPRIVATE);
	}
    /*
    ** Free Interaction lists.
    */
    free(kd->ilp);
    free(kd->ilc);
    if (kd->kdTopPRIVATE) free(kd->kdTopPRIVATE);
    /*free(kd->ew.ewt);*/
    free(kd->pStorePRIVATE);
    free(kd->pTempPRIVATE);
    free(kd);
    }

