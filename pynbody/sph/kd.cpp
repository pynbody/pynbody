#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "kd.h"


#define MAX_ROOT_ITTR	32



int kdInit(KD *pkd,int nBucket)
{
	KD kd;

	kd = (KD)malloc(sizeof(struct kdContext));
	assert(kd != NULL);
	kd->nBucket = nBucket;
	kd->p = NULL;
	kd->kdNodes = NULL;
	*pkd = kd;
	return(1);
}


void kdCombine(KDN *p1,KDN *p2,KDN *pOut)
{
	int j;

	/*
	 ** Combine the bounds.
	 */
	for (j=0;j<3;++j) {
		if (p2->bnd.fMin[j] < p1->bnd.fMin[j])
			pOut->bnd.fMin[j] = p2->bnd.fMin[j];
		else
			pOut->bnd.fMin[j] = p1->bnd.fMin[j];
		if (p2->bnd.fMax[j] > p1->bnd.fMax[j])
			pOut->bnd.fMax[j] = p2->bnd.fMax[j];
		else
			pOut->bnd.fMax[j] = p1->bnd.fMax[j];
		}
	}



int cmpParticles(const void *v1,const void *v2)
{
	PARTICLE *p1=(PARTICLE *)v1,*p2=(PARTICLE *)v2;

	return(p1->iOrder - p2->iOrder);
}


void kdOrder(KD kd)
{
	qsort(kd->p,kd->nActive,sizeof(PARTICLE),cmpParticles);
}


void kdFinish(KD kd)
{
	free(kd->p);
	free(kd->kdNodes);
	free(kd);
}



template<typename T>
void kdSelect(KD kd,int d,int k,int l,int r)
{
	PARTICLE *p,t;
	T v;
	int i,j;

	p = kd->p;
	while (r > l) {
		v = GET2<T>(kd->pNumpyPos,p[k].iOrder,d);
		t = p[r];
		p[r] = p[k];
		p[k] = t;
		i = l - 1;
		j = r;
		while (1) {
			while (i < j) if (GET2<T>(kd->pNumpyPos,p[++i].iOrder,d) >= v) break;
			while (i < j) if (GET2<T>(kd->pNumpyPos,p[--j].iOrder,d) <= v) break;
			t = p[i];
			p[i] = p[j];
			p[j] = t;
			if (j <= i) break;
			}
		p[j] = p[i];
		p[i] = p[r];
		p[r] = t;
		if (i >= k) r = i - 1;
		if (i <= k) l = i + 1;
	}
}


template<typename T>
void kdUpPass(KD kd,int iCell)
{
	KDN *c;
	int l,u,pj,j;
	double rj;
	c = kd->kdNodes;
	if (c[iCell].iDim != -1) {
		l = LOWER(iCell);
		u = UPPER(iCell);
		kdUpPass<T>(kd,l);
		kdUpPass<T>(kd,u);
		kdCombine(&c[l],&c[u],&c[iCell]);
		}
	else {
		l = c[iCell].pLower;
		u = c[iCell].pUpper;
		for (j=0;j<3;++j) {
			c[iCell].bnd.fMin[j] = GET2<T>(kd->pNumpyPos,kd->p[u].iOrder,j);
			c[iCell].bnd.fMax[j] = c[iCell].bnd.fMin[j];
			}
		for (pj=l;pj<u;++pj) {
			for (j=0;j<3;++j) {
				rj = GET2<T>(kd->pNumpyPos,kd->p[pj].iOrder,j);
				if (rj < c[iCell].bnd.fMin[j])
					c[iCell].bnd.fMin[j] = rj;
				if (rj > c[iCell].bnd.fMax[j])
					c[iCell].bnd.fMax[j] = rj;
				}
			}
		}
	}

template <typename T>
void kdBuildTree(KD kd)
{
	int l,n,i,j;
	T rj;
	BND bnd;

	n = kd->nActive;
	kd->nLevels = 1;
	l = 1;
	while (n > kd->nBucket) {
		n = n>>1;
		l = l<<1;
		++kd->nLevels;
		}
	kd->nSplit = l;
	kd->nNodes = l<<1;
	if (kd->kdNodes != NULL) free(kd->kdNodes);
	kd->kdNodes = (KDN *)malloc(kd->nNodes*sizeof(KDN));
	assert(kd->kdNodes != NULL);

	// Calculate bounds
	// Initialize with any particle:
	for (j=0;j<3;++j) {
		rj = GET2<T>(kd->pNumpyPos,kd->p[0].iOrder,j);
		bnd.fMin[j] = rj;
		bnd.fMax[j] = rj;
	}

	// Expand to enclose all particles:
	for (i=1;i<kd->nActive;++i) {
		for (j=0;j<3;++j) {
			rj = GET2<T>(kd->pNumpyPos,kd->p[i].iOrder,j);
			if (bnd.fMin[j] > rj)
				bnd.fMin[j] = rj;
			else if (bnd.fMax[j] < rj)
				bnd.fMax[j] = rj;
			}
		}

	// Set up root node
	kd->kdNodes[ROOT].pLower = 0;
	kd->kdNodes[ROOT].pUpper = kd->nActive-1;
	kd->kdNodes[ROOT].bnd = bnd;

	// Recursively build tree
	kdBuildNode<T>(kd, ROOT);

	// Calculate and store bounds information by passing it up the tree
	kdUpPass<T>(kd,ROOT);
}

struct KDargs {
	KD kd;
	int local_root;
};

template <typename T>
void kdBuildNodeRemote(struct KDargs *a) {
	kdBuildNode<T>(a->kd, a->local_root);
}

template <typename T>
void kdBuildNode(KD kd, int local_root) {

	int i=local_root;
	int d,j,m,diff;
	KDN *nodes;
	nodes = kd->kdNodes;

#ifdef KDT_THREADING
	pthread_t remote_thread;
	struct KDargs remote_args;
#endif

	while (1) {
		assert(nodes[i].pUpper - nodes[i].pLower + 1 > 0);
		if (i < kd->nSplit && (nodes[i].pUpper - nodes[i].pLower) > 0) {

			// Select splitting dimensions on the basis of keeping things
			// as square as possible
			d = 0;
			for (j=1;j<3;++j) {
				if (nodes[i].bnd.fMax[j]-nodes[i].bnd.fMin[j] >
					nodes[i].bnd.fMax[d]-nodes[i].bnd.fMin[d]) d = j;
				}
			nodes[i].iDim = d;

			// Find mid-point of particle list at which splitting will
			// ultimately take place
			m = (nodes[i].pLower + nodes[i].pUpper)/2;

			// Sort list to ensure particles between lower and m are to
			// the 'left' of particles between m and upper
			kdSelect<T>(kd,d,m,nodes[i].pLower,nodes[i].pUpper);

			// Note split point based on median particle
			nodes[i].fSplit = GET2<T>(kd->pNumpyPos,kd->p[m].iOrder,d);

			// Set up lower cell
			nodes[LOWER(i)].bnd = nodes[i].bnd;
			nodes[LOWER(i)].bnd.fMax[d] = nodes[i].fSplit;
			nodes[LOWER(i)].pLower = nodes[i].pLower;
			nodes[LOWER(i)].pUpper = m;

			// Set up upper cell
			nodes[UPPER(i)].bnd = nodes[i].bnd;
			nodes[UPPER(i)].bnd.fMin[d] = nodes[i].fSplit;
			nodes[UPPER(i)].pLower = m+1;
			nodes[UPPER(i)].pUpper = nodes[i].pUpper;
			diff = (m-nodes[i].pLower+1)-(nodes[i].pUpper-m);
			assert(diff == 0 || diff == 1);

#if 0
			// Threaded KDT build is disabled at the moment.
			//
			// It does work, but it turns out to be slower than working on a single
			// thread. This is presumably because the threads are writing to widely
			// different bits of memory and the CPU cache doesn't like it?

			if(i<=2) {
				// Launch a thread to handle the lower part of the tree,
				// handle the upper part on the current thread.
				printf("launch thread for id=%d...\n",LOWER(i));

				remote_args.kd = kd;
				remote_args.local_root = LOWER(i);


				// do lower part on another thread
				if(pthread_create(&remote_thread, NULL, kdBuildNodeRemote, &remote_args)) {
					fprintf(stderr,"Threading error");
					return;
				}

				// do upper part locally
				kdBuildNode(kd,UPPER(i));


				printf("join...");
				if(pthread_join(remote_thread,NULL)) {
					fprintf(stderr,"Threading error");
					return;
				}

				// at this point, we've finished processing this node and all subnodes,
				// so continue back up the tree

				SETNEXT(i,local_root);

			} else{
				// Continue processing on these threads
				// Next cell is the lower one. Upper one will be processed
				// on the way up.
				i = LOWER(i);
			}

#else
	// On single thread, always switch attention to the lower branch
			// (and upper branch gets processed on way up).
		i = LOWER(i);
#endif

		} else {
			// Cell does not need to be split. Mark as leaf
			nodes[i].iDim = -1;

			// Go back up the tree and process the UPPER cells where
			// necessary
			SETNEXT(i,local_root);
		}
		if (i == local_root) break; // We got back to the top, so we're done.
	}
}


// instantiate the actual functions that are available:

template
void kdSelect<double>(KD kd,int d,int k,int l,int r);

template
void kdUpPass<double>(KD kd,int iCell);

template
void kdBuildTree<double>(KD kd);

template
void kdBuildNodeRemote<double>(struct KDargs *a);

template
void kdBuildNode<double>(KD kd, int local_root);



template
void kdSelect<float>(KD kd,int d,int k,int l,int r);

template
void kdUpPass<float>(KD kd,int iCell);

template
void kdBuildTree<float>(KD kd);

template
void kdBuildNodeRemote<float>(struct KDargs *a);

template
void kdBuildNode<float>(KD kd, int local_root);
