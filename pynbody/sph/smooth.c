#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "smooth.h"
#include "kd.h"


int smInit(SMX *psmx,KD kd,int nSmooth,float *fPeriod)
{
	SMX smx;
	KDN *root;
	int pi,j;
	int bError=0;

	root = &kd->kdNodes[ROOT];
	assert(root != NULL);
	/*
	 ** Check to make sure that the bounds of the simulation agree 
	 ** with the period specified, if not cause an error.
	 */
	for (j=0;j<3;++j) {
		if (root->bnd.fMax[j] - root->bnd.fMin[j] > fPeriod[j]) {
			fprintf(stderr,"ERROR(smInit):Bounds of the simulation volume exceed ");
			fprintf(stderr,"the period specified in the %c-dimension. (%f > %f)\n",'x'+j, root->bnd.fMax[j] - root->bnd.fMin[j], fPeriod[j]);
			bError = 1;
			}
		}
	if (bError) exit(1);
	assert(nSmooth <= kd->nActive);
	smx = (SMX)malloc(sizeof(struct smContext)); assert(smx != NULL);
	smx->kd = kd;
	smx->nSmooth = nSmooth;
	smx->pq = (PQ *)malloc(nSmooth*sizeof(PQ));                     assert(smx->pq != NULL);
	PQ_INIT(smx->pq,nSmooth);
	smx->pfBall2 = (float *)malloc((kd->nActive+1)*sizeof(int));    assert(smx->pfBall2 != NULL);
	smx->iMark = (char *)malloc(kd->nActive*sizeof(char));          assert(smx->iMark != NULL);
	smx->nListSize = smx->nSmooth+RESMOOTH_SAFE;
	smx->fList = (float *)malloc(smx->nListSize*sizeof(float));     assert(smx->fList != NULL);
	smx->pList = (int *)malloc(smx->nListSize*sizeof(int));         assert(smx->pList != NULL);
	/*
	 ** Set for Periodic Boundary Conditions.
	 */
	for (j=0;j<3;++j) smx->fPeriod[j] = fPeriod[j];
	/*
	 ** Initialize arrays for calculated quantities.
	 */
	for (pi=0;pi<smx->kd->nActive;++pi) {
		smx->kd->p[pi].fDensity = 0.0;
		for (j=0;j<3;++j) smx->kd->p[pi].vMean[j] = 0.0;
		smx->kd->p[pi].fVel2 = 0.0;
		smx->kd->p[pi].fDivv = 0.0;
		}
	*psmx = smx;	
	return(1);
	}


void smFinish(SMX smx)
{
	free(smx->pfBall2);
	free(smx->iMark);
	free(smx->pq);
	free(smx->fList);
	free(smx->pList);
	free(smx);
	}


void smBallSearch(SMX smx,float fBall2,float *ri)
{
	KDN *c;
	PARTICLE *p;
	int cell,cp,ct,pj;
	float fDist2,dx,dy,dz,lx,ly,lz,sx,sy,sz,x,y,z;
	PQ *pq;

	c = smx->kd->kdNodes;
	p = smx->kd->p;
	pq = smx->pqHead;
	x = ri[0];
	y = ri[1];
	z = ri[2];
	lx = smx->fPeriod[0];
	ly = smx->fPeriod[1];
	lz = smx->fPeriod[2];
	cell = ROOT;
	/*
	 ** First find the "local" Bucket.
	 ** This could mearly be the closest bucket to ri[3].
	 */
	while (cell < smx->kd->nSplit) {
		if (ri[c[cell].iDim] < c[cell].fSplit) cell = LOWER(cell);
		else cell = UPPER(cell);
		}
	/*
	 ** Now start the search from the bucket given by cell!
	 */
	for (pj=c[cell].pLower;pj<=c[cell].pUpper;++pj) {
		dx = x - p[pj].r[0];
		dy = y - p[pj].r[1];
		dz = z - p[pj].r[2];
		fDist2 = dx*dx + dy*dy + dz*dz;
		if (fDist2 < fBall2) {
			if (smx->iMark[pj]) continue;
			smx->iMark[pq->p] = 0;
			smx->iMark[pj] = 1;
			pq->fKey = fDist2;
			pq->p = pj;
			pq->ax = 0.0;
			pq->ay = 0.0;
			pq->az = 0.0;
			PQ_REPLACE(pq);
			fBall2 = pq->fKey;
			}
		}
	while (cell != ROOT) {
		cp = SIBLING(cell);
		ct = cp;
		SETNEXT(ct);
		while (1) {
			INTERSECT(c,cp,fBall2,lx,ly,lz,x,y,z,sx,sy,sz);
			/*
			 ** We have an intersection to test.
			 */
			if (cp < smx->kd->nSplit) {
				cp = LOWER(cp);
				continue;
				}
			else {
				for (pj=c[cp].pLower;pj<=c[cp].pUpper;++pj) {
					dx = sx - p[pj].r[0];
					dy = sy - p[pj].r[1];
					dz = sz - p[pj].r[2];
					fDist2 = dx*dx + dy*dy + dz*dz;
					if (fDist2 < fBall2) {
						if (smx->iMark[pj]) continue;
						smx->iMark[pq->p] = 0;
						smx->iMark[pj] = 1;
						pq->fKey = fDist2;
						pq->p = pj;
						pq->ax = sx - x;
						pq->ay = sy - y;
						pq->az = sz - z;
						PQ_REPLACE(pq);
						fBall2 = pq->fKey;
						}
					}
				}
		GetNextCell:
			SETNEXT(cp);
			if (cp == ct) break;
			}
		cell = PARENT(cell);
		}
	smx->pqHead = pq;
	}


int smBallGather(SMX smx,float fBall2,float *ri)
{
	KDN *c;
	PARTICLE *p;
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
				dx = sx - p[pj].r[0];
				dy = sy - p[pj].r[1];
				dz = sz - p[pj].r[2];
				fDist2 = dx*dx + dy*dy + dz*dz;
				if (fDist2 <= fBall2) {
				  if(nCnt>=smx->nListSize) {
				    if(!smx->warnings) fprintf(stderr, "Smooth - particle cache too small for local density - results will be incorrect\n");
				    smx->warnings=true;
				    break;
				  }
				  smx->fList[nCnt] = fDist2;
				  smx->pList[nCnt++] = pj;
				  
				}
				
			}
			
			}
	GetNextCell:
		SETNEXT(cp);
		if (cp == ROOT) break;
		}
	assert(nCnt <= smx->nListSize);
	return(nCnt);
	}


void smSmoothInitStep(SMX smx)
{
	KDN *c;
	PARTICLE *p;
    PQ *pq,*pqLast;
	int pi,pin,pj,pNext,nSmooth;
    float ax,ay,az;

	c = smx->kd->kdNodes;
	p = smx->kd->p;
	for (pi=0;pi<smx->kd->nActive;++pi) {
        smx->pfBall2[pi] = -1.0;
		}
	smx->pfBall2[smx->kd->nActive] = -1.0; /* stop condition */
	for (pi=0;pi<smx->kd->nActive;++pi) {
		smx->iMark[pi] = 0;
		}
	pqLast = &smx->pq[smx->nSmooth-1];
	nSmooth = smx->nSmooth;
	/*
	 ** Initialize Priority Queue.
	 */
	pin = 0;
	pNext = 1;
	ax = 0.0;
	ay = 0.0;
	az = 0.0;
	for (pq=smx->pq,pj=0;pq<=pqLast;++pq,++pj) {
		smx->iMark[pj] = 1;
		pq->p = pj;
		pq->ax = ax;
		pq->ay = ay;
		pq->az = az;
		}

    smx->pin = pin;
    smx->pNext = pNext;
    smx->ax = ax;
    smx->ay = ay;
    smx->az = az;
	}

int smSmoothStep(SMX smx,void (*fncSmooth)(SMX,int,int,int *,float *))
{
       KDN *c;
       PARTICLE *p;
       PQ *pq,*pqLast;
       int cell;
       int pi,pin,pj,pNext,nCnt,nSmooth;
       float dx,dy,dz,x,y,z,h2,ax,ay,az;

       c = smx->kd->kdNodes;
       p = smx->kd->p;
       pqLast = &smx->pq[smx->nSmooth-1];
       nSmooth = smx->nSmooth;
       pin = smx->pin;
       pNext = smx->pNext;
       ax = smx->ax;
       ay = smx->ay;
       az = smx->az;

	while (1) {
		if (smx->pfBall2[pin] >= 0) {
			/*
			 ** Find next particle which is not done, and load the
			 ** priority queue with nSmooth number of particles.
			 */
			while (smx->pfBall2[pNext] >= 0) ++pNext;
			/*
			 ** Check if we are really finished.
			 */
			if (pNext == smx->kd->nActive) return 0;
			pi = pNext;
			++pNext;
			x = p[pi].r[0];
			y = p[pi].r[1];
			z = p[pi].r[2];
			/*
			 ** First find the "local" Bucket.
			 ** This could mearly be the closest bucket to ri[3].
			 */
			cell = ROOT;
			while (cell < smx->kd->nSplit) {
				if (p[pi].r[c[cell].iDim] < c[cell].fSplit)
					cell = LOWER(cell);
				else
					cell = UPPER(cell);
				}
			/*
			 ** Remove everything from the queue.
			 */
			smx->pqHead = NULL;
			for (pq=smx->pq;pq<=pqLast;++pq) smx->iMark[pq->p] = 0;
			/*
			 ** Add everything from pj up to and including pj+nSmooth-1.
			 */
			pj = c[cell].pLower;
			if (pj > smx->kd->nActive - nSmooth)
				pj = smx->kd->nActive - nSmooth;
			for (pq=smx->pq;pq<=pqLast;++pq) {
				smx->iMark[pj] = 1;
				dx = x - p[pj].r[0];
				dy = y - p[pj].r[1];
				dz = z - p[pj].r[2];
				pq->fKey = dx*dx + dy*dy + dz*dz;
				pq->p = pj++;
				pq->ax = 0.0;
				pq->ay = 0.0;
				pq->az = 0.0;
				}
			PQ_BUILD(smx->pq,nSmooth,smx->pqHead);
			}
		else {
			/*
			 ** Calculate the priority queue using the previous particles!
			 */
			pi = pin;
			x = p[pi].r[0];
			y = p[pi].r[1];
			z = p[pi].r[2];
			smx->pqHead = NULL;
			for (pq=smx->pq;pq<=pqLast;++pq) {
				pq->ax -= ax;
				pq->ay -= ay;
				pq->az -= az;
				dx = x + pq->ax - p[pq->p].r[0];
				dy = y + pq->ay - p[pq->p].r[1];
				dz = z + pq->az - p[pq->p].r[2];
				pq->fKey = dx*dx + dy*dy + dz*dz;
				}
			PQ_BUILD(smx->pq,nSmooth,smx->pqHead);
			ax = 0.0;
			ay = 0.0;
			az = 0.0;
			}
		smBallSearch(smx,smx->pqHead->fKey,p[pi].r);
		smx->pfBall2[pi] = smx->pqHead->fKey;
		p[pi].fSmooth = 0.5*sqrt(smx->pfBall2[pi]);
		/*
		 ** Pick next particle, 'pin'.
		 ** Create fList and pList for function 'fncSmooth'.
		 */
		pin = pi;
		nCnt = 0;
		h2 = smx->pqHead->fKey;
		for (pq=smx->pq;pq<=pqLast;++pq) {

                  /* the next line is commented out because it results in the furthest
                     particle being excluded from the nearest-neighbor list - this means
                     that although the user requests 32 NN, she would get back 31. By
                     including the furthest particle, the neighbor list is always 32 long */

                  //if (pq == smx->pqHead) continue;
		        if(nCnt>=smx->nListSize) {
			  // no room left
			  if(!smx->warnings) fprintf(stderr, "Smooth - particle cache too small for local density - results will be incorrect\n");
			  smx->warnings = false;
			  break;
			}

			smx->pList[nCnt] = pq->p;
			smx->fList[nCnt++] = pq->fKey;
			
			if (smx->pfBall2[pq->p] >= 0) continue;
			if (pq->fKey < h2) {
				pin = pq->p;
				h2 = pq->fKey;
				ax = pq->ax;
				ay = pq->ay;
				az = pq->az;
				}
			
		}
		
		//(*fncSmooth)(smx,pi,nCnt,smx->pList,smx->fList);

        smx->pi = pi;
        smx->pin = pin;
        smx->pNext = pNext;
        smx->ax = ax;
        smx->ay = ay;
        smx->az = az;

        return nCnt;
		}
	}


void smDensitySym(SMX smx,int pi,int nSmooth,int *pList,float *fList)
{
	float fNorm,ih2,r2,rs;
	int i,pj;

        ih2 = 4.0/smx->pfBall2[pi];
        fNorm = 0.5*M_1_PI*sqrt(ih2)*ih2;
	for (i=0;i<nSmooth;++i) {
                pj = pList[i];
		r2 = fList[i]*ih2;
                rs = 2.0 - sqrt(r2);
		if (r2 < 1.0) rs = (1.0 - 0.75*rs*r2);
		else rs = 0.25*rs*rs*rs;
		if(rs<0) {
		  fprintf(stderr, "Internal consistency error\n");
		  smx->warnings=true;
		}
		rs *= fNorm;
		smx->kd->p[pi].fDensity += rs*smx->kd->p[pj].fMass;
		smx->kd->p[pj].fDensity += rs*smx->kd->p[pi].fMass;
        }
}



void smMeanVel(SMX smx,int pi,int nSmooth,int *pList,float *fList)
{
	float fNorm,ih2,r2,rs;
	int i,j,pj;

	ih2 = 4.0/smx->pfBall2[pi];
	fNorm = M_1_PI*sqrt(ih2)*ih2;
	for (i=0;i<nSmooth;++i) {
		pj = pList[i];
		r2 = fList[i]*ih2;
		rs = 2.0 - sqrt(r2);
		if (r2 < 1.0) rs = (1.0 - 0.75*rs*r2);
		else rs = 0.25*rs*rs*rs;
		rs *= fNorm;
		for (j=0;j<3;++j) {
			smx->kd->p[pi].vMean[j] += rs*smx->kd->p[pj].fMass/
				smx->kd->p[pj].fDensity*smx->kd->p[pj].v[j];
			}
		}
	}

void smMeanVelSym(SMX smx,int pi,int nSmooth,int *pList,float *fList)
{
	float fNorm,ih2,r2,rs;
	int i,j,pj;

	ih2 = 4.0/smx->pfBall2[pi];
	fNorm = 0.5*M_1_PI*sqrt(ih2)*ih2;
	for (i=0;i<nSmooth;++i) {
		pj = pList[i];
		r2 = fList[i]*ih2;
		rs = 2.0 - sqrt(r2);
		if (r2 < 1.0) rs = (1.0 - 0.75*rs*r2);
		else rs = 0.25*rs*rs*rs;
		rs *= fNorm;
		for (j=0;j<3;++j) {
			smx->kd->p[pi].vMean[j] += rs*smx->kd->p[pj].fMass/
				smx->kd->p[pj].fDensity*smx->kd->p[pj].v[j];
			smx->kd->p[pj].vMean[j] += rs*smx->kd->p[pi].fMass/
				smx->kd->p[pi].fDensity*smx->kd->p[pi].v[j];
			}
		}
	}

void smDivvSym(SMX smx,int pi,int nSmooth,int *pList,float *fList)
{
	float fNorm,ih2,r2,rs;
	float r, rs1, dvdotdr, fNorm1;
	int i,j,pj;

	ih2 = 4.0/smx->pfBall2[pi];
	fNorm = 0.5*M_1_PI*sqrt(ih2)*ih2;
	fNorm1 = fNorm*ih2;
	for (i=0;i<nSmooth;++i) {
		pj = pList[i];
		r2 = fList[i]*ih2;
		r = sqrt(r2);
		rs = 2.0 - r;
		if (r2 < 1.0) {
			rs1 = -3 + 2.25*r;
			}
		else {
			rs1 = -0.75*rs*rs/r;
			}
		rs1 *= fNorm1;
		dvdotdr = 0.0;
		for (j=0;j<3;++j) {
			dvdotdr += (smx->kd->p[pj].v[j] - smx->kd->p[pi].v[j])*
				(smx->kd->p[pj].r[j] - smx->kd->p[pi].r[j]);
			}
		smx->kd->p[pi].fDivv -= rs1*smx->kd->p[pj].fMass/
			smx->kd->p[pj].fDensity*dvdotdr;
		smx->kd->p[pj].fDivv -= rs1*smx->kd->p[pi].fMass/
			smx->kd->p[pi].fDensity*dvdotdr;
		}
	}

void smVelDisp(SMX smx,int pi,int nSmooth,int *pList,float *fList)
{
	float fNorm,ih2,r2,rs,tv2;
	int i,j,pj;

	ih2 = 4.0/smx->pfBall2[pi];
	fNorm = M_1_PI*sqrt(ih2)*ih2;
	for (i=0;i<nSmooth;++i) {
		pj = pList[i];
		r2 = fList[i]*ih2;
		rs = 2.0 - sqrt(r2);
		if (r2 < 1.0) rs = (1.0 - 0.75*rs*r2);
		else rs = 0.25*rs*rs*rs;
		rs *= fNorm;
		tv2 = 0.0;
		for (j=0;j<3;++j) {
                  tv2 += (smx->kd->p[pj].v[j] - smx->kd->p[pi].vMean[j])*
                    (smx->kd->p[pj].v[j] - smx->kd->p[pi].vMean[j]);
                }
		smx->kd->p[pi].fVel2 += rs*smx->kd->p[pj].fMass/smx->kd->p[pj].fDensity*tv2;
        }
}	



void smVelDispSym(SMX smx,int pi,int nSmooth,int *pList,float *fList)
{
	float fNorm,ih2,r2,rs,tv2;
	int i,j,pj;

	ih2 = 4.0/smx->pfBall2[pi];
	fNorm = 0.5*M_1_PI*sqrt(ih2)*ih2;
	for (i=0;i<nSmooth;++i) {
		pj = pList[i];
		r2 = fList[i]*ih2;
		rs = 2.0 - sqrt(r2);
		if (r2 < 1.0) rs = (1.0 - 0.75*rs*r2);
		else rs = 0.25*rs*rs*rs;
		rs *= fNorm;
		tv2 = 0.0;
		for (j=0;j<3;++j) {
			tv2 += (smx->kd->p[pj].v[j] - smx->kd->p[pi].vMean[j])*
				(smx->kd->p[pj].v[j] - smx->kd->p[pi].vMean[j]);
			}
		smx->kd->p[pi].fVel2 += rs*smx->kd->p[pj].fMass/
			smx->kd->p[pj].fDensity*tv2;
		tv2 = 0.0;
		for (j=0;j<3;++j) {
			tv2 += (smx->kd->p[pi].v[j] - smx->kd->p[pj].vMean[j])*
				(smx->kd->p[pi].v[j] - smx->kd->p[pj].vMean[j]);
			}
		smx->kd->p[pj].fVel2 += rs*smx->kd->p[pi].fMass/
			smx->kd->p[pi].fDensity*tv2;
		}
	}

void smVelDispNBSym(SMX smx,int pi,int nSmooth,int *pList,float *fList)
{
	float fNorm,ih2,r2,rs,tv2;
	float dr;
	int i,j,pj;

	ih2 = 4.0/smx->pfBall2[pi];
	fNorm = 0.5*M_1_PI*sqrt(ih2)*ih2;
	for (i=0;i<nSmooth;++i) {
		pj = pList[i];
		r2 = fList[i]*ih2;
		rs = 2.0 - sqrt(r2);
		if (r2 < 1.0) rs = (1.0 - 0.75*rs*r2);
		else rs = 0.25*rs*rs*rs;
		rs *= fNorm;
		tv2 = 0.0;
		for (j=0;j<3;++j) {
			dr = smx->kd->p[pj].r[j] - smx->kd->p[pi].r[j];
			tv2 += (smx->kd->p[pj].v[j] - smx->kd->p[pi].vMean[j] -
					smx->kd->p[pi].fDivv*dr)*
				(smx->kd->p[pj].v[j] - smx->kd->p[pi].vMean[j] -
				 smx->kd->p[pi].fDivv*dr);
			}
		smx->kd->p[pi].fVel2 += rs*smx->kd->p[pj].fMass/
			smx->kd->p[pj].fDensity*tv2;
		smx->kd->p[pj].fVel2 += rs*smx->kd->p[pi].fMass/
			smx->kd->p[pi].fDensity*tv2;
		}
	}	
