#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdlib.h>
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
#include <stddef.h>
#include <assert.h>
#include <time.h>
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#include "kd.h"
#include "moments.h"


#define SQRT1(d2,dir)				\
    {						\
	dir = 1/sqrt(d2);			\
	}

/*
** This version of grav.c does all the operations inline, including
** v_sqrt's and such.
** Returns nActive.
*/
void kdGravInteract(KD kd, KDN *pBucket,ILP ilp,ILC ilc,int bEwald,PARTICLE *p) {
    KDN *kdn = pBucket;
    pBND bnd;
    float ax,ay,az,fPot;
    float d2,dir,dir2;
    float fMass,fSoft;
    float fx, fy, fz;
    float tax,tay,taz;
    const float onethird = 1.0f/3.0f;
    float u,g0,g2,g3,g4;
    float x,y,z;
    float tx,ty,tz;
    float xx,xy,xz,yy,yz,zz;
    float xxx,xxz,yyy,yyz,xxy,xyy,xyz;
    ILPTILE tile;
    ILCTILE ctile;
    int j,nSoft;
    float fourh2;

    kdNodeBnd(kd, kdn, &bnd);

    /*
    ** Process the two interaction lists for each active particle.
    */
    fMass = p->fMass;
    fSoft = p->fSoft;
    nSoft = 0;
    
    fx = p->r[0] - ilc->cx;
    fy = p->r[1] - ilc->cy;
    fz = p->r[2] - ilc->cz;
    
    ilc->cx = p->r[0]; /* => cx += fx */
    ilc->cy = p->r[1];
    ilc->cz = p->r[2];
    ilcCompute(ilc,fx,fy,fz);
    ILC_LOOP(ilc,ctile) {
	for (j=0;j<ctile->nCell;++j) {
	    SQRT1(ctile->d2.f[j],dir);
	    u = ctile->u.f[j]*dir;
	    g0 = dir;
	    g2 = 3*dir*u*u;
	    g3 = 5*g2*u;
	    g4 = 7*g3*u;
	    /*
	    ** Calculate the funky distance terms.
	    */
	    x = ctile->dx.f[j]*dir;
	    y = ctile->dy.f[j]*dir;
	    z = ctile->dz.f[j]*dir;
	    xx = 0.5*x*x;
	    xy = x*y;
	    xz = x*z;
	    yy = 0.5*y*y;
	    yz = y*z;
	    zz = 0.5*z*z;
	    xxx = x*(onethird*xx - zz);
	    xxz = z*(xx - onethird*zz);
	    yyy = y*(onethird*yy - zz);
	    yyz = z*(yy - onethird*zz);
	    xx -= zz;
	    yy -= zz;
	    xxy = y*xx;
	    xyy = x*yy;
	    xyz = xy*z;
	    /*
	    ** Now calculate the interaction up to Hexadecapole order.
	    */
	    tx = g4*(ctile->xxxx.f[j]*xxx + ctile->xyyy.f[j]*yyy + ctile->xxxy.f[j]*xxy + ctile->xxxz.f[j]*xxz + ctile->xxyy.f[j]*xyy + ctile->xxyz.f[j]*xyz + ctile->xyyz.f[j]*yyz);
	    ty = g4*(ctile->xyyy.f[j]*xyy + ctile->xxxy.f[j]*xxx + ctile->yyyy.f[j]*yyy + ctile->yyyz.f[j]*yyz + ctile->xxyy.f[j]*xxy + ctile->xxyz.f[j]*xxz + ctile->xyyz.f[j]*xyz);
	    tz = g4*(-ctile->xxxx.f[j]*xxz - (ctile->xyyy.f[j] + ctile->xxxy.f[j])*xyz - ctile->yyyy.f[j]*yyz + ctile->xxxz.f[j]*xxx + ctile->yyyz.f[j]*yyy - ctile->xxyy.f[j]*(xxz + yyz) + ctile->xxyz.f[j]*xxy + ctile->xyyz.f[j]*xyy);
	    g4 = 0.25*(tx*x + ty*y + tz*z);
	    xxx = g3*(ctile->xxx.f[j]*xx + ctile->xyy.f[j]*yy + ctile->xxy.f[j]*xy + ctile->xxz.f[j]*xz + ctile->xyz.f[j]*yz);
	    xxy = g3*(ctile->xyy.f[j]*xy + ctile->xxy.f[j]*xx + ctile->yyy.f[j]*yy + ctile->yyz.f[j]*yz + ctile->xyz.f[j]*xz);
	    xxz = g3*(-(ctile->xxx.f[j] + ctile->xyy.f[j])*xz - (ctile->xxy.f[j] + ctile->yyy.f[j])*yz + ctile->xxz.f[j]*xx + ctile->yyz.f[j]*yy + ctile->xyz.f[j]*xy);
	    g3 = onethird*(xxx*x + xxy*y + xxz*z);
	    xx = g2*(ctile->xx.f[j]*x + ctile->xy.f[j]*y + ctile->xz.f[j]*z);
	    xy = g2*(ctile->yy.f[j]*y + ctile->xy.f[j]*x + ctile->yz.f[j]*z);
	    xz = g2*(-(ctile->xx.f[j] + ctile->yy.f[j])*z + ctile->xz.f[j]*x + ctile->yz.f[j]*y);
	    g2 = 0.5*(xx*x + xy*y + xz*z);
	    g0 *= ctile->m.f[j];
	    fPot -= g0 + g2 + g3 + g4;
	    g0 += 5*g2 + 7*g3 + 9*g4;
	    tax = dir*(xx + xxx + tx - x*g0);
	    tay = dir*(xy + xxy + ty - y*g0);
	    taz = dir*(xz + xxz + tz - z*g0);
	    ax += tax;
	    ay += tay;
	    az += taz;
	    }
	} /* end of cell list gravity loop */

    ILP_LOOP(ilp,tile) {
	for (j=0;j<tile->nPart;++j) {
	    d2 = tile->d2.f[j];
	    fourh2 = softmassweight(fMass,4*fSoft*fSoft,
				    tile->m.f[j],tile->fourh2.f[j]);
	    if (d2 > fourh2) {
		SQRT1(d2,dir);
		dir2 = dir*dir*dir;
		}
	    else {
		/*
		** This uses the Dehnen K1 kernel function now, it's fast!
		*/
		SQRT1(fourh2,dir);
		dir2 = dir*dir;
		d2 *= dir2;
		dir2 *= dir;
		d2 = 1 - d2;
		dir *= 1.0 + d2*(0.5 + d2*(3.0/8.0 + d2*(45.0/32.0)));
		dir2 *= 1.0 + d2*(1.5 + d2*(135.0/16.0));
		++nSoft;
		}
	    dir2 *= tile->m.f[j];
	    tax = -tile->dx.f[j]*dir2;
	    tay = -tile->dy.f[j]*dir2;
	    taz = -tile->dz.f[j]*dir2;
	    fPot -= tile->m.f[j]*dir;
	    ax += tax;
	    ay += tay;
	    az += taz;
	    }
	} /* end of particle list gravity loop */
    /*
    ** Finally set new acceleration and potential.
    ** Note that after this point we cannot use the new timestepping criterion since we
    ** overwrite the acceleration.
    ** WANT TO RETURN THESE
    */
    p->fPot = fPot;
    p->a[0] = ax;
    p->a[1] = ay;
    p->a[2] = az;
    /*
    ** Now finally calculate the Ewald correction for this particle, if it is
    ** required.
    /
    if (bEwald) {
	kdParticleEwald(kd,uRungLo,uRungHi,p);
	}
    */
    }
