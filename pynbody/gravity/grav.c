/*
 * Gravity code stolen from Zurich group's (Joachim Stadel, Doug Potter,
 * Jonathan Coles, et al) amazing gravity code pkdgrav2 and severly
 * bastardized and converted from parallel to serial for pynbody by
 * Greg Stinson with help from Jonathan, Tom Quinn, Rok Roskar and
 * Andrew Pontzen.
 */

#include "config.h"

#include <math.h>
#include <stdlib.h>
#if defined(__APPLE__)
#undef HAVE_MALLOC_H
#endif
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
#include <stddef.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
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
void kdGravInteract(KD kd,KDN *pBucket, ILP *ilp,int nPart,
		   ILC *ilc,int nCell, int bEwald, PARTICLE *p) 
{
    KDN *kdn = pBucket;
    pBND kbnd;
    const double onethird = 1.0/3.0;
    float fPot;
    double ax,ay,az;
    double x,y,z,d2,dir,dir2,g2,g3,g4;
    double xx,xy,xz,yy,yz,zz;
    double xxx,xxz,yyy,yyz,xxy,xyy,xyz;
    double tx,ty,tz;
    double fourh2;
    momFloat tax,tay,taz;
#ifdef SOFTSQUARE
    double ptwoh2;
#endif
    int j, nSoft;
    double three_eighths = 3.0/8.0;
    double ffbytt = 45.0/32.0;
    double otfbysixteen = 135.0/16.0;
    kdNodeBnd(kd, kdn, &kbnd);

    printf("nPart: %d, nCell: %d\n",nPart,nCell);
    /*
    ** Now process the two interaction lists for each active particle.
    */
    nSoft = 0;
    ax = 0;
    ay = 0;
    az = 0;

    /*
     * Process cell interaction list
     */
    for (j=0;j<nCell;++j) {
	x = p->r[0] - ilc[j].x;
	y = p->r[1] - ilc[j].y;
	z = p->r[2] - ilc[j].z;
	d2 = x*x + y*y + z*z;
	SQRT1(d2,dir);
	dir2 = dir*dir;
	g2 = 3*dir*dir2*dir2;
	g3 = 5*g2*dir2;
	g4 = 7*g3*dir2;
	/*
	** Calculate the funky distance terms.
	*/
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
	tx = g4*(ilc[j].mom.xxxx*xxx + ilc[j].mom.xyyy*yyy + ilc[j].mom.xxxy*xxy + ilc[j].mom.xxxz*xxz + ilc[j].mom.xxyy*xyy + ilc[j].mom.xxyz*xyz + ilc[j].mom.xyyz*yyz);
	ty = g4*(ilc[j].mom.xyyy*xyy + ilc[j].mom.xxxy*xxx + ilc[j].mom.yyyy*yyy + ilc[j].mom.yyyz*yyz + ilc[j].mom.xxyy*xxy + ilc[j].mom.xxyz*xxz + ilc[j].mom.xyyz*xyz);
	tz = g4*(-ilc[j].mom.xxxx*xxz - (ilc[j].mom.xyyy + ilc[j].mom.xxxy)*xyz - ilc[j].mom.yyyy*yyz + ilc[j].mom.xxxz*xxx + ilc[j].mom.yyyz*yyy - ilc[j].mom.xxyy*(xxz + yyz) + ilc[j].mom.xxyz*xxy + ilc[j].mom.xyyz*xyy);
	g4 = 0.25*(tx*x + ty*y + tz*z);
	xxx = g3*(ilc[j].mom.xxx*xx + ilc[j].mom.xyy*yy + ilc[j].mom.xxy*xy + ilc[j].mom.xxz*xz + ilc[j].mom.xyz*yz);
	xxy = g3*(ilc[j].mom.xyy*xy + ilc[j].mom.xxy*xx + ilc[j].mom.yyy*yy + ilc[j].mom.yyz*yz + ilc[j].mom.xyz*xz);
	xxz = g3*(-(ilc[j].mom.xxx + ilc[j].mom.xyy)*xz - (ilc[j].mom.xxy + ilc[j].mom.yyy)*yz + ilc[j].mom.xxz*xx + ilc[j].mom.yyz*yy + ilc[j].mom.xyz*xy);
	g3 = onethird*(xxx*x + xxy*y + xxz*z);
	xx = g2*(ilc[j].mom.xx*x + ilc[j].mom.xy*y + ilc[j].mom.xz*z);
	xy = g2*(ilc[j].mom.yy*y + ilc[j].mom.xy*x + ilc[j].mom.yz*z);
	xz = g2*(-(ilc[j].mom.xx + ilc[j].mom.yy)*z + ilc[j].mom.xz*x + ilc[j].mom.yz*y);
	g2 = 0.5*(xx*x + xy*y + xz*z);
	dir *= ilc[j].mom.m;
	dir2 *= dir + 5*g2 + 7*g3 + 9*g4;
	fPot -= dir + g2 + g3 + g4;
	tax = xx + xxx + tx - x*dir2;
	tay = xy + xxy + ty - y*dir2;
	taz = xz + xxz + tz - z*dir2;
	ax += tax;
	ay += tay;
	az += taz;
	} /* end of cell list gravity loop */

    printf("after cells:  ax: %g ay: %g az: %g fPot: %g\n", ax, ay, az, fPot);
    /*
     * Process particle interaction list
     */
#ifdef SOFTSQUARE
    ptwoh2 = 2*kdSoft(kd,p)*kdSoft(kd,p);
#endif
    for (j=0;j<nPart;++j) {
	x = p->r[0] - ilp[j].x;
	y = p->r[1] - ilp[j].y;
	z = p->r[2] - ilp[j].z;
	d2 = x*x + y*y + z*z;
#ifdef SOFTLINEAR
	fourh2 = kdSoft(kd,p) + ilp[j].h;
	fourh2 *= fourh2;
#endif
#if !defined(SOFTLINEAR) && !defined(SOFTSQUARE)
#ifdef SOFTENING_NOT_MASS_WEIGHTED
	fourh2 = ilp[j].fourh2;
#else
	fourh2 = softmassweight(p->fMass,4*p->fSoft*p->fSoft,ilp[j].m,ilp[j].fourh2);
#endif
#endif
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
	    d2 = 1.0 - d2;
	    dir *= 1.0 + d2*(0.5 + d2*(three_eighths + d2*(ffbytt)));
	    dir2 *= 1.0 + d2*(1.5 + d2*(otfbysixteen));
	    ++nSoft;
	    }

	dir2 *= ilp[j].m;
	tax = -x*dir2;
	tay = -y*dir2;
	taz = -z*dir2;
	fPot -= ilp[j].m*dir;
	ax += tax;
	ay += tay;
	az += taz;
	} /* end of particle list gravity loop */
    printf("after particles:  ax: %g ay: %g az: %g fPot: %g\n", ax, ay, az, fPot);
    /*
    ** Finally set new acceleration and potential.
    */
    p->fPot = fPot;
    p->a[0] = ax;
    p->a[1] = ay;
    p->a[2] = az;
    /*
    ** Now finally calculate the Ewald correction for this particle, if it is
    ** required.
    if (bEwald) {
	*pdEwFlop += kdParticleEwald(kd,uRungLo,uRungHi,p);
	}
    */

    }


