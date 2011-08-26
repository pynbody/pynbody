#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "moments.h"

/*
 ** This function calculates a complete multipole from a single
 ** particle at position <x,y,z> from the center of mass.
 ** The strange order of evaluation reduces the number of
 ** multiplications to a minimum.
 ** <x,y,z> := d := r(particle) - rcm.
 **
 ** OpCount (*,+) = (34,0)
 **
 */
void momMakeMomc(MOMC *mc,momFloat m,momFloat x,momFloat y,momFloat z) {
    momFloat tx,ty,tz;

    mc->m = m;
    /*
     ** Calculate the Quadrupole Moment.
     */
    tx = m*x;
    ty = m*y;
    mc->xy = tx*y;
    mc->xz = tx*z;
    mc->yz = ty*z;
    tx *= x;
    ty *= y;
    tz = m*z*z;
    mc->xx = tx;
    mc->yy = ty;
    mc->zz = tz;
    /*
     ** Calculate the Octopole Moment.
     */
    mc->xxy = tx*y;
    mc->xxz = tx*z;
    mc->yyz = ty*z;
    mc->xyy = ty*x;
    mc->xzz = tz*x;
    mc->yzz = tz*y;
    mc->xyz = mc->xy*z;
    tx *= x;
    ty *= y;
    tz *= z;
    mc->xxx = tx;
    mc->yyy = ty;
    mc->zzz = tz;
    /*
     ** Calculate the Hexadecapole Moment.
     */
    mc->xxxx = tx*x;
    mc->xxxy = tx*y;
    mc->xxxz = tx*z;
    mc->xyyy = ty*x;
    mc->yyyy = ty*y;
    mc->yyyz = ty*z;
    mc->xzzz = tz*x;
    mc->yzzz = tz*y;
    mc->zzzz = tz*z;
    mc->xxyy = mc->xxy*y;
    mc->xxyz = mc->xxy*z;
    mc->xyyz = mc->yyz*x;
    mc->yyzz = mc->yyz*z;
    mc->xxzz = mc->xzz*x;
    mc->xyzz = mc->xzz*y;
    }


/*
 ** This function calculates a reduced multipole from a single
 ** particle at position <x,y,z> from the center of mass.
 ** The strange order of evaluation reduces the number of
 ** multiplications to a minimum.
 ** <x,y,z> := d := r(particle) - rcm.
 ** returns: d^2
 **
 ** OpCount (*,+) = (43,18) = ~60
 */
momFloat momMakeMomr(MOMR *mr,momFloat m,momFloat x,momFloat y,momFloat z) {
    momFloat tx,ty,t,dx,dy;
    momFloat x2 = x*x;
    momFloat y2 = y*y;
    momFloat d2 = x2 + y2 + z*z;

    /*assert( m > 0.0 ); -- massless tracer particles are allowed*/

    mr->m = m;
    /*
     ** Calculate the Quadrupole Moment.
     */
    tx = m*x;
    ty = m*y;
    mr->xy = tx*y;
    mr->xz = tx*z;
    mr->yz = ty*z;
    tx *= x;
    ty *= y;
    m *= d2;
    t = m/3;
    mr->xx = tx - t;
    mr->yy = ty - t;
    /*
     ** Calculate the Octopole Moment.
     */
    t = 0.2*m;
    dx = tx - t;
    dy = ty - t;
    mr->xxy = dx*y;
    mr->xxz = dx*z;
    mr->yyz = dy*z;
    mr->xyy = dy*x;
    mr->xyz = mr->xy*z;
    t *= 3;
    mr->xxx = (tx - t)*x;
    mr->yyy = (ty - t)*y;
    /*
     ** Calculate the Hexadecapole Moment.
     */
    t = m/7;
    mr->xxyz = (tx - t)*y*z;
    mr->xyyz = (ty - t)*x*z;
    dx = (tx - 3*t)*x;
    dy = (ty - 3*t)*y;
    mr->xxxy = dx*y;
    mr->xxxz = dx*z;
    mr->xyyy = dy*x;
    mr->yyyz = dy*z;
    dx = t*(x2 - 0.1*d2);
    dy = t*(y2 - 0.1*d2);
    mr->xxxx = tx*x2 - 6*dx;
    mr->yyyy = ty*y2 - 6*dy;
    mr->xxyy = tx*y2 - dx - dy;

    return(d2);
    }


/*
 ** This function calculates a reduced scaled multipole from a single
 ** particle at position <x,y,z> from the any type of "center". A scaling
 ** factor 'u' for the positions must also be specified, which should 
 ** typically be the value of b_max.
 **
 ** The strange order of evaluation reduces the number of
 ** multiplications to a minimum.
 ** <x,y,z> := d := r(particle) - rcm.
 ** returns: d^2 scaled by u^2.
 **
 ** OpCount (*,+) = (43,18) = ~60
 */
float momMakeFmomr(FMOMR *mr,float m,float u,float x,float y,float z) {
    float tx,ty,t,dx,dy;
    float x2;
    float y2;
    float d2,iu;

    assert(u > 0.0);
    iu = 1.0f/u;
    x *= iu;
    y *= iu;
    z *= iu;
    x2 = x*x;
    y2 = y*y;
    d2 = x2 + y2 + z*z;

    mr->m = m;
    tx = m*x;
    ty = m*y;
    /*
     ** Calculate the Quadrupole Moment.
     */
    mr->xy = tx*y;
    mr->xz = tx*z;
    mr->yz = ty*z;
    tx *= x;
    ty *= y;
    m *= d2;
    t = (1.0f/3.0f)*m;
    mr->xx = tx - t;
    mr->yy = ty - t;
    /*
     ** Calculate the Octopole Moment.
     */
    t = 0.2f*m;
    dx = tx - t;
    dy = ty - t;
    mr->xxy = dx*y;
    mr->xxz = dx*z;
    mr->yyz = dy*z;
    mr->xyy = dy*x;
    mr->xyz = mr->xy*z;
    t *= 3.0f;
    mr->xxx = (tx - t)*x;
    mr->yyy = (ty - t)*y;
    /*
     ** Calculate the Hexadecapole Moment.
     */
    t = (1.0f/7.0f)*m;
    mr->xxyz = (tx - t)*y*z;
    mr->xyyz = (ty - t)*x*z;
    dx = (tx - 3.0f*t)*x;
    dy = (ty - 3.0f*t)*y;
    mr->xxxy = dx*y;
    mr->xxxz = dx*z;
    mr->xyyy = dy*x;
    mr->yyyz = dy*z;
    dx = t*(x2 - 0.1f*d2);
    dy = t*(y2 - 0.1f*d2);
    mr->xxxx = tx*x2 - 6.0f*dx;
    mr->yyyy = ty*y2 - 6.0f*dy;
    mr->xxyy = tx*y2 - dx - dy;
    return(d2);
    }

/*
 ** This function calculates a reduced multipole from a single
 ** particle at position <x,y,z> from the center of mass.
 ** This is the "straight forward" implementation which we
 ** used in the original version of PKDGRAV. It remains a good
 ** test of more peculiar looking code.
 **
 ** <x,y,z> := d := r(particle) - rcm.
 **
 ** OpCount (*,+) = (115,20) = ~135
 */
void momOldMakeMomr(MOMR *mr,momFloat m,momFloat x,momFloat y,momFloat z) {
    momFloat d2 = x*x + y*y + z*z;

    mr->xxxx = m*(x*x*x*x - 6.0/7.0*d2*(x*x - 0.1*d2));
    mr->xyyy = m*(x*y*y*y - 3.0/7.0*d2*x*y);
    mr->xxxy = m*(x*x*x*y - 3.0/7.0*d2*x*y);
    mr->yyyy = m*(y*y*y*y - 6.0/7.0*d2*(y*y - 0.1*d2));
    mr->xxxz = m*(x*x*x*z - 3.0/7.0*d2*x*z);
    mr->yyyz = m*(y*y*y*z - 3.0/7.0*d2*y*z);
    mr->xxyy = m*(x*x*y*y - 1.0/7.0*d2*(x*x + y*y - 0.2*d2));
    mr->xxyz = m*(x*x*y*z - 1.0/7.0*d2*y*z);
    mr->xyyz = m*(x*y*y*z - 1.0/7.0*d2*x*z);
    /*
     ** Calculate reduced octopole moment...
     */
    mr->xxx = m*(x*x*x - 0.6*d2*x);
    mr->xyy = m*(x*y*y - 0.2*d2*x);
    mr->xxy = m*(x*x*y - 0.2*d2*y);
    mr->yyy = m*(y*y*y - 0.6*d2*y);
    mr->xxz = m*(x*x*z - 0.2*d2*z);
    mr->yyz = m*(y*y*z - 0.2*d2*z);
    mr->xyz = m*x*y*z;
    /*
     ** Calculate quadrupole moment...
     */
    mr->xx = m*(x*x - 1.0/3.0*d2);
    mr->yy = m*(y*y - 1.0/3.0*d2);
    mr->xy = m*x*y;
    mr->xz = m*x*z;
    mr->yz = m*y*z;
    mr->m = m;
    }


void momMomr2Momc(MOMR *ma,MOMC *mc) {
    mc->m = ma->m;
    mc->xx = ma->xx;
    mc->yy = ma->yy;
    mc->xy = ma->xy;
    mc->xz = ma->xz;
    mc->yz = ma->yz;
    mc->xxx = ma->xxx;
    mc->xyy = ma->xyy;
    mc->xxy = ma->xxy;
    mc->yyy = ma->yyy;
    mc->xxz = ma->xxz;
    mc->yyz = ma->yyz;
    mc->xyz = ma->xyz;
    mc->xxxx = ma->xxxx;
    mc->xyyy = ma->xyyy;
    mc->xxxy = ma->xxxy;
    mc->yyyy = ma->yyyy;
    mc->xxxz = ma->xxxz;
    mc->yyyz = ma->yyyz;
    mc->xxyy = ma->xxyy;
    mc->xxyz = ma->xxyz;
    mc->xyyz = ma->xyyz;
    mc->zz = -(ma->xx + ma->yy);
    mc->xzz = -(ma->xxx + ma->xyy);
    mc->yzz = -(ma->xxy + ma->yyy);
    mc->zzz = -(ma->xxz + ma->yyz);
    mc->xxzz = -(ma->xxxx + ma->xxyy);
    mc->xyzz = -(ma->xxxy + ma->xyyy);
    mc->xzzz = -(ma->xxxz + ma->xyyz);
    mc->yyzz = -(ma->xxyy + ma->yyyy);
    mc->yzzz = -(ma->xxyz + ma->yyyz);
    mc->zzzz = -(mc->xxzz + mc->yyzz);
    }

/*
 ** This function converts a complete multipole (MOMC) to a reduced one (MOMR).
 */
void momReduceMomc(MOMC *mc,MOMR *mr) {
    momFloat  t,tx,ty,tz,txx,txy,txz,tyy,tyz,tzz;

    /*
     ** First reduce Hexadecapole.
     */
    txx = (mc->xxxx + mc->xxyy + mc->xxzz)/7;
    txy = (mc->xxxy + mc->xyyy + mc->xyzz)/7;
    txz = (mc->xxxz + mc->xyyz + mc->xzzz)/7;
    tyy = (mc->xxyy + mc->yyyy + mc->yyzz)/7;
    tyz = (mc->xxyz + mc->yyyz + mc->yzzz)/7;
    tzz = (mc->xxzz + mc->yyzz + mc->zzzz)/7;
    t = 0.1*(txx + tyy + tzz);
    mr->xxxx = mc->xxxx - 6*(txx - t);
    mr->xyyy = mc->xyyy - 3*txy;
    mr->xxxy = mc->xxxy - 3*txy;
    mr->yyyy = mc->yyyy - 6*(tyy - t);
    mr->xxxz = mc->xxxz - 3*txz;
    mr->yyyz = mc->yyyz - 3*tyz;
    mr->xxyy = mc->xxyy - (txx + tyy - 2*t);
    mr->xxyz = mc->xxyz - tyz;
    mr->xyyz = mc->xyyz - txz;
    /*
     ** Now reduce the Octopole.
     */
    tx = (mc->xxx + mc->xyy + mc->xzz)/5;
    ty = (mc->xxy + mc->yyy + mc->yzz)/5;
    tz = (mc->xxz + mc->yyz + mc->zzz)/5;
    mr->xxx = mc->xxx - 3*tx;
    mr->xyy = mc->xyy - tx;
    mr->xxy = mc->xxy - ty;
    mr->yyy = mc->yyy - 3*ty;
    mr->xxz = mc->xxz - tz;
    mr->yyz = mc->yyz - tz;
    mr->xyz = mc->xyz;
    /*
     ** Now reduce the Quadrupole.
     */
    t = (mc->xx + mc->yy + mc->zz)/3;
    mr->xx = mc->xx - t;
    mr->yy = mc->yy - t;
    mr->xy = mc->xy;
    mr->xz = mc->xz;
    mr->yz = mc->yz;
    /*
     ** Finally the mass remains the same.
     */
    mr->m = mc->m;
    }


/*
 ** This function shifts a complete multipole (MOMC) to a new center of mass.
 ** <x,y,z> := d := rcm(old) - rcm(new).
 **
 ** OpCount ShiftMomc   (*,+) = (111,84)
 **         MakeMomc    (*,+) = (34,0)
 **         MulAddMomc  (*,+) = (32,32)
 **         Total       (*,+) = (177,116) = 293
 */
void momShiftMomc(MOMC *m,momFloat x,momFloat y,momFloat z) {
    MOMC f;

    momMakeMomc(&f,1,x,y,z);
    /*
     ** Shift the Hexadecapole.
     */
    m->xxxx += 4*m->xxx*x + 6*m->xx*f.xx;
    m->yyyy += 4*m->yyy*y + 6*m->yy*f.yy;
    m->zzzz += 4*m->zzz*z + 6*m->zz*f.zz;
    m->xyyy += m->yyy*x + 3*(m->xyy*y + m->yy*f.xy + m->xy*f.yy);
    m->xxxy += m->xxx*y + 3*(m->xxy*x + m->xx*f.xy + m->xy*f.xx);
    m->xxxz += m->xxx*z + 3*(m->xxz*x + m->xx*f.xz + m->xz*f.xx);
    m->yyyz += m->yyy*z + 3*(m->yyz*y + m->yy*f.yz + m->yz*f.yy);
    m->xzzz += m->zzz*x + 3*(m->xzz*z + m->zz*f.xz + m->xz*f.zz);
    m->yzzz += m->zzz*y + 3*(m->yzz*z + m->zz*f.yz + m->yz*f.zz);
    m->xxyy += 2*(m->xxy*y + m->xyy*x) + m->xx*f.yy + m->yy*f.xx + 4*m->xy*f.xy;
    m->xxzz += 2*(m->xxz*z + m->xzz*x) + m->xx*f.zz + m->zz*f.xx + 4*m->xz*f.xz;
    m->yyzz += 2*(m->yyz*z + m->yzz*y) + m->yy*f.zz + m->zz*f.yy + 4*m->yz*f.yz;
    m->xxyz += m->xxy*z + m->xxz*y + m->xx*f.yz + m->yz*f.xx + 2*(m->xyz*x + m->xy*f.xz + m->xz*f.xy);
    m->xyyz += m->xyy*z + m->yyz*x + m->yy*f.xz + m->xz*f.yy + 2*(m->xyz*y + m->xy*f.yz + m->yz*f.xy);
    m->xyzz += m->yzz*x + m->xzz*y + m->zz*f.xy + m->xy*f.zz + 2*(m->xyz*z + m->yz*f.xz + m->xz*f.yz);
    /*
     ** Now shift the Octopole.
     */
    m->xxx += 3*m->xx*x;
    m->yyy += 3*m->yy*y;
    m->zzz += 3*m->zz*z;
    m->xyy += 2*m->xy*y + m->yy*x;
    m->xzz += 2*m->xz*z + m->zz*x;
    m->yzz += 2*m->yz*z + m->zz*y;
    m->xxy += 2*m->xy*x + m->xx*y;
    m->xxz += 2*m->xz*x + m->xx*z;
    m->yyz += 2*m->yz*y + m->yy*z;
    m->xyz += m->xy*z + m->xz*y + m->yz*x;
    /*
     ** Now deal with the monopole terms.
     */
    f.m = 0;
    momMulAddMomc(m,m->m,&f);
    }


/*
 ** This function shifts a reduced multipole (MOMR) to a new center of mass.
 ** <x,y,z> := d := rcm(old) - rcm(new).
 **
 ** OpCount ShiftMomr  (*,+) = (128,111)
 **         MakeMomr   (*,+) = (43,18)
 **         MulAddMomr (*,+) = (22,22)
 **         Total      (*,+) = (193,151) = 344
 */
void momShiftMomr(MOMR *m,momFloat x,momFloat y,momFloat z) {
    MOMR f;
    momFloat t,tx,ty,tz,txx,tyy,txy,tyz,txz;
    const momFloat twosevenths = 2.0/7.0;

    momMakeMomr(&f,1,x,y,z);
    /*
     ** Calculate the correction terms.
     */
    tx = 0.4*(m->xx*x + m->xy*y + m->xz*z);
    ty = 0.4*(m->xy*x + m->yy*y + m->yz*z);
    tz = 0.4*(m->xz*x + m->yz*y - (m->xx + m->yy)*z);
    t = tx*x + ty*y + tz*z;
    txx = twosevenths*(m->xxx*x + m->xxy*y + m->xxz*z + 2*(m->xx*f.xx + m->xy*f.xy + m->xz*f.xz) - 0.5*t);
    tyy = twosevenths*(m->xyy*x + m->yyy*y + m->yyz*z + 2*(m->xy*f.xy + m->yy*f.yy + m->yz*f.yz) - 0.5*t);
    txy = twosevenths*(m->xxy*x + m->xyy*y + m->xyz*z + m->xy*(f.xx + f.yy) + (m->xx + m->yy)*f.xy + m->yz*f.xz + m->xz*f.yz);
    tyz = twosevenths*(m->xyz*x + m->yyz*y - (m->xxy + m->yyy)*z - m->yz*f.xx - m->xx*f.yz + m->xz*f.xy + m->xy*f.xz);
    txz = twosevenths*(m->xxz*x + m->xyz*y - (m->xxx + m->xyy)*z - m->xz*f.yy - m->yy*f.xz + m->yz*f.xy + m->xy*f.yz);
    /*
     ** Shift the Hexadecapole.
     */
    m->xxxx += 4*m->xxx*x + 6*(m->xx*f.xx - txx);
    m->yyyy += 4*m->yyy*y + 6*(m->yy*f.yy - tyy);
    m->xyyy += m->yyy*x + 3*(m->xyy*y + m->yy*f.xy + m->xy*f.yy - txy);
    m->xxxy += m->xxx*y + 3*(m->xxy*x + m->xx*f.xy + m->xy*f.xx - txy);
    m->xxxz += m->xxx*z + 3*(m->xxz*x + m->xx*f.xz + m->xz*f.xx - txz);
    m->yyyz += m->yyy*z + 3*(m->yyz*y + m->yy*f.yz + m->yz*f.yy - tyz);
    m->xxyy += 2*(m->xxy*y + m->xyy*x) + m->xx*f.yy + m->yy*f.xx + 4*m->xy*f.xy - txx - tyy;
    m->xxyz += m->xxy*z + m->xxz*y + m->xx*f.yz + m->yz*f.xx + 2*(m->xyz*x + m->xy*f.xz + m->xz*f.xy) - tyz;
    m->xyyz += m->xyy*z + m->yyz*x + m->yy*f.xz + m->xz*f.yy + 2*(m->xyz*y + m->xy*f.yz + m->yz*f.xy) - txz;
    /*
     ** Now shift the Octopole.
     */
    m->xxx += 3*(m->xx*x - tx);
    m->xyy += 2*m->xy*y + m->yy*x - tx;
    m->yyy += 3*(m->yy*y - ty);
    m->xxy += 2*m->xy*x + m->xx*y - ty;
    m->xxz += 2*m->xz*x + m->xx*z - tz;
    m->yyz += 2*m->yz*y + m->yy*z - tz;
    m->xyz += m->xy*z + m->xz*y + m->yz*x;
    /*
     ** Now deal with the monopole terms.
     */
    f.m = 0;
    momMulAddMomr(m,m->m,&f);
    }


/*
 ** This function shifts a reduced scaled multipole (FMOMR) to a new center of mass.
 ** <x,y,z> := d := rcm(old) - rcm(new).
 **
 ** OpCount ShiftMomr  (*,+) = (128,111)
 **         MakeMomr   (*,+) = (43,18)
 **         MulAddMomr (*,+) = (22,22)
 **         Total      (*,+) = (193,151) = 344
 */
void momShiftFmomr(FMOMR *m,float u,float x,float y,float z) {
    FMOMR f;
    float t,tx,ty,tz,txx,tyy,txy,tyz,txz,iu;
    const float twosevenths = 2.0f/7.0f;

    momMakeFmomr(&f,1.0f,u,x,y,z);
    iu = 1.0f/u;
    x *= iu;
    y *= iu;
    z *= iu;
    /*
     ** Calculate the correction terms.
     */
    tx = 0.4f*(m->xx*x + m->xy*y + m->xz*z);
    ty = 0.4f*(m->xy*x + m->yy*y + m->yz*z);
    tz = 0.4f*(m->xz*x + m->yz*y - (m->xx + m->yy)*z);
    t = tx*x + ty*y + tz*z;
    txx = twosevenths*(m->xxx*x + m->xxy*y + m->xxz*z + 2.0f*(m->xx*f.xx + m->xy*f.xy + m->xz*f.xz) - 0.5f*t);
    tyy = twosevenths*(m->xyy*x + m->yyy*y + m->yyz*z + 2.0f*(m->xy*f.xy + m->yy*f.yy + m->yz*f.yz) - 0.5f*t);
    txy = twosevenths*(m->xxy*x + m->xyy*y + m->xyz*z + m->xy*(f.xx + f.yy) + (m->xx + m->yy)*f.xy + m->yz*f.xz + m->xz*f.yz);
    tyz = twosevenths*(m->xyz*x + m->yyz*y - (m->xxy + m->yyy)*z - m->yz*f.xx - m->xx*f.yz + m->xz*f.xy + m->xy*f.xz);
    txz = twosevenths*(m->xxz*x + m->xyz*y - (m->xxx + m->xyy)*z - m->xz*f.yy - m->yy*f.xz + m->yz*f.xy + m->xy*f.yz);
    /*
     ** Shift the Hexadecapole.
     */
    m->xxxx += 4.0f*m->xxx*x + 6.0f*(m->xx*f.xx - txx);
    m->yyyy += 4.0f*m->yyy*y + 6.0f*(m->yy*f.yy - tyy);
    m->xyyy += m->yyy*x + 3.0f*(m->xyy*y + m->yy*f.xy + m->xy*f.yy - txy);
    m->xxxy += m->xxx*y + 3.0f*(m->xxy*x + m->xx*f.xy + m->xy*f.xx - txy);
    m->xxxz += m->xxx*z + 3.0f*(m->xxz*x + m->xx*f.xz + m->xz*f.xx - txz);
    m->yyyz += m->yyy*z + 3.0f*(m->yyz*y + m->yy*f.yz + m->yz*f.yy - tyz);
    m->xxyy += 2.0f*(m->xxy*y + m->xyy*x) + m->xx*f.yy + m->yy*f.xx + 4.0f*m->xy*f.xy - txx - tyy;
    m->xxyz += m->xxy*z + m->xxz*y + m->xx*f.yz + m->yz*f.xx + 2.0f*(m->xyz*x + m->xy*f.xz + m->xz*f.xy) - tyz;
    m->xyyz += m->xyy*z + m->yyz*x + m->yy*f.xz + m->xz*f.yy + 2.0f*(m->xyz*y + m->xy*f.yz + m->yz*f.xy) - txz;
    /*
     ** Now shift the Octopole.
     */
    m->xxx += 3.0f*(m->xx*x - tx);
    m->xyy += 2.0f*m->xy*y + m->yy*x - tx;
    m->yyy += 3.0f*(m->yy*y - ty);
    m->xxy += 2.0f*m->xy*x + m->xx*y - ty;
    m->xxz += 2.0f*m->xz*x + m->xx*z - tz;
    m->yyz += 2.0f*m->yz*y + m->yy*z - tz;
    m->xyz += m->xy*z + m->xz*y + m->yz*x;
    /*
     ** Now deal with the monopole terms.
     */
    f.m = 0;
    momMulAddFmomr(m,1.0,m->m,&f,1.0);
    }


/*
 ** This function shifts a reduced local expansion (LOCR) to a new center of expansion.
 ** <x,y,z> := d := rexp(new) - rexp(old).
 **
 ** Op Count (*,+) = (159,173) = 332
 */
double momShiftLocr(LOCR *l,momFloat x,momFloat y,momFloat z) {
    const momFloat onethird = 1.0/3.0;
    momFloat hx,hy,hz,tx,ty,tz;
    momFloat L,Lx,Ly,Lz,Lxx,Lxy,Lxz,Lyy,Lyz,Lxxx,Lxxy,Lxxz,Lxyy,Lxyz,Lyyy,Lyyz;
    momFloat Lxxxx,Lxxxy,Lxxyy,Lxyyy,Lyyyy,Lxxxz,Lxxyz,Lxyyz,Lyyyz;

    hx = 0.5*x;
    hy = 0.5*y;
    hz = 0.5*z;
    tx = onethird*x;
    ty = onethird*y;
    tz = onethird*z;

    L = l->x*x + l->y*y + l->z*z;
    l->m += L;

    Lx = l->xx*x + l->xy*y + l->xz*z;
    Ly = l->xy*x + l->yy*y + l->yz*z;
    Lz = l->xz*x + l->yz*y - (l->xx + l->yy)*z;
    L = Lx*hx + Ly*hy + Lz*hz;
    l->x += Lx;
    l->y += Ly;
    l->z += Lz;
    l->m += L;

    Lxx = l->xxx*x + l->xxy*y + l->xxz*z;
    Lxy = l->xxy*x + l->xyy*y + l->xyz*z;
    Lxz = l->xxz*x + l->xyz*y - (l->xxx + l->xyy)*z;
    Lyy = l->xyy*x + l->yyy*y + l->yyz*z;
    Lyz = l->xyz*x + l->yyz*y - (l->xxy + l->yyy)*z;
    Lx = Lxx*hx + Lxy*hy + Lxz*hz;
    Ly = Lxy*hx + Lyy*hy + Lyz*hz;
    Lz = Lxz*hx + Lyz*hy - (Lxx + Lyy)*hz;
    L = Lx*tx + Ly*ty + Lz*tz;
    l->xx += Lxx;
    l->xy += Lxy;
    l->xz += Lxz;
    l->yy += Lyy;
    l->yz += Lyz;
    l->x += Lx;
    l->y += Ly;
    l->z += Lz;
    l->m += L;

    Lxxx = l->xxxx*x + l->xxxy*y + l->xxxz*z;
    Lxxy = l->xxxy*x + l->xxyy*y + l->xxyz*z;
    Lxxz = l->xxxz*x + l->xxyz*y - (l->xxxx + l->xxyy)*z;
    Lxyy = l->xxyy*x + l->xyyy*y + l->xyyz*z;
    Lxyz = l->xxyz*x + l->xyyz*y - (l->xxxy + l->xyyy)*z;
    Lyyy = l->xyyy*x + l->yyyy*y + l->yyyz*z;
    Lyyz = l->xyyz*x + l->yyyz*y - (l->xxyy + l->yyyy)*z;
    Lxx = Lxxx*hx + Lxxy*hy + Lxxz*hz;
    Lxy = Lxxy*hx + Lxyy*hy + Lxyz*hz;
    Lxz = Lxxz*hx + Lxyz*hy - (Lxxx + Lxyy)*hz;
    Lyy = Lxyy*hx + Lyyy*hy + Lyyz*hz;
    Lyz = Lxyz*hx + Lyyz*hy - (Lxxy + Lyyy)*hz;
    Lx = Lxx*tx + Lxy*ty + Lxz*tz;
    Ly = Lxy*tx + Lyy*ty + Lyz*tz;
    Lz = Lxz*tx + Lyz*ty - (Lxx + Lyy)*tz;
    L = Lx*hx + Ly*hy + Lz*hz;

    l->xxx += Lxxx;
    l->xxy += Lxxy;
    l->xxz += Lxxz;
    l->xyy += Lxyy;
    l->xyz += Lxyz;
    l->yyy += Lyyy;
    l->yyz += Lyyz;
    l->xx += Lxx;
    l->xy += Lxy;
    l->xz += Lxz;
    l->yy += Lyy;
    l->yz += Lyz;
    l->x += Lx;
    l->y += Ly;
    l->z += Lz;
    l->m += 0.5*L;

    Lxxxx = l->xxxxx*x + l->xxxxy*y + l->xxxxz*z;
    Lxxxy = l->xxxxy*x + l->xxxyy*y + l->xxxyz*z;
    Lxxyy = l->xxxyy*x + l->xxyyy*y + l->xxyyz*z;
    Lxyyy = l->xxyyy*x + l->xyyyy*y + l->xyyyz*z;
    Lyyyy = l->xyyyy*x + l->yyyyy*y + l->yyyyz*z;
    Lxxxz = l->xxxxz*x + l->xxxyz*y - (l->xxxxx + l->xxxyy)*z;
    Lxxyz = l->xxxyz*x + l->xxyyz*y - (l->xxxxy + l->xxyyy)*z;
    Lxyyz = l->xxyyz*x + l->xyyyz*y - (l->xxxyy + l->xyyyy)*z;
    Lyyyz = l->xyyyz*x + l->yyyyz*y - (l->xxyyy + l->yyyyy)*z;
    Lxxx = Lxxxx*hx + Lxxxy*hy + Lxxxz*hz;
    Lxxy = Lxxxy*hx + Lxxyy*hy + Lxxyz*hz;
    Lxyy = Lxxyy*hx + Lxyyy*hy + Lxyyz*hz;
    Lyyy = Lxyyy*hx + Lyyyy*hy + Lyyyz*hz;
    Lxxz = Lxxxz*hx + Lxxyz*hy - (Lxxxx + Lxxyy)*hz;
    Lxyz = Lxxyz*hx + Lxyyz*hy - (Lxxxy + Lxyyy)*hz;
    Lyyz = Lxyyz*hx + Lyyyz*hy - (Lxxyy + Lyyyy)*hz;
    Lxx = Lxxx*tx + Lxxy*ty + Lxxz*tz;
    Lxy = Lxxy*tx + Lxyy*ty + Lxyz*tz;
    Lyy = Lxyy*tx + Lyyy*ty + Lyyz*tz;
    Lxz = Lxxz*tx + Lxyz*ty - (Lxxx + Lxyy)*tz;
    Lyz = Lxyz*tx + Lyyz*ty - (Lxxy + Lyyy)*tz;
    Lx = Lxx*hx + Lxy*hy + Lxz*hz;
    Ly = Lxy*hx + Lyy*hy + Lyz*hz;
    Lz = Lxz*hx + Lyz*hy - (Lxx + Lyy)*hz;
    L = Lx*hx + Ly*hy + Lz*hz;
    l->xxxx += Lxxxx;
    l->xxxy += Lxxxy;
    l->xxyy += Lxxyy;
    l->xyyy += Lxyyy;
    l->yyyy += Lyyyy;
    l->xxxz += Lxxxz;
    l->xxyz += Lxxyz;
    l->xyyz += Lxyyz;
    l->yyyz += Lyyyz;
    l->xxx += Lxxx;
    l->xxy += Lxxy;
    l->xxz += Lxxz;
    l->xyy += Lxyy;
    l->xyz += Lxyz;
    l->yyy += Lyyy;
    l->yyz += Lyyz;
    l->xx += Lxx;
    l->xy += Lxy;
    l->xz += Lxz;
    l->yy += Lyy;
    l->yz += Lyz;
    l->x += 0.5*Lx;
    l->y += 0.5*Ly;
    l->z += 0.5*Lz;
    l->m += 0.2*L;

    return 332.0;
    }


/*
 ** This function shifts a reduced scaled local expansion (FLOCR) to a new center of expansion.
 ** <x,y,z> := d := rexp(new) - rexp(old).
 **
 ** Op Count (/,*,+) = (1,162,173) = 336
 */
double momShiftFlocr(FLOCR *l,float v,float x,float y,float z) {
    const float onethird = 1.0f/3.0f;
    float iv,hx,hy,hz,tx,ty,tz;
    float L,Lx,Ly,Lz,Lxx,Lxy,Lxz,Lyy,Lyz,Lxxx,Lxxy,Lxxz,Lxyy,Lxyz,Lyyy,Lyyz;
    float Lxxxx,Lxxxy,Lxxyy,Lxyyy,Lyyyy,Lxxxz,Lxxyz,Lxyyz,Lyyyz;

    iv = 1.0f/v;
    x *= iv;
    y *= iv;
    z *= iv;

    hx = 0.5*x;
    hy = 0.5*y;
    hz = 0.5*z;
    tx = onethird*x;
    ty = onethird*y;
    tz = onethird*z;

    L = l->x*x + l->y*y + l->z*z;
    l->m += L;

    Lx = l->xx*x + l->xy*y + l->xz*z;
    Ly = l->xy*x + l->yy*y + l->yz*z;
    Lz = l->xz*x + l->yz*y - (l->xx + l->yy)*z;
    L = Lx*hx + Ly*hy + Lz*hz;
    l->x += Lx;
    l->y += Ly;
    l->z += Lz;
    l->m += L;

    Lxx = l->xxx*x + l->xxy*y + l->xxz*z;
    Lxy = l->xxy*x + l->xyy*y + l->xyz*z;
    Lxz = l->xxz*x + l->xyz*y - (l->xxx + l->xyy)*z;
    Lyy = l->xyy*x + l->yyy*y + l->yyz*z;
    Lyz = l->xyz*x + l->yyz*y - (l->xxy + l->yyy)*z;
    Lx = Lxx*hx + Lxy*hy + Lxz*hz;
    Ly = Lxy*hx + Lyy*hy + Lyz*hz;
    Lz = Lxz*hx + Lyz*hy - (Lxx + Lyy)*hz;
    L = Lx*tx + Ly*ty + Lz*tz;
    l->xx += Lxx;
    l->xy += Lxy;
    l->xz += Lxz;
    l->yy += Lyy;
    l->yz += Lyz;
    l->x += Lx;
    l->y += Ly;
    l->z += Lz;
    l->m += L;

    Lxxx = l->xxxx*x + l->xxxy*y + l->xxxz*z;
    Lxxy = l->xxxy*x + l->xxyy*y + l->xxyz*z;
    Lxxz = l->xxxz*x + l->xxyz*y - (l->xxxx + l->xxyy)*z;
    Lxyy = l->xxyy*x + l->xyyy*y + l->xyyz*z;
    Lxyz = l->xxyz*x + l->xyyz*y - (l->xxxy + l->xyyy)*z;
    Lyyy = l->xyyy*x + l->yyyy*y + l->yyyz*z;
    Lyyz = l->xyyz*x + l->yyyz*y - (l->xxyy + l->yyyy)*z;
    Lxx = Lxxx*hx + Lxxy*hy + Lxxz*hz;
    Lxy = Lxxy*hx + Lxyy*hy + Lxyz*hz;
    Lxz = Lxxz*hx + Lxyz*hy - (Lxxx + Lxyy)*hz;
    Lyy = Lxyy*hx + Lyyy*hy + Lyyz*hz;
    Lyz = Lxyz*hx + Lyyz*hy - (Lxxy + Lyyy)*hz;
    Lx = Lxx*tx + Lxy*ty + Lxz*tz;
    Ly = Lxy*tx + Lyy*ty + Lyz*tz;
    Lz = Lxz*tx + Lyz*ty - (Lxx + Lyy)*tz;
    L = Lx*hx + Ly*hy + Lz*hz;
    l->xxx += Lxxx;
    l->xxy += Lxxy;
    l->xxz += Lxxz;
    l->xyy += Lxyy;
    l->xyz += Lxyz;
    l->yyy += Lyyy;
    l->yyz += Lyyz;
    l->xx += Lxx;
    l->xy += Lxy;
    l->xz += Lxz;
    l->yy += Lyy;
    l->yz += Lyz;
    l->x += Lx;
    l->y += Ly;
    l->z += Lz;
    l->m += 0.5*L;

    Lxxxx = l->xxxxx*x + l->xxxxy*y + l->xxxxz*z;
    Lxxxy = l->xxxxy*x + l->xxxyy*y + l->xxxyz*z;
    Lxxyy = l->xxxyy*x + l->xxyyy*y + l->xxyyz*z;
    Lxyyy = l->xxyyy*x + l->xyyyy*y + l->xyyyz*z;
    Lyyyy = l->xyyyy*x + l->yyyyy*y + l->yyyyz*z;
    Lxxxz = l->xxxxz*x + l->xxxyz*y - (l->xxxxx + l->xxxyy)*z;
    Lxxyz = l->xxxyz*x + l->xxyyz*y - (l->xxxxy + l->xxyyy)*z;
    Lxyyz = l->xxyyz*x + l->xyyyz*y - (l->xxxyy + l->xyyyy)*z;
    Lyyyz = l->xyyyz*x + l->yyyyz*y - (l->xxyyy + l->yyyyy)*z;
    Lxxx = Lxxxx*hx + Lxxxy*hy + Lxxxz*hz;
    Lxxy = Lxxxy*hx + Lxxyy*hy + Lxxyz*hz;
    Lxyy = Lxxyy*hx + Lxyyy*hy + Lxyyz*hz;
    Lyyy = Lxyyy*hx + Lyyyy*hy + Lyyyz*hz;
    Lxxz = Lxxxz*hx + Lxxyz*hy - (Lxxxx + Lxxyy)*hz;
    Lxyz = Lxxyz*hx + Lxyyz*hy - (Lxxxy + Lxyyy)*hz;
    Lyyz = Lxyyz*hx + Lyyyz*hy - (Lxxyy + Lyyyy)*hz;
    Lxx = Lxxx*tx + Lxxy*ty + Lxxz*tz;
    Lxy = Lxxy*tx + Lxyy*ty + Lxyz*tz;
    Lyy = Lxyy*tx + Lyyy*ty + Lyyz*tz;
    Lxz = Lxxz*tx + Lxyz*ty - (Lxxx + Lxyy)*tz;
    Lyz = Lxyz*tx + Lyyz*ty - (Lxxy + Lyyy)*tz;
    Lx = Lxx*hx + Lxy*hy + Lxz*hz;
    Ly = Lxy*hx + Lyy*hy + Lyz*hz;
    Lz = Lxz*hx + Lyz*hy - (Lxx + Lyy)*hz;
    L = Lx*hx + Ly*hy + Lz*hz;
    l->xxxx += Lxxxx;
    l->xxxy += Lxxxy;
    l->xxyy += Lxxyy;
    l->xyyy += Lxyyy;
    l->yyyy += Lyyyy;
    l->xxxz += Lxxxz;
    l->xxyz += Lxxyz;
    l->xyyz += Lxyyz;
    l->yyyz += Lyyyz;
    l->xxx += Lxxx;
    l->xxy += Lxxy;
    l->xxz += Lxxz;
    l->xyy += Lxyy;
    l->xyz += Lxyz;
    l->yyy += Lyyy;
    l->yyz += Lyyz;
    l->xx += Lxx;
    l->xy += Lxy;
    l->xz += Lxz;
    l->yy += Lyy;
    l->yz += Lyz;
    l->x += 0.5*Lx;
    l->y += 0.5*Ly;
    l->z += 0.5*Lz;
    l->m += 0.2*L;

    return 332.0;
    }


/*
 ** This is a new fast version of QEVAL which evaluates
 ** the interaction due to the reduced moment 'm'.
 ** This version is nearly two times as fast as a naive
 ** implementation.
 **
 ** March 23, 2007: This function now uses unit vectors
 ** which reduces the required precision in the exponent
 ** since the highest power of r is now 5 (g4 ~ r^(-5)).
 **
 ** OpCount = (*,+) = (105,72) = 177 - 8 = 169
 **
 ** CAREFUL: this function no longer accumulates on fPot,ax,ay,az!
 */
void momEvalMomr(MOMR *m,momFloat dir,momFloat x,momFloat y,momFloat z,
		 momFloat *fPot,momFloat *ax,momFloat *ay,momFloat *az,momFloat *magai) {
    const momFloat onethird = 1.0/3.0;
    momFloat xx,xy,xz,yy,yz,zz;
    momFloat xxx,xxy,xxz,xyy,yyy,yyz,xyz;
    momFloat tx,ty,tz,g0,g2,g3,g4;

    g0 = dir;
    g2 = 3*dir*dir*dir;
    g3 = 5*g2*dir;
    g4 = 7*g3*dir;
    /*
     ** Calculate the trace-free distance terms.
     */
    x *= dir;
    y *= dir;
    z *= dir;
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
    tx = g4*(m->xxxx*xxx + m->xyyy*yyy + m->xxxy*xxy + m->xxxz*xxz + m->xxyy*xyy + m->xxyz*xyz + m->xyyz*yyz);
    ty = g4*(m->xyyy*xyy + m->xxxy*xxx + m->yyyy*yyy + m->yyyz*yyz + m->xxyy*xxy + m->xxyz*xxz + m->xyyz*xyz);
    tz = g4*(-m->xxxx*xxz - (m->xyyy + m->xxxy)*xyz - m->yyyy*yyz + m->xxxz*xxx + m->yyyz*yyy - m->xxyy*(xxz + yyz) + m->xxyz*xxy + m->xyyz*xyy);
    g4 = 0.25*(tx*x + ty*y + tz*z);
    xxx = g3*(m->xxx*xx + m->xyy*yy + m->xxy*xy + m->xxz*xz + m->xyz*yz);
    xxy = g3*(m->xyy*xy + m->xxy*xx + m->yyy*yy + m->yyz*yz + m->xyz*xz);
    xxz = g3*(-(m->xxx + m->xyy)*xz - (m->xxy + m->yyy)*yz + m->xxz*xx + m->yyz*yy + m->xyz*xy);
    g3 = onethird*(xxx*x + xxy*y + xxz*z);
    xx = g2*(m->xx*x + m->xy*y + m->xz*z);
    xy = g2*(m->yy*y + m->xy*x + m->yz*z);
    xz = g2*(-(m->xx + m->yy)*z + m->xz*x + m->yz*y);
    g2 = 0.5*(xx*x + xy*y + xz*z);
    g0 *= m->m;
    *fPot = -(g0 + g2 + g3 + g4);
    g0 += 5*g2 + 7*g3 + 9*g4;
    *ax = dir*(xx + xxx + tx - x*g0);
    *ay = dir*(xy + xxy + ty - y*g0);
    *az = dir*(xz + xxz + tz - z*g0);
    *magai = g0*dir;
    }


/*
** The generalized version of the above.
**
** CAREFUL: this function no longer accumulates on fPot,ax,ay,az!
*/
void momGenEvalMomr(MOMR *m,momFloat g0,momFloat g1,momFloat g2,momFloat g3,momFloat g4,momFloat g5,
		    momFloat x,momFloat y,momFloat z,
		    momFloat *fPot,momFloat *ax,momFloat *ay,momFloat *az,momFloat *magai) {
    const momFloat onethird = 1.0/3.0;
    momFloat xx,xy,xz,yy,yz,zz;
    momFloat xxx,xxy,xxz,xyy,yyy,yyz,xyz;
    momFloat A,Ax,Ay,Az,B,Bx,By,Bz,C,Cx,Cy,Cz,R;

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
    Cx = m->xxxx*xxx + m->xyyy*yyy + m->xxxy*xxy + m->xxxz*xxz + m->xxyy*xyy + m->xxyz*xyz + m->xyyz*yyz;
    Cy = m->xyyy*xyy + m->xxxy*xxx + m->yyyy*yyy + m->yyyz*yyz + m->xxyy*xxy + m->xxyz*xxz + m->xyyz*xyz;
    Cz = -m->xxxx*xxz - (m->xyyy + m->xxxy)*xyz - m->yyyy*yyz + m->xxxz*xxx + m->yyyz*yyy - m->xxyy*(xxz + yyz) + m->xxyz*xxy + m->xyyz*xyy;
    C = 0.25*(Cx*x + Cy*y + Cz*z);
    Bx = m->xxx*xx + m->xyy*yy + m->xxy*xy + m->xxz*xz + m->xyz*yz;
    By = m->xyy*xy + m->xxy*xx + m->yyy*yy + m->yyz*yz + m->xyz*xz;
    Bz = -(m->xxx + m->xyy)*xz - (m->xxy + m->yyy)*yz + m->xxz*xx + m->yyz*yy + m->xyz*xy;
    B = onethird*(Bx*x + By*y + Bz*z);
    Ax = m->xx*x + m->xy*y + m->xz*z;
    Ay = m->yy*y + m->xy*x + m->yz*z;
    Az = -(m->xx + m->yy)*z + m->xz*x + m->yz*y;
    A = 0.5*(Ax*x + Ay*y + Az*z);
    *fPot = g0*m->m + g2*A - g3*B + g4*C;
    R = g1*m->m + g3*A - g4*B + g5*C;
    *ax = -g2*Ax + g3*Bx - g4*Cx - x*R;
    *ay = -g2*Ay + g3*By - g4*Cy - y*R;
    *az = -g2*Az + g3*Bz - g4*Cz - z*R;
    *magai = sqrt((*ax)*(*ax) + (*ay)*(*ay) + (*az)*(*az));
    }


/*
 ** This is a new fast version of QEVAL which evaluates
 ** the interaction due to the reduced moment 'm'.
 ** This version is nearly two times as fast as a naive
 ** implementation.
 **
 ** March 23, 2007: This function now uses unit vectors
 ** which reduces the required precision in the exponent
 ** since the highest power of r is now 5 (g4 ~ r^(-5)).
 **
 ** OpCount = (*,+) = (106,72) = 178 - 8 = 170
 **
 ** CAREFUL: this function no longer accumulates on fPot,ax,ay,az!
 */
void momEvalFmomrcm(FMOMR *m,float u,float dir,float x,float y,float z,
		    float *fPot,float *ax,float *ay,float *az,float *magai) {
    const float onethird = 1.0f/3.0f;
    float xx,xy,xz,yy,yz,zz;
    float xxx,xxy,xxz,xyy,yyy,yyz,xyz;
    float tx,ty,tz,g0,g2,g3,g4;

    u *= dir;
    g0 = dir;
    g2 = 3*dir*u*u;
    g3 = 5*g2*u;
    g4 = 7*g3*u;
    /*
     ** Calculate the trace-free distance terms.
     */
    x *= dir;
    y *= dir;
    z *= dir;
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
    tx = g4*(m->xxxx*xxx + m->xyyy*yyy + m->xxxy*xxy + m->xxxz*xxz + m->xxyy*xyy + m->xxyz*xyz + m->xyyz*yyz);
    ty = g4*(m->xyyy*xyy + m->xxxy*xxx + m->yyyy*yyy + m->yyyz*yyz + m->xxyy*xxy + m->xxyz*xxz + m->xyyz*xyz);
    tz = g4*(-m->xxxx*xxz - (m->xyyy + m->xxxy)*xyz - m->yyyy*yyz + m->xxxz*xxx + m->yyyz*yyy - m->xxyy*(xxz + yyz) + m->xxyz*xxy + m->xyyz*xyy);
    g4 = 0.25*(tx*x + ty*y + tz*z);
    xxx = g3*(m->xxx*xx + m->xyy*yy + m->xxy*xy + m->xxz*xz + m->xyz*yz);
    xxy = g3*(m->xyy*xy + m->xxy*xx + m->yyy*yy + m->yyz*yz + m->xyz*xz);
    xxz = g3*(-(m->xxx + m->xyy)*xz - (m->xxy + m->yyy)*yz + m->xxz*xx + m->yyz*yy + m->xyz*xy);
    g3 = onethird*(xxx*x + xxy*y + xxz*z);
    xx = g2*(m->xx*x + m->xy*y + m->xz*z);
    xy = g2*(m->yy*y + m->xy*x + m->yz*z);
    xz = g2*(-(m->xx + m->yy)*z + m->xz*x + m->yz*y);
    g2 = 0.5*(xx*x + xy*y + xz*z);
    g0 *= m->m;
    *fPot = -(g0 + g2 + g3 + g4);
    g0 += 5*g2 + 7*g3 + 9*g4;
    *ax = dir*(xx + xxx + tx - x*g0);
    *ay = dir*(xy + xxy + ty - y*g0);
    *az = dir*(xz + xxz + tz - z*g0);
    *magai = g0*dir;
    }


/*
** Op Count = (*,+) = (302,207) = 509
*/
double momLocrAddMomr5cm(LOCR *l,MOMR *m,momFloat dir,momFloat x,momFloat y,momFloat z,double *tax,double *tay,double *taz) {
    const momFloat onethird = 1.0/3.0;
    momFloat xx,xy,xz,yy,yz,zz;
    momFloat xxx,xxy,xyy,yyy,xxz,xyz,yyz;
    momFloat Ax,Ay,Az,A,Bxx,Bxy,Byy,Bxz,Byz,Bx,By,Bz,B,Cx,Cy,Cz,C;
    momFloat R1,R2,R3,T2,T3;
    momFloat g0,g1,g2,g3,g4,g5;
    momFloat g4xx,g4yy,g5xx,g5yy,fxx,fyy;

    x *= dir;
    y *= dir;
    z *= dir;
    g0 = -dir;
    g1 = -g0*dir;
    g2 = -3*g1*dir;
    g3 = -5*g2*dir;
    g4 = -7*g3*dir;
    g5 = -9*g4*dir;
    /*
    ** Calculate the funky distance terms.
    */
    xx = 0.5*x*x;
    xy = x*y;
    yy = 0.5*y*y;
    xz = x*z;
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

    Bxx = x*m->xxx + y*m->xxy + z*m->xxz;
    Bxy = x*m->xxy + y*m->xyy + z*m->xyz;
    Byy = x*m->xyy + y*m->yyy + z*m->yyz;
    Bxz = x*m->xxz + y*m->xyz - z*(m->xxx + m->xyy);
    Byz = x*m->xyz + y*m->yyz - z*(m->xxy + m->yyy);

    Cx = m->xxxx*xxx + m->xyyy*yyy + m->xxxy*xxy + m->xxxz*xxz + m->xxyy*xyy + m->xxyz*xyz + m->xyyz*yyz;
    Cy = m->xyyy*xyy + m->xxxy*xxx + m->yyyy*yyy + m->yyyz*yyz + m->xxyy*xxy + m->xxyz*xxz + m->xyyz*xyz;
    Cz = -m->xxxx*xxz - (m->xyyy + m->xxxy)*xyz - m->yyyy*yyz + m->xxxz*xxx + m->yyyz*yyy - m->xxyy*(xxz + yyz) + m->xxyz*xxy + m->xyyz*xyy;

    Ax = x*m->xx + y*m->xy + z*m->xz;
    Ay = x*m->xy + y*m->yy + z*m->yz;
    Az = x*m->xz + y*m->yz - z*(m->xx + m->yy);

    Bx = 0.5*(x*Bxx + y*Bxy + z*Bxz);
    By = 0.5*(x*Bxy + y*Byy + z*Byz);
    Bz = 0.5*(x*Bxz + y*Byz - z*(Bxx + Byy));

    C = 0.25*(x*Cx + y*Cy + z*Cz);

    A = 0.5*(x*Ax + y*Ay + z*Az);

    B = onethird*(x*Bx + y*By + z*Bz);

    xx = x*x;
    yy = y*y;

    l->m += g0*m->m + g2*A - g3*B + g4*C;
    R1 = g1*m->m + g3*A - g4*B + g5*C;
    R2 = g2*m->m + g4*A - g5*B;
    R3 = g3*m->m + g5*A;

    g1 *= dir;
    g2 *= dir;
    g3 *= dir;

    T2 = g1*m->m + g3*A;
    T3 = g2*m->m;

    *tax = -(g2*Ax - g3*Bx + x*R1);
    *tay = -(g2*Ay - g3*By + y*R1);
    *taz = -(g2*Az - g3*Bz + z*R1);

    g2 *= dir;

    g4xx = g4*xx;
    g4yy = g4*yy;

    l->xxxx += (3*g2 + (6*g3 + g4xx)*xx)*m->m;
    l->yyyy += (3*g2 + (6*g3 + g4yy)*yy)*m->m;
    fxx = (3*g3 + g4xx)*m->m;
    l->xxxy += fxx*xy;
    l->xxxz += fxx*xz;
    fyy = (3*g3 + g4yy)*m->m;
    l->xyyy += fyy*xy;
    l->yyyz += fyy*yz;
    fxx = (g3 + g4xx);
    l->xxyz += fxx*yz*m->m;
    l->xxyy += (g2 + g3*xx + fxx*yy)*m->m;
    l->xyyz += (g3 + g4yy)*xz*m->m;

    g4 *= dir;

    T2 -= g4*B;
    T3 += g4*A;

    *tax -= g4*Cx;
    *tay -= g4*Cy;
    *taz -= g4*Cz;
    l->x -= *tax;
    l->y -= *tay;
    l->z -= *taz;

    l->xx += g2*m->xx + T2 + R2*xx + 2*g3*Ax*x - 2*g4*Bx*x;
    l->yy += g2*m->yy + T2 + R2*yy + 2*g3*Ay*y - 2*g4*By*y;
    l->xy += g2*m->xy + R2*xy + g3*(Ax*y + Ay*x) - g4*(Bx*y + By*x);
    l->xz += g2*m->xz + R2*xz + g3*(Ax*z + Az*x) - g4*(Bx*z + Bz*x);
    l->yz += g2*m->yz + R2*yz + g3*(Ay*z + Az*y) - g4*(By*z + Bz*y);

    g3 *= dir;

    g5xx = g5*xx;
    g5yy = g5*yy;

    l->xx -= g3*Bxx;
    l->xy -= g3*Bxy;
    l->yy -= g3*Byy;
    l->xz -= g3*Bxz;
    l->yz -= g3*Byz;

    fxx = T3 + R3*xx;
    fyy = T3 + R3*yy;

    l->xxy += fxx*y + g4*(2*xy*Ax + xx*Ay) + g3*(Ay + 2*m->xy*x + m->xx*y);
    l->xxz += fxx*z + g4*(2*xz*Ax + xx*Az) + g3*(Az + 2*m->xz*x + m->xx*z);
    l->xyy += fyy*x + g4*(2*xy*Ay + yy*Ax) + g3*(Ax + 2*m->xy*y + m->yy*x);
    l->yyz += fyy*z + g4*(2*yz*Ay + yy*Az) + g3*(Az + 2*m->yz*y + m->yy*z);
    l->xyz += R3*xy*z + g4*(yz*Ax + xz*Ay + xy*Az) + g3*(m->xy*z + m->xz*y + m->yz*x);

    fxx += 2*T3;
    fyy += 2*T3;

    l->xxx += fxx*x + 3*(g4*xx*Ax + g3*(Ax + m->xx*x));
    l->yyy += fyy*y + 3*(g4*yy*Ay + g3*(Ay + m->yy*y));

    fxx = 3*g3 + (6*g4 + g5xx)*xx;
    fyy = 3*g3 + (6*g4 + g5yy)*yy;

    x *= m->m;
    y *= m->m;
    z *= m->m;

    l->xxxxx += (15*g3 + (10*g4 + g5xx)*xx)*x;
    l->yyyyy += (15*g3 + (10*g4 + g5yy)*yy)*y;
    l->xxxyy += (3*g3 + 3*g4*yy + (g4 + g5yy)*xx)*x;
    l->xxyyy += (3*g3 + 3*g4*xx + (g4 + g5xx)*yy)*y;
    l->xxyyz += (g3 + g4*xx + (g4 + g5xx)*yy)*z;
    l->xxxxy += fxx*y;
    l->xxxxz += fxx*z;
    l->xyyyy += fyy*x;
    l->yyyyz += fyy*z;
    l->xxxyz += (3*g4 + g5xx)*xy*z;
    l->xyyyz += (3*g4 + g5yy)*xy*z;

    return 509.0;
    }



/*
** Op Count = (*,+) = (129,77) = 206 
*/
double momLocrAddMono5(LOCR *l,momFloat m,momFloat dir,momFloat x,momFloat y,momFloat z,double *tax,double *tay,double *taz) {
    const momFloat onethird = 1.0/3.0;
    momFloat xx,xy,xz,yy,yz,zz;
    momFloat xxx,xxy,xyy,yyy,xxz,xyz,yyz;
    momFloat R1,R2,R3,T2,T3;
    momFloat g0,g1,g2,g3,g4,g5;
    momFloat g4xx,g4yy,g5xx,g5yy,fxx,fyy;

    x *= dir;
    y *= dir;
    z *= dir;
    g0 = -dir;
    g1 = -g0*dir;
    g2 = -3*g1*dir;
    g3 = -5*g2*dir;
    g4 = -7*g3*dir;
    g5 = -9*g4*dir;
    /*
    ** Calculate the funky distance terms.
    */
    xx = 0.5*x*x;
    xy = x*y;
    yy = 0.5*y*y;
    xz = x*z;
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

    xx = x*x;
    yy = y*y;

    l->m += g0*m;
    R1 = g1*m;
    R2 = g2*m;
    R3 = g3*m;

    g1 *= dir;
    g2 *= dir;
    g3 *= dir;

    T2 = g1*m;
    T3 = g2*m;

    *tax = -(x*R1);
    *tay = -(y*R1);
    *taz = -(z*R1);

    g2 *= dir;

    g4xx = g4*xx;
    g4yy = g4*yy;

    l->xxxx += (3*g2 + (6*g3 + g4xx)*xx)*m;
    l->yyyy += (3*g2 + (6*g3 + g4yy)*yy)*m;
    fxx = (3*g3 + g4xx)*m;
    l->xxxy += fxx*xy;
    l->xxxz += fxx*xz;
    fyy = (3*g3 + g4yy)*m;
    l->xyyy += fyy*xy;
    l->yyyz += fyy*yz;
    fxx = (g3 + g4xx);
    l->xxyz += fxx*yz*m;
    l->xxyy += (g2 + g3*xx + fxx*yy)*m;
    l->xyyz += (g3 + g4yy)*xz*m;

    g4 *= dir;

    l->x -= *tax;
    l->y -= *tay;
    l->z -= *taz;

    l->xx += T2 + R2*xx;
    l->yy += T2 + R2*yy;
    l->xy += R2*xy;
    l->xz += R2*xz;
    l->yz += R2*yz;

    g3 *= dir;

    g5xx = g5*xx;
    g5yy = g5*yy;

    fxx = T3 + R3*xx;
    fyy = T3 + R3*yy;

    l->xxy += fxx*y;
    l->xxz += fxx*z;
    l->xyy += fyy*x;
    l->yyz += fyy*z;
    l->xyz += R3*xy*z;

    fxx += 2*T3;
    fyy += 2*T3;

    l->xxx += fxx*x;
    l->yyy += fyy*y;

    fxx = 3*g3 + (6*g4 + g5xx)*xx;
    fyy = 3*g3 + (6*g4 + g5yy)*yy;

    x *= m;
    y *= m;
    z *= m;

    l->xxxxx += (15*g3 + (10*g4 + g5xx)*xx)*x;
    l->yyyyy += (15*g3 + (10*g4 + g5yy)*yy)*y;
    l->xxxyy += (3*g3 + 3*g4*yy + (g4 + g5yy)*xx)*x;
    l->xxyyy += (3*g3 + 3*g4*xx + (g4 + g5xx)*yy)*y;
    l->xxyyz += (g3 + g4*xx + (g4 + g5xx)*yy)*z;
    l->xxxxy += fxx*y;
    l->xxxxz += fxx*z;
    l->xyyyy += fyy*x;
    l->yyyyz += fyy*z;
    l->xxxyz += (3*g4 + g5xx)*xy*z;
    l->xyyyz += (3*g4 + g5yy)*xy*z;

    return 206.0;
    }


/*
** Op Count = (*,+,-) = (265,150,49) = 464
*/
double momFlocrAddFmomr5cm(FLOCR *l,float v,FMOMR *m,float u,float dir,float x,float y,float z,float *tax,float *tay,float *taz) {
    const float onethird = 1.0f/3.0f;
    float u2,u3,u4;
    float xx,xy,xz,yy,yz,zz;
    float xxx,xxy,xyy,yyy,xxz,xyz,yyz;
    float R2xx,R2xy,R2xz,R2yy,R2yz,R2x,R2y,R2z,R2,R3xx,R3xy,R3yy,R3xz,R3yz,R3x,R3y,R3z,R3,R4x,R4y,R4z,R4;
    float T0,txx,tyy,t1,t1x,t1y,t1z,t1xx,t1yy,t2x,t2y,t2z,t2xx,t2yy,txxxx,tyyyy;

    u *= dir;
    x *= dir;
    y *= dir;
    z *= dir;
    dir = -dir;
    v *= dir;
    u2 = 15.0f*u*u;  /* becomes 15.0f*u2! */
    /*
    ** Calculate the funky distance terms.
    */
    xx = 0.5f*x*x;
    xy = x*y;
    yy = 0.5f*y*y;
    xz = x*z;
    yz = y*z;
    zz = 0.5f*z*z;
    xxx = x*(onethird*xx - zz);
    xxz = z*(xx - onethird*zz);
    yyy = y*(onethird*yy - zz);
    yyz = z*(yy - onethird*zz);
    xx -= zz;
    yy -= zz;
    xxy = y*xx;
    xyy = x*yy;
    xyz = xy*z;

    u3 = u2*u;  /* becomes 5.0f*u3! */

    R2xx = u2*m->xx;
    R2xy = u2*m->xy;
    R2xz = u2*m->xz;
    R2yy = u2*m->yy;
    R2yz = u2*m->yz;

    u4 = 7.0f*u3*u;  /* becomes 7.0f*5.0f*u4! */

    R2x = x*R2xx + y*R2xy + z*R2xz;
    R2y = x*R2xy + y*R2yy + z*R2yz;
    R2z = x*R2xz + y*R2yz - z*(R2xx + R2yy);

    R3xx = u3*(x*m->xxx + y*m->xxy + z*m->xxz);
    R3xy = u3*(x*m->xxy + y*m->xyy + z*m->xyz);
    R3yy = u3*(x*m->xyy + y*m->yyy + z*m->yyz);
    R3xz = u3*(x*m->xxz + y*m->xyz - z*(m->xxx + m->xyy));
    R3yz = u3*(x*m->xyz + y*m->yyz - z*(m->xxy + m->yyy));

    R4x = u4*(m->xxxx*xxx + m->xyyy*yyy + m->xxxy*xxy + m->xxxz*xxz + m->xxyy*xyy + m->xxyz*xyz + m->xyyz*yyz);
    R4y = u4*(m->xyyy*xyy + m->xxxy*xxx + m->yyyy*yyy + m->yyyz*yyz + m->xxyy*xxy + m->xxyz*xxz + m->xyyz*xyz);
    R4z = u4*(-m->xxxx*xxz - (m->xyyy + m->xxxy)*xyz - m->yyyy*yyz + m->xxxz*xxx + m->yyyz*yyy - m->xxyy*(xxz + yyz) + m->xxyz*xxy + m->xyyz*xyy);


    R3x = 0.5f*(x*R3xx + y*R3xy + z*R3xz);
    R3y = 0.5f*(x*R3xy + y*R3yy + z*R3yz);
    R3z = 0.5f*(x*R3xz + y*R3yz - z*(R3xx + R3yy));

    R4 = 0.25f*(x*R4x + y*R4y + z*R4z);

    R2 = 0.5f*(x*R2x + y*R2y + z*R2z);

    R3 = onethird*(x*R3x + y*R3y + z*R3z);

    xx = x*x;
    yy = y*y;

    /*
    ** Now we use the 'R's.
    */
    l->m += dir*(m->m + 0.2f*R2 + R3 + R4);

    dir *= v;
    T0 = -(m->m + R2 + 7.0f*R3 + 9.0f*R4);

    *tax = dir*(T0*x + 0.2f*R2x + R3x + R4x);
    *tay = dir*(T0*y + 0.2f*R2y + R3y + R4y);
    *taz = dir*(T0*z + 0.2f*R2z + R3z + R4z);
    l->x -= *tax;
    l->y -= *tay;
    l->z -= *taz;

    dir *= v;
    T0 = 3.0f*m->m + 7.0f*(R2 + 9.0f*R3);

    t1 = m->m + R2 + 7.0f*R3;
    t1x = R2x + 7.0f*R3x;
    t1y = R2y + 7.0f*R3y;
    t1z = R2z + 7.0f*R3z;
    l->xx += dir*(T0*xx - t1 - 2.0f*x*t1x + 0.2f*R2xx + R3xx);
    l->yy += dir*(T0*yy - t1 - 2.0f*y*t1y + 0.2f*R2yy + R3yy);
    l->xy += dir*(T0*xy - y*t1x - x*t1y + 0.2f*R2xy + R3xy);
    l->xz += dir*(T0*xz - z*t1x - x*t1z + 0.2f*R2xz + R3xz);
    l->yz += dir*(T0*yz - z*t1y - y*t1z + 0.2f*R2yz + R3yz);

    dir *= v;
    T0 = 15.0f*m->m + 63.0f*R2;
    txx = T0*xx;
    tyy = T0*yy;

    t1 = 3.0f*m->m + 7.0f*R2;
    t2x = -7.0f*R2x;
    t2y = -7.0f*R2y;
    t2z = -7.0f*R2z;
    t1xx = txx - t1 + 2.0f*x*t2x + R2xx;
    t1yy = tyy - t1 + 2.0f*y*t2y + R2yy;

    l->xxx += dir*(x*(txx - 3.0f*(t1 - t2x*x - R2xx)) + 3.0f*R2x);
    l->yyy += dir*(y*(tyy - 3.0f*(t1 - t2y*y - R2yy)) + 3.0f*R2y);
    l->xxy += dir*(y*t1xx + xx*t2y + R2y + 2.0f*R2xy*x);
    l->xxz += dir*(z*t1xx + xx*t2z + R2z + 2.0f*R2xz*x);
    l->xyy += dir*(x*t1yy + yy*t2x + R2x + 2.0f*R2xy*y);
    l->yyz += dir*(z*t1yy + yy*t2z + R2z + 2.0f*R2yz*y);
    l->xyz += dir*(T0*xyz + (yz*t2x + xz*t2y + xy*t2z) + R2xy*z + R2yz*x + R2xz*y);

    dir *= v*m->m;
    txx = 105.0f*xx;
    tyy = 105.0f*yy;
    t2xx = txx - 90.0f;
    t2yy = tyy - 90.0f;
    l->xxxx += dir*(xx*t2xx + 9.0f);
    l->yyyy += dir*(yy*t2yy + 9.0f);
    t2xx += 45.0f;
    t2yy += 45.0f;
    l->xxxy += dir*xy*t2xx;
    l->xxxz += dir*xz*t2xx;
    l->xyyy += dir*xy*t2yy;
    l->yyyz += dir*yz*t2yy;
    t2xx += 30.0f;
    t2yy += 30.0f;
    l->xxyy += dir*(yy*t2xx - xx*15.0f + 3.0f);
    l->xxyz += dir*(yz*t2xx);
    l->xyyz += dir*(xz*t2yy);

    dir *= v;
    x *= dir;
    y *= dir;
    z *= dir;
    txx = 9.0f*xx - 10.0f;
    tyy = 9.0f*yy - 10.0f;
    xx *= 105.0f;
    yy *= 105.0f;
    xy *= z*105.0f;
    l->xxxxx += x*(xx*txx + 225.0f);
    l->yyyyy += y*(yy*tyy + 225.0f);
    txx += 4.0f;
    tyy += 4.0f;
    txxxx = xx*txx + 45.0f;
    tyyyy = yy*tyy + 45.0f;
    l->xxxxy += y*txxxx;
    l->xxxxz += z*txxxx;
    l->xyyyy += x*tyyyy;
    l->yyyyz += z*tyyyy;
    txx += 3.0f;
    tyy += 3.0f;
    l->xxxyz += xy*txx;
    l->xyyyz += xy*tyy;
    l->xxxyy += x*(yy*txx - xx + 45.0f);
    l->xxyyy += y*(xx*tyy - yy + 45.0f);
    tyy += 2.0f;
    l->xxyyz += z*(xx*tyy - yy + 15.0f);
    return(464.0);
}


/*
** Op Count = (*,+,-) = (,,) = 
*/
double momFlocrAddMono5(FLOCR *l,float v,float m,float dir,float x,float y,float z,float *tax,float *tay,float *taz) {
    float xx,xy,xz,yy,yz;
    float T0,txx,tyy,t1,t1xx,t1yy,t2,txxxx,tyyyy;

    x *= dir;
    y *= dir;
    z *= dir;
    dir = -dir;
    v *= dir;
    /*
    ** Calculate the funky distance terms.
    */
    xy = x*y;
    xz = x*z;
    yz = y*z;
    xx = x*x;
    yy = y*y;

    l->m += dir*m;

    dir *= v;
    T0 = -m*dir;

    *tax = T0*x;
    *tay = T0*y;
    *taz = T0*z;
    l->x -= *tax;
    l->y -= *tay;
    l->z -= *taz;

    dir *= v;
    T0 = 3.0f*m*dir;

    t1 = m*dir;
    l->xx += T0*xx - t1;
    l->yy += T0*yy - t1;
    l->xy += T0*xy;
    l->xz += T0*xz;
    l->yz += T0*yz;

    dir *= v;
    T0 = 15.0f*m*dir;
    txx = T0*xx;
    tyy = T0*yy;

    t1 = 3.0f*m*dir;
    t1xx = txx - t1;
    t1yy = tyy - t1;

    l->xxx += x*(txx - 3.0f*t1);
    l->yyy += y*(tyy - 3.0f*t1);
    l->xxy += y*t1xx;
    l->xxz += z*t1xx;
    l->xyy += x*t1yy;
    l->yyz += z*t1yy;
    l->xyz += T0*x*yz;

    dir *= v;
    T0 = 105.0f*m*dir;
    txx = T0*xx;
    tyy = T0*yy;
    t2 = 3.0f*m*dir;
    t1 = 5.0f*t2;

    l->xxxx += xx*(txx - 6.0f*t1) + 3.0f*t2;
    l->yyyy += yy*(tyy - 6.0f*t1) + 3.0f*t2;
    t1xx = txx - 3.0f*t1;
    t1yy = tyy - 3.0f*t1;
    l->xxxy += xy*t1xx;
    l->xxxz += xz*t1xx;
    l->xyyy += xy*t1yy;
    l->yyyz += yz*t1yy;
    l->xxyy += yy*txx - (xx + yy)*t1 + t2;
    l->xxyz += yz*(txx - t1);
    l->xyyz += xz*(tyy - t1);

    dir *= v*m;
    x *= dir;
    y *= dir;
    z *= dir;
    txx = 9.0f*xx - 10.0f;
    tyy = 9.0f*yy - 10.0f;
    xx *= 105.0f;
    yy *= 105.0f;
    xy *= z*105.0f;
    l->xxxxx += x*(xx*txx + 225.0f);
    l->yyyyy += y*(yy*tyy + 225.0f);
    txx += 4.0f;
    tyy += 4.0f;
    txxxx = xx*txx + 45.0f;
    tyyyy = yy*tyy + 45.0f;
    l->xxxxy += y*txxxx;
    l->xxxxz += z*txxxx;
    l->xyyyy += x*tyyyy;
    l->yyyyz += z*tyyyy;
    txx += 3.0f;
    tyy += 3.0f;
    l->xxxyz += xy*txx;
    l->xyyyz += xy*tyy;
    l->xxxyy += x*(yy*txx - xx + 45.0f);
    l->xxyyy += y*(xx*tyy - yy + 45.0f);
    tyy += 2.0f;
    l->xxyyz += z*(xx*tyy - yy + 15.0f);
    return(200.0);
}


void momEvalLocr(LOCR *l,momFloat x,momFloat y,momFloat z,
		 momFloat *fPot,momFloat *ax,momFloat *ay,momFloat *az) {
    const momFloat onethird = 1.0/3.0;
    momFloat xx,xy,xz,yy,yz,zz,xxx,xxz,yyy,yyz,xxy,xyy,xyz,xxxx,xxxy,xxyy,xyyy,yyyy,xxxz,xxyz,xyyz,yyyz;
    momFloat g1,A,Ax,Ay,Az,B,Bx,By,Bz,C,Cx,Cy,Cz,D,Dx,Dy,Dz;

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
    xxxx = 0.25*(x*xxx - z*xxz);
    xxxy = y*xxx;
    xxyy = xx*yy - 2*onethird*zz*zz;
    xyyy = x*yyy;
    yyyy = 0.25*(y*yyy - z*yyz);
    xxxz = onethird*x*z*xx;
    xxyz = y*xxz;
    xyyz = x*yyz;
    yyyz = y*yyz;
    /*
     ** Now calculate the interaction.
     */
    Dx = l->xxxxx*xxxx + l->xxxxy*xxxy + l->xxxyy*xxyy + l->xxyyy*xyyy + l->xyyyy*yyyy + l->xxxxz*xxxz + l->xxxyz*xxyz + l->xxyyz*xyyz + l->xyyyz*yyyz;
    Dy = l->xxxxy*xxxx + l->xxxyy*xxxy + l->xxyyy*xxyy + l->xyyyy*xyyy + l->yyyyy*yyyy + l->xxxyz*xxxz + l->xxyyz*xxyz + l->xyyyz*xyyz + l->yyyyz*yyyz;
    Dz = l->xxxxz*xxxx + l->xxxyz*xxxy + l->xxyyz*xxyy + l->xyyyz*xyyy + l->yyyyz*yyyy
	 - l->xxxxx*xxxz - l->xxxxy*xxyz - l->xxxyy*(xxxz + xyyz) - l->xxyyy*(xxyz + yyyz) + l->xyyyy*xyyz + l->yyyyy*yyyz;
    Cx = l->xxxx*xxx + l->xyyy*yyy + l->xxxy*xxy + l->xxxz*xxz + l->xxyy*xyy + l->xxyz*xyz + l->xyyz*yyz;
    Cy = l->xyyy*xyy + l->xxxy*xxx + l->yyyy*yyy + l->yyyz*yyz + l->xxyy*xxy + l->xxyz*xxz + l->xyyz*xyz;
    Cz = -l->xxxx*xxz - (l->xyyy + l->xxxy)*xyz - l->yyyy*yyz + l->xxxz*xxx + l->yyyz*yyy - l->xxyy*(xxz + yyz) + l->xxyz*xxy + l->xyyz*xyy;
    Bx = l->xxx*xx + l->xyy*yy + l->xxy*xy + l->xxz*xz + l->xyz*yz;
    By = l->xyy*xy + l->xxy*xx + l->yyy*yy + l->yyz*yz + l->xyz*xz;
    Bz = -(l->xxx + l->xyy)*xz - (l->xxy + l->yyy)*yz + l->xxz*xx + l->yyz*yy + l->xyz*xy;
    Ax = l->xx*x + l->xy*y + l->xz*z;
    Ay = l->yy*y + l->xy*x + l->yz*z;
    Az = -(l->xx + l->yy)*z + l->xz*x + l->yz*y;
    D = 0.2*(Dx*x + Dy*y + Dz*z);
    C = 0.25*(Cx*x + Cy*y + Cz*z);
    B = onethird*(Bx*x + By*y + Bz*z);
    A = 0.5*(Ax*x + Ay*y + Az*z);
    g1 = x*l->x + y*l->y + z*l->z;
    *ax -= l->x + Ax + Bx + Cx + Dx;
    *ay -= l->y + Ay + By + Cy + Dy;
    *az -= l->z + Az + Bz + Cz + Dz;
    *fPot += l->m + g1 + A + B + C + D;
    }


void momEvalFlocr(FLOCR *l,float v,float x,float y,float z,
		  float *fPot,float *ax,float *ay,float *az) {
    const float onethird = 1.0f/3.0f;
    float xx,xy,xz,yy,yz,zz,xxx,xxz,yyy,yyz,xxy,xyy,xyz,xxxx,xxxy,xxyy,xyyy,yyyy,xxxz,xxyz,xyyz,yyyz;
    float g1,A,Ax,Ay,Az,B,Bx,By,Bz,C,Cx,Cy,Cz,D,Dx,Dy,Dz;
    float iv;

    /*
     ** Calculate the funky distance terms, but first scale x,y,z so that they are of order unity!
     */
    iv = 1/v;
    x *= iv;
    y *= iv;
    z *= iv;
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
    xxxx = 0.25*(x*xxx - z*xxz);
    xxxy = y*xxx;
    xxyy = xx*yy - 2*onethird*zz*zz;
    xyyy = x*yyy;
    yyyy = 0.25*(y*yyy - z*yyz);
    xxxz = onethird*x*z*xx;
    xxyz = y*xxz;
    xyyz = x*yyz;
    yyyz = y*yyz;
    /*
     ** Now calculate the interaction.
     */
    Dx = l->xxxxx*xxxx + l->xxxxy*xxxy + l->xxxyy*xxyy + l->xxyyy*xyyy + l->xyyyy*yyyy + l->xxxxz*xxxz + l->xxxyz*xxyz + l->xxyyz*xyyz + l->xyyyz*yyyz;
    Dy = l->xxxxy*xxxx + l->xxxyy*xxxy + l->xxyyy*xxyy + l->xyyyy*xyyy + l->yyyyy*yyyy + l->xxxyz*xxxz + l->xxyyz*xxyz + l->xyyyz*xyyz + l->yyyyz*yyyz;
    Dz = l->xxxxz*xxxx + l->xxxyz*xxxy + l->xxyyz*xxyy + l->xyyyz*xyyy + l->yyyyz*yyyy
	 - l->xxxxx*xxxz - l->xxxxy*xxyz - l->xxxyy*(xxxz + xyyz) - l->xxyyy*(xxyz + yyyz) + l->xyyyy*xyyz + l->yyyyy*yyyz;
    Cx = l->xxxx*xxx + l->xyyy*yyy + l->xxxy*xxy + l->xxxz*xxz + l->xxyy*xyy + l->xxyz*xyz + l->xyyz*yyz;
    Cy = l->xyyy*xyy + l->xxxy*xxx + l->yyyy*yyy + l->yyyz*yyz + l->xxyy*xxy + l->xxyz*xxz + l->xyyz*xyz;
    Cz = -l->xxxx*xxz - (l->xyyy + l->xxxy)*xyz - l->yyyy*yyz + l->xxxz*xxx + l->yyyz*yyy - l->xxyy*(xxz + yyz) + l->xxyz*xxy + l->xyyz*xyy;
    Bx = l->xxx*xx + l->xyy*yy + l->xxy*xy + l->xxz*xz + l->xyz*yz;
    By = l->xyy*xy + l->xxy*xx + l->yyy*yy + l->yyz*yz + l->xyz*xz;
    Bz = -(l->xxx + l->xyy)*xz - (l->xxy + l->yyy)*yz + l->xxz*xx + l->yyz*yy + l->xyz*xy;
    Ax = l->xx*x + l->xy*y + l->xz*z;
    Ay = l->yy*y + l->xy*x + l->yz*z;
    Az = -(l->xx + l->yy)*z + l->xz*x + l->yz*y;
    D = 0.2*(Dx*x + Dy*y + Dz*z);
    C = 0.25*(Cx*x + Cy*y + Cz*z);
    B = onethird*(Bx*x + By*y + Bz*z);
    A = 0.5*(Ax*x + Ay*y + Az*z);
    g1 = x*l->x + y*l->y + z*l->z;
    *ax -= iv*(l->x + Ax + Bx + Cx + Dx);
    *ay -= iv*(l->y + Ay + By + Cy + Dy);
    *az -= iv*(l->z + Az + Bz + Cz + Dz);
    *fPot += l->m + g1 + A + B + C + D;
    }


void momClearMomc(MOMC *l) {
    l->m = 0;
    l->xx = 0;
    l->yy = 0;
    l->xy = 0;
    l->xz = 0;
    l->yz = 0;
    l->xxx = 0;
    l->xyy = 0;
    l->xxy = 0;
    l->yyy = 0;
    l->xxz = 0;
    l->yyz = 0;
    l->xyz = 0;
    l->xxxx = 0;
    l->xyyy = 0;
    l->xxxy = 0;
    l->yyyy = 0;
    l->xxxz = 0;
    l->yyyz = 0;
    l->xxyy = 0;
    l->xxyz = 0;
    l->xyyz = 0;
    l->zz = 0;
    l->xzz = 0;
    l->yzz = 0;
    l->zzz = 0;
    l->xxzz = 0;
    l->xyzz = 0;
    l->xzzz = 0;
    l->yyzz = 0;
    l->yzzz = 0;
    l->zzzz = 0;
    }

void momClearMomr(MOMR *l) {
    l->m = 0;
    l->xx = 0;
    l->yy = 0;
    l->xy = 0;
    l->xz = 0;
    l->yz = 0;
    l->xxx = 0;
    l->xyy = 0;
    l->xxy = 0;
    l->yyy = 0;
    l->xxz = 0;
    l->yyz = 0;
    l->xyz = 0;
    l->xxxx = 0;
    l->xyyy = 0;
    l->xxxy = 0;
    l->yyyy = 0;
    l->xxxz = 0;
    l->yyyz = 0;
    l->xxyy = 0;
    l->xxyz = 0;
    l->xyyz = 0;
    }

void momClearFmomr(FMOMR *l) {
    l->m = 0;
    l->xx = 0;
    l->yy = 0;
    l->xy = 0;
    l->xz = 0;
    l->yz = 0;
    l->xxx = 0;
    l->xyy = 0;
    l->xxy = 0;
    l->yyy = 0;
    l->xxz = 0;
    l->yyz = 0;
    l->xyz = 0;
    l->xxxx = 0;
    l->xyyy = 0;
    l->xxxy = 0;
    l->yyyy = 0;
    l->xxxz = 0;
    l->yyyz = 0;
    l->xxyy = 0;
    l->xxyz = 0;
    l->xyyz = 0;
    }

void momClearLocr(LOCR *l) {
    l->m = 0;
    l->x = 0;
    l->y = 0;
    l->z = 0;
    l->xx = 0;
    l->yy = 0;
    l->xy = 0;
    l->xz = 0;
    l->yz = 0;
    l->xxx = 0;
    l->xyy = 0;
    l->xxy = 0;
    l->yyy = 0;
    l->xxz = 0;
    l->yyz = 0;
    l->xyz = 0;
    l->xxxx = 0;
    l->xyyy = 0;
    l->xxxy = 0;
    l->yyyy = 0;
    l->xxxz = 0;
    l->yyyz = 0;
    l->xxyy = 0;
    l->xxyz = 0;
    l->xyyz = 0;
    l->xxxxx = 0;
    l->xyyyy = 0;
    l->xxxxy = 0;
    l->yyyyy = 0;
    l->xxxxz = 0;
    l->yyyyz = 0;
    l->xxxyy = 0;
    l->xxyyy = 0;
    l->xxxyz = 0;
    l->xyyyz = 0;
    l->xxyyz = 0;
    }

void momClearFlocr(FLOCR *l) {
    l->m = 0;
    l->x = 0;
    l->y = 0;
    l->z = 0;
    l->xx = 0;
    l->yy = 0;
    l->xy = 0;
    l->xz = 0;
    l->yz = 0;
    l->xxx = 0;
    l->xyy = 0;
    l->xxy = 0;
    l->yyy = 0;
    l->xxz = 0;
    l->yyz = 0;
    l->xyz = 0;
    l->xxxx = 0;
    l->xyyy = 0;
    l->xxxy = 0;
    l->yyyy = 0;
    l->xxxz = 0;
    l->yyyz = 0;
    l->xxyy = 0;
    l->xxyz = 0;
    l->xyyz = 0;
    l->xxxxx = 0;
    l->xyyyy = 0;
    l->xxxxy = 0;
    l->yyyyy = 0;
    l->xxxxz = 0;
    l->yyyyz = 0;
    l->xxxyy = 0;
    l->xxyyy = 0;
    l->xxxyz = 0;
    l->xyyyz = 0;
    l->xxyyz = 0;
    }


/*
 ** This function adds the complete moment ma to the complete moment mc
 */
void momAddMomc(MOMC *mc,MOMC *ma) {
    mc->m += ma->m;
    mc->xx += ma->xx;
    mc->yy += ma->yy;
    mc->xy += ma->xy;
    mc->xz += ma->xz;
    mc->yz += ma->yz;
    mc->xxx += ma->xxx;
    mc->xyy += ma->xyy;
    mc->xxy += ma->xxy;
    mc->yyy += ma->yyy;
    mc->xxz += ma->xxz;
    mc->yyz += ma->yyz;
    mc->xyz += ma->xyz;
    mc->xxxx += ma->xxxx;
    mc->xyyy += ma->xyyy;
    mc->xxxy += ma->xxxy;
    mc->yyyy += ma->yyyy;
    mc->xxxz += ma->xxxz;
    mc->yyyz += ma->yyyz;
    mc->xxyy += ma->xxyy;
    mc->xxyz += ma->xxyz;
    mc->xyyz += ma->xyyz;
    mc->zz += ma->zz;
    mc->xzz += ma->xzz;
    mc->yzz += ma->yzz;
    mc->zzz += ma->zzz;
    mc->xxzz += ma->xxzz;
    mc->xyzz += ma->xyzz;
    mc->xzzz += ma->xzzz;
    mc->yyzz += ma->yyzz;
    mc->yzzz += ma->yzzz;
    mc->zzzz += ma->zzzz;
    }


/*
 ** This function adds the reduced moment ma to the reduced moment mr
 */
void momAddMomr(MOMR *mr,MOMR *ma) {
    mr->m += ma->m;
    mr->xx += ma->xx;
    mr->yy += ma->yy;
    mr->xy += ma->xy;
    mr->xz += ma->xz;
    mr->yz += ma->yz;
    mr->xxx += ma->xxx;
    mr->xyy += ma->xyy;
    mr->xxy += ma->xxy;
    mr->yyy += ma->yyy;
    mr->xxz += ma->xxz;
    mr->yyz += ma->yyz;
    mr->xyz += ma->xyz;
    mr->xxxx += ma->xxxx;
    mr->xyyy += ma->xyyy;
    mr->xxxy += ma->xxxy;
    mr->yyyy += ma->yyyy;
    mr->xxxz += ma->xxxz;
    mr->yyyz += ma->yyyz;
    mr->xxyy += ma->xxyy;
    mr->xxyz += ma->xxyz;
    mr->xyyz += ma->xyyz;
    }


/*
 ** This function adds the reduced moment ma to the reduced moment mr.
 */
void momAddFmomr(FMOMR *mr,FMOMR *ma) {
    mr->m += ma->m;
    mr->xx += ma->xx;
    mr->yy += ma->yy;
    mr->xy += ma->xy;
    mr->xz += ma->xz;
    mr->yz += ma->yz;
    mr->xxx += ma->xxx;
    mr->xyy += ma->xyy;
    mr->xxy += ma->xxy;
    mr->yyy += ma->yyy;
    mr->xxz += ma->xxz;
    mr->yyz += ma->yyz;
    mr->xyz += ma->xyz;
    mr->xxxx += ma->xxxx;
    mr->xyyy += ma->xyyy;
    mr->xxxy += ma->xxxy;
    mr->yyyy += ma->yyyy;
    mr->xxxz += ma->xxxz;
    mr->yyyz += ma->yyyz;
    mr->xxyy += ma->xxyy;
    mr->xxyz += ma->xxyz;
    mr->xyyz += ma->xyyz;
    }


/*
 ** This function adds the reduced scaled moment ma to the reduced scaled moment mr.
 ** It needs to correctly rescale the moments of ma to be compatible with the scaling of mr.
 */
void momScaledAddFmomr(FMOMR *mr,float ur,FMOMR *ma,float ua) {
    float f,s;
    assert(ur > 0.0 && ua > 0);
    f = ua/ur;
    s = f;
    mr->m += ma->m;
    s *= f;
    mr->xx += s*ma->xx;
    mr->yy += s*ma->yy;
    mr->xy += s*ma->xy;
    mr->xz += s*ma->xz;
    mr->yz += s*ma->yz;
    s *= f;
    mr->xxx += s*ma->xxx;
    mr->xyy += s*ma->xyy;
    mr->xxy += s*ma->xxy;
    mr->yyy += s*ma->yyy;
    mr->xxz += s*ma->xxz;
    mr->yyz += s*ma->yyz;
    mr->xyz += s*ma->xyz;
    s *= f;
    mr->xxxx += s*ma->xxxx;
    mr->xyyy += s*ma->xyyy;
    mr->xxxy += s*ma->xxxy;
    mr->yyyy += s*ma->yyyy;
    mr->xxxz += s*ma->xxxz;
    mr->yyyz += s*ma->yyyz;
    mr->xxyy += s*ma->xxyy;
    mr->xxyz += s*ma->xxyz;
    mr->xyyz += s*ma->xyyz;
    }


/*
 ** This function rescales reduced scaled moment mr.
 */
void momRescaleFmomr(FMOMR *mr,float unew,float uold) {
    float f,s;
    f = uold/unew;
    s = f;
    s *= f;
    mr->xx *= s;
    mr->yy *= s;
    mr->xy *= s;
    mr->xz *= s;
    mr->yz *= s;
    s *= f;
    mr->xxx *= s;
    mr->xyy *= s;
    mr->xxy *= s;
    mr->yyy *= s;
    mr->xxz *= s;
    mr->yyz *= s;
    mr->xyz *= s;
    s *= f;
    mr->xxxx *= s;
    mr->xyyy *= s;
    mr->xxxy *= s;
    mr->yyyy *= s;
    mr->xxxz *= s;
    mr->yyyz *= s;
    mr->xxyy *= s;
    mr->xxyz *= s;
    mr->xyyz *= s;
    }

/*
 ** This function multiply-adds the complete moment ma
 */
void momMulAddMomc(MOMC *mc,momFloat m,MOMC *ma) {
    mc->m += m*ma->m;
    mc->xx += m*ma->xx;
    mc->yy += m*ma->yy;
    mc->xy += m*ma->xy;
    mc->xz += m*ma->xz;
    mc->yz += m*ma->yz;
    mc->xxx += m*ma->xxx;
    mc->xyy += m*ma->xyy;
    mc->xxy += m*ma->xxy;
    mc->yyy += m*ma->yyy;
    mc->xxz += m*ma->xxz;
    mc->yyz += m*ma->yyz;
    mc->xyz += m*ma->xyz;
    mc->xxxx += m*ma->xxxx;
    mc->xyyy += m*ma->xyyy;
    mc->xxxy += m*ma->xxxy;
    mc->yyyy += m*ma->yyyy;
    mc->xxxz += m*ma->xxxz;
    mc->yyyz += m*ma->yyyz;
    mc->xxyy += m*ma->xxyy;
    mc->xxyz += m*ma->xxyz;
    mc->xyyz += m*ma->xyyz;
    mc->zz += m*ma->zz;
    mc->xzz += m*ma->xzz;
    mc->yzz += m*ma->yzz;
    mc->zzz += m*ma->zzz;
    mc->xxzz += m*ma->xxzz;
    mc->xyzz += m*ma->xyzz;
    mc->xzzz += m*ma->xzzz;
    mc->yyzz += m*ma->yyzz;
    mc->yzzz += m*ma->yzzz;
    mc->zzzz += m*ma->zzzz;
    }


/*
 ** This function multiply-adds the reduced moment ma
 */
void momMulAddMomr(MOMR *mr,momFloat m,MOMR *ma) {
    mr->m += m*ma->m;
    mr->xx += m*ma->xx;
    mr->yy += m*ma->yy;
    mr->xy += m*ma->xy;
    mr->xz += m*ma->xz;
    mr->yz += m*ma->yz;
    mr->xxx += m*ma->xxx;
    mr->xyy += m*ma->xyy;
    mr->xxy += m*ma->xxy;
    mr->yyy += m*ma->yyy;
    mr->xxz += m*ma->xxz;
    mr->yyz += m*ma->yyz;
    mr->xyz += m*ma->xyz;
    mr->xxxx += m*ma->xxxx;
    mr->xyyy += m*ma->xyyy;
    mr->xxxy += m*ma->xxxy;
    mr->yyyy += m*ma->yyyy;
    mr->xxxz += m*ma->xxxz;
    mr->yyyz += m*ma->yyyz;
    mr->xxyy += m*ma->xxyy;
    mr->xxyz += m*ma->xxyz;
    mr->xyyz += m*ma->xyyz;
    }


/*
 ** This function multiply-adds the reduced scaled moment ma
 */
void momMulAddFmomr(FMOMR *mr,float ur,float m,FMOMR *ma,float ua) {
    float f;
    assert(ua > 0.0 && ur > 0.0);
    f = ua/ur;
    mr->m += m*ma->m;
    m *= f;
    m *= f;
    mr->xx += m*ma->xx;
    mr->yy += m*ma->yy;
    mr->xy += m*ma->xy;
    mr->xz += m*ma->xz;
    mr->yz += m*ma->yz;
    m *= f;
    mr->xxx += m*ma->xxx;
    mr->xyy += m*ma->xyy;
    mr->xxy += m*ma->xxy;
    mr->yyy += m*ma->yyy;
    mr->xxz += m*ma->xxz;
    mr->yyz += m*ma->yyz;
    mr->xyz += m*ma->xyz;
    m *= f;
    mr->xxxx += m*ma->xxxx;
    mr->xyyy += m*ma->xyyy;
    mr->xxxy += m*ma->xxxy;
    mr->yyyy += m*ma->yyyy;
    mr->xxxz += m*ma->xxxz;
    mr->yyyz += m*ma->yyyz;
    mr->xxyy += m*ma->xxyy;
    mr->xxyz += m*ma->xxyz;
    mr->xyyz += m*ma->xyyz;
    }


/*
** This function adds the reduced local expansion la to the reduced local explansion lr.
*/
void momAddLocr(LOCR *lr,LOCR *la) {
    lr->m += la->m;
    lr->x += la->x;
    lr->y += la->y;
    lr->z += la->z;
    lr->xx += la->xx;
    lr->yy += la->yy;
    lr->xy += la->xy;
    lr->xz += la->xz;
    lr->yz += la->yz;
    lr->xxx += la->xxx;
    lr->xyy += la->xyy;
    lr->xxy += la->xxy;
    lr->yyy += la->yyy;
    lr->xxz += la->xxz;
    lr->yyz += la->yyz;
    lr->xyz += la->xyz;
    lr->xxxx += la->xxxx;
    lr->xyyy += la->xyyy;
    lr->xxxy += la->xxxy;
    lr->yyyy += la->yyyy;
    lr->xxxz += la->xxxz;
    lr->yyyz += la->yyyz;
    lr->xxyy += la->xxyy;
    lr->xxyz += la->xxyz;
    lr->xyyz += la->xyyz;
    lr->xxxxx += la->xxxxx;
    lr->xyyyy += la->xyyyy;
    lr->xxxxy += la->xxxxy;
    lr->yyyyy += la->yyyyy;
    lr->xxxxz += la->xxxxz;
    lr->yyyyz += la->yyyyz;
    lr->xxxyy += la->xxxyy;
    lr->xxyyy += la->xxyyy;
    lr->xxxyz += la->xxxyz;
    lr->xyyyz += la->xyyyz;
    lr->xxyyz += la->xxyyz;
}

/*
** This function adds the reduced local expansion la to the reduced local explansion lr.
*/
void momAddFlocr(FLOCR *lr,FLOCR *la) {
    lr->m += la->m;
    lr->x += la->x;
    lr->y += la->y;
    lr->z += la->z;
    lr->xx += la->xx;
    lr->yy += la->yy;
    lr->xy += la->xy;
    lr->xz += la->xz;
    lr->yz += la->yz;
    lr->xxx += la->xxx;
    lr->xyy += la->xyy;
    lr->xxy += la->xxy;
    lr->yyy += la->yyy;
    lr->xxz += la->xxz;
    lr->yyz += la->yyz;
    lr->xyz += la->xyz;
    lr->xxxx += la->xxxx;
    lr->xyyy += la->xyyy;
    lr->xxxy += la->xxxy;
    lr->yyyy += la->yyyy;
    lr->xxxz += la->xxxz;
    lr->yyyz += la->yyyz;
    lr->xxyy += la->xxyy;
    lr->xxyz += la->xxyz;
    lr->xyyz += la->xyyz;
    lr->xxxxx += la->xxxxx;
    lr->xyyyy += la->xyyyy;
    lr->xxxxy += la->xxxxy;
    lr->yyyyy += la->yyyyy;
    lr->xxxxz += la->xxxxz;
    lr->yyyyz += la->yyyyz;
    lr->xxxyy += la->xxxyy;
    lr->xxyyy += la->xxyyy;
    lr->xxxyz += la->xxxyz;
    lr->xyyyz += la->xyyyz;
    lr->xxyyz += la->xxyyz;
}


/*
 ** This function adds the reduced scaled local expansion la to the reduced scaled local expansion lr.
 ** It needs to correctly rescale the elements of la to be compatible with the scaling of lr.
 */
void momScaledAddFlocr(FLOCR *lr,float vr,FLOCR *la,float va) {
    float f,s;
    assert(vr > 0.0 && va > 0);
    f = va/vr;
    s = f;
    lr->m += la->m;
    lr->x += s*la->x;
    lr->y += s*la->y;
    lr->z += s*la->z;
    s *= f;
    lr->xx += s*la->xx;
    lr->yy += s*la->yy;
    lr->xy += s*la->xy;
    lr->xz += s*la->xz;
    lr->yz += s*la->yz;
    s *= f;
    lr->xxx += s*la->xxx;
    lr->xyy += s*la->xyy;
    lr->xxy += s*la->xxy;
    lr->yyy += s*la->yyy;
    lr->xxz += s*la->xxz;
    lr->yyz += s*la->yyz;
    lr->xyz += s*la->xyz;
    s *= f;
    lr->xxxx += s*la->xxxx;
    lr->xyyy += s*la->xyyy;
    lr->xxxy += s*la->xxxy;
    lr->yyyy += s*la->yyyy;
    lr->xxxz += s*la->xxxz;
    lr->yyyz += s*la->yyyz;
    lr->xxyy += s*la->xxyy;
    lr->xxyz += s*la->xxyz;
    lr->xyyz += s*la->xyyz;
    s *= f;
    lr->xxxxx += s*la->xxxxx;
    lr->xyyyy += s*la->xyyyy;
    lr->xxxxy += s*la->xxxxy;
    lr->yyyyy += s*la->yyyyy;
    lr->xxxxz += s*la->xxxxz;
    lr->yyyyz += s*la->yyyyz;
    lr->xxxyy += s*la->xxxyy;
    lr->xxyyy += s*la->xxyyy;
    lr->xxxyz += s*la->xxxyz;
    lr->xyyyz += s*la->xyyyz;
    lr->xxyyz += s*la->xxyyz;
    }


/*
 ** This function rescales the reduced scaled local expansion lr.
 */
void momRescaleFlocr(FLOCR *lr,float vnew,float vold) {
    float f,s;
    assert(vnew > 0.0 && vold > 0);
    f = vnew/vold;
    s = f;
    lr->x *= s;
    lr->y *= s;
    lr->z *= s;
    s *= f;
    lr->xx *= s;
    lr->yy *= s;
    lr->xy *= s;
    lr->xz *= s;
    lr->yz *= s;
    s *= f;
    lr->xxx *= s;
    lr->xyy *= s;
    lr->xxy *= s;
    lr->yyy *= s;
    lr->xxz *= s;
    lr->yyz *= s;
    lr->xyz *= s;
    s *= f;
    lr->xxxx *= s;
    lr->xyyy *= s;
    lr->xxxy *= s;
    lr->yyyy *= s;
    lr->xxxz *= s;
    lr->yyyz *= s;
    lr->xxyy *= s;
    lr->xxyz *= s;
    lr->xyyz *= s;
    s *= f;
    lr->xxxxx *= s;
    lr->xyyyy *= s;
    lr->xxxxy *= s;
    lr->yyyyy *= s;
    lr->xxxxz *= s;
    lr->yyyyz *= s;
    lr->xxxyy *= s;
    lr->xxyyy *= s;
    lr->xxxyz *= s;
    lr->xyyyz *= s;
    lr->xxyyz *= s;
    }


/*
 ** This function subtracts the complete moment ma from the complete moment mc
 ** (rarely used)
 */
void momSubMomc(MOMC *mc,MOMC *ma) {
    mc->m -= ma->m;
    mc->xx -= ma->xx;
    mc->yy -= ma->yy;
    mc->xy -= ma->xy;
    mc->xz -= ma->xz;
    mc->yz -= ma->yz;
    mc->xxx -= ma->xxx;
    mc->xyy -= ma->xyy;
    mc->xxy -= ma->xxy;
    mc->yyy -= ma->yyy;
    mc->xxz -= ma->xxz;
    mc->yyz -= ma->yyz;
    mc->xyz -= ma->xyz;
    mc->xxxx -= ma->xxxx;
    mc->xyyy -= ma->xyyy;
    mc->xxxy -= ma->xxxy;
    mc->yyyy -= ma->yyyy;
    mc->xxxz -= ma->xxxz;
    mc->yyyz -= ma->yyyz;
    mc->xxyy -= ma->xxyy;
    mc->xxyz -= ma->xxyz;
    mc->xyyz -= ma->xyyz;
    mc->zz -= ma->zz;
    mc->xzz -= ma->xzz;
    mc->yzz -= ma->yzz;
    mc->zzz -= ma->zzz;
    mc->xxzz -= ma->xxzz;
    mc->xyzz -= ma->xyzz;
    mc->xzzz -= ma->xzzz;
    mc->yyzz -= ma->yyzz;
    mc->yzzz -= ma->yzzz;
    mc->zzzz -= ma->zzzz;
    }


/*
 ** This function subtracts the reduced moment ma from the reduced moment mc
 ** (rarely used)
 */
void momSubMomr(MOMR *mr,MOMR *ma) {
    mr->m -= ma->m;
    mr->xx -= ma->xx;
    mr->yy -= ma->yy;
    mr->xy -= ma->xy;
    mr->xz -= ma->xz;
    mr->yz -= ma->yz;
    mr->xxx -= ma->xxx;
    mr->xyy -= ma->xyy;
    mr->xxy -= ma->xxy;
    mr->yyy -= ma->yyy;
    mr->xxz -= ma->xxz;
    mr->yyz -= ma->yyz;
    mr->xyz -= ma->xyz;
    mr->xxxx -= ma->xxxx;
    mr->xyyy -= ma->xyyy;
    mr->xxxy -= ma->xxxy;
    mr->yyyy -= ma->yyyy;
    mr->xxxz -= ma->xxxz;
    mr->yyyz -= ma->yyyz;
    mr->xxyy -= ma->xxyy;
    mr->xxyz -= ma->xxyz;
    mr->xyyz -= ma->xyyz;
    }


void momPrintMomc(MOMC *m) {
    printf("MOMC :%20.15g\n",(double)m->m);
    printf("   xx:%20.15g   yy:%20.15g   zz:%20.15g\n",(double)m->xx,(double)m->yy,(double)m->zz);
    printf("   xy:%20.15g   yz:%20.15g   xz:%20.15g\n",(double)m->xy,(double)m->yz,(double)m->xz);
    printf("  xxx:%20.15g  xyy:%20.15g  xzz:%20.15g\n",(double)m->xxx,(double)m->xyy,(double)m->xzz);
    printf("  xxy:%20.15g  yyy:%20.15g  yzz:%20.15g\n",(double)m->xxy,(double)m->yyy,(double)m->yzz);
    printf("  xxz:%20.15g  yyz:%20.15g  zzz:%20.15g\n",(double)m->xxz,(double)m->yyz,(double)m->zzz);
    printf("  xyz:%20.15g\n",(double)m->xyz);
    printf(" xxxx:%20.15g xxxy:%20.15g xxxz:%20.15g\n",(double)m->xxxx,(double)m->xxxy,(double)m->xxxz);
    printf(" xyyy:%20.15g yyyy:%20.15g yyyz:%20.15g\n",(double)m->xyyy,(double)m->yyyy,(double)m->yyyz);
    printf(" xzzz:%20.15g yzzz:%20.15g zzzz:%20.15g\n",(double)m->xzzz,(double)m->yzzz,(double)m->zzzz);
    printf(" xxyy:%20.15g xxyz:%20.15g xyyz:%20.15g\n",(double)m->xxyy,(double)m->xxyz,(double)m->xyyz);
    printf(" yyzz:%20.15g xxzz:%20.15g xyzz:%20.15g\n",(double)m->yyzz,(double)m->xxzz,(double)m->xyzz);
    }


void momPrintMomr(MOMR *m) {
    printf("MOMR :%20.15g\n",(double)m->m);
    printf("   xx:%20.15g   xy:%20.15g   xz:%20.15g\n",(double)m->xx,(double)m->xy,(double)m->xz);
    printf("   yy:%20.15g   yz:%20.15g  xxx:%20.15g\n",(double)m->yy,(double)m->yz,(double)m->xxx);
    printf("  xxy:%20.15g  xxz:%20.15g  xyy:%20.15g\n",(double)m->xxy,(double)m->xxz,(double)m->xyy);
    printf("  xyz:%20.15g  yyy:%20.15g  yyz:%20.15g\n",(double)m->xyz,(double)m->yyy,(double)m->yyz);
    printf(" xxxx:%20.15g xxxy:%20.15g xxxz:%20.15g\n",(double)m->xxxx,(double)m->xxxy,(double)m->xxxz);
    printf(" xxyy:%20.15g xxyz:%20.15g xyyy:%20.15g\n",(double)m->xxyy,(double)m->xxyz,(double)m->xyyy);
    printf(" xyyz:%20.15g yyyy:%20.15g yyyz:%20.15g\n",(double)m->xyyz,(double)m->yyyy,(double)m->yyyz);
    }

void momPrintFmomr(FMOMR *m,float u) {
    double uu = u;
    printf("FMOMR:%20.8g\n",m->m);
    uu *= u;
    printf("   xx:%20.8g   xy:%20.8g   xz:%20.8g\n",m->xx*uu,m->xy*uu,m->xz*uu);
    printf("   yy:%20.8g   yz:%20.8g",m->yy*uu,m->yz*uu);
    uu *= u;
    printf("  xxx:%20.8g\n",m->xxx*uu);
    printf("  xxy:%20.8g  xxz:%20.8g  xyy:%20.8g\n",m->xxy*uu,m->xxz*uu,m->xyy*uu);
    printf("  xyz:%20.8g  yyy:%20.8g  yyz:%20.8g\n",m->xyz*uu,m->yyy*uu,m->yyz*uu);
    uu *= u;
    printf(" xxxx:%20.8g xxxy:%20.8g xxxz:%20.8g\n",m->xxxx*uu,m->xxxy*uu,m->xxxz*uu);
    printf(" xxyy:%20.8g xxyz:%20.8g xyyy:%20.8g\n",m->xxyy*uu,m->xxyz*uu,m->xyyy*uu);
    printf(" xyyz:%20.8g yyyy:%20.8g yyyz:%20.8g\n",m->xyyz*uu,m->yyyy*uu,m->yyyz*uu);
    }

void momPrintLocr(LOCR *m) {
    printf("LOCR :%20.15g\n",(double)m->m);
    printf("    x:%20.15g     y:%20.15g     z:%20.15g\n",(double)m->x,(double)m->y,(double)m->z);
    printf("   xx:%20.15g    xy:%20.15g    xz:%20.15g\n",(double)m->xx,(double)m->xy,(double)m->xz);
    printf("   yy:%20.15g    yz:%20.15g   xxx:%20.15g\n",(double)m->yy,(double)m->yz,(double)m->xxx);
    printf("  xxy:%20.15g   xxz:%20.15g   xyy:%20.15g\n",(double)m->xxy,(double)m->xxz,(double)m->xyy);
    printf("  xyz:%20.15g   yyy:%20.15g   yyz:%20.15g\n",(double)m->xyz,(double)m->yyy,(double)m->yyz);
    printf(" xxxx:%20.15g  xxxy:%20.15g  xxxz:%20.15g\n",(double)m->xxxx,(double)m->xxxy,(double)m->xxxz);
    printf(" xxyy:%20.15g  xxyz:%20.15g  xyyy:%20.15g\n",(double)m->xxyy,(double)m->xxyz,(double)m->xyyy);
    printf(" xyyz:%20.15g  yyyy:%20.15g  yyyz:%20.15g\n",(double)m->xyyz,(double)m->yyyy,(double)m->yyyz);
    printf("xxxxx:%20.15g xxxxy:%20.15g xxxxz:%20.15g\n",(double)m->xxxxx,(double)m->xxxxy,(double)m->xxxxz);
    printf("xxxyy:%20.15g xxyyy:%20.15g xxxyz:%20.15g\n",(double)m->xxxyy,(double)m->xxyyy,(double)m->xxxyz);
    printf("xyyyy:%20.15g yyyyy:%20.15g xxyyz:%20.15g\n",(double)m->xyyyy,(double)m->yyyyy,(double)m->xxyyz);
    printf("xyyyz:%20.15g yyyyz:%20.15g              \n",(double)m->xyyyz,(double)m->yyyyz);
    }


void momPrintFlocr(FLOCR *m,float v) {
    double u = 1.0/((double)v);
    double uu;
    printf("FLOCR:%20.8g\n",m->m);
    printf("    x:%20.8g     y:%20.8g     z:%20.8g\n",m->x*u,m->y*u,m->z*u);
    uu = u*u;
    printf("   xx:%20.8g    xy:%20.8g    xz:%20.8g\n",m->xx*uu,m->xy*uu,m->xz*uu);
    printf("   yy:%20.8g    yz:%20.8g",m->yy*uu,m->yz*uu);
    uu *= u;
    printf("   xxx:%20.8g\n",m->xxx*uu);    
    printf("  xxy:%20.8g   xxz:%20.8g   xyy:%20.8g\n",m->xxy*uu,m->xxz*uu,m->xyy*uu);
    printf("  xyz:%20.8g   yyy:%20.8g   yyz:%20.8g\n",m->xyz*uu,m->yyy*uu,m->yyz*uu);
    uu *= u;
    printf(" xxxx:%20.8g  xxxy:%20.8g  xxxz:%20.8g\n",m->xxxx*uu,m->xxxy*uu,m->xxxz*uu);
    printf(" xxyy:%20.8g  xxyz:%20.8g  xyyy:%20.8g\n",m->xxyy*uu,m->xxyz*uu,m->xyyy*uu);
    printf(" xyyz:%20.8g  yyyy:%20.8g  yyyz:%20.8g\n",m->xyyz*uu,m->yyyy*uu,m->yyyz*uu);
    uu *= u;
    printf("xxxxx:%20.8g xxxxy:%20.8g xxxxz:%20.8g\n",m->xxxxx*uu,m->xxxxy*uu,m->xxxxz*uu);
    printf("xxxyy:%20.8g xxyyy:%20.8g xxxyz:%20.8g\n",m->xxxyy*uu,m->xxyyy*uu,m->xxxyz*uu);
    printf("xyyyy:%20.8g yyyyy:%20.8g xxyyz:%20.8g\n",m->xyyyy*uu,m->yyyyy*uu,m->xxyyz*uu);
    printf("xyyyz:%20.8g yyyyz:%20.8g             \n",m->xyyyz*uu,m->yyyyz*uu);
    }

