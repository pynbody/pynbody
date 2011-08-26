#ifndef MOMENTS_INCLUDED
#define MOMENTS_INCLUDED

#ifdef MOMQUAD
typedef long double momFloat;
#define sqrt(x)	sqrtl(x)
#else
typedef double momFloat;
#endif

/*
** The first 4 MOM/LOC structures can use different precisions by defining
** momFloat to float, double or long double, but are not guaranteed to be 
** safe to use at float precision. The moments are also assumed to be expanded
** about the center of mass, such that no diapole term is present.
*/

/*
 ** moment tensor components for reduced multipoles.
 */
typedef struct momReduced {
    momFloat m;
    momFloat xx,yy,xy,xz,yz;
    momFloat xxx,xyy,xxy,yyy,xxz,yyz,xyz;
    momFloat xxxx,xyyy,xxxy,yyyy,xxxz,yyyz,xxyy,xxyz,xyyz;
    } MOMR;

/*
 ** moment tensor components for complete multipoles.
 */
typedef struct momComplete {
    momFloat m;
    momFloat xx,yy,xy,xz,yz;
    momFloat xxx,xyy,xxy,yyy,xxz,yyz,xyz;
    momFloat xxxx,xyyy,xxxy,yyyy,xxxz,yyyz,xxyy,xxyz,xyyz;
    momFloat zz;
    momFloat xzz,yzz,zzz;
    momFloat xxzz,xyzz,xzzz,yyzz,yzzz,zzzz;
    } MOMC;

/*
 ** moment tensor components for reduced local expansion.
 ** note that we have the 5th-order terms here now!
 */
typedef struct locReduced {
    momFloat m;
    momFloat x,y,z;
    momFloat xx,yy,xy,xz,yz;
    momFloat xxx,xyy,xxy,yyy,xxz,yyz,xyz;
    momFloat xxxx,xyyy,xxxy,yyyy,xxxz,yyyz,xxyy,xxyz,xyyz;
    momFloat xxxxx,xyyyy,xxxxy,yyyyy,xxxxz,yyyyz,xxxyy,xxyyy,xxxyz,xyyyz,xxyyz;
    } LOCR;

/*
** The next set of data structures are intended specifically for use with float
** precision. These moments are usually scaled to a characteristic size of the 
** cell or volume. The convention is to use the scaling factor u for the multipole
** moments and scaling factor v for the local expansion.
*/
typedef struct fmomReduced {
    float m;
    float xx,yy,xy,xz,yz;
    float xxx,xyy,xxy,yyy,xxz,yyz,xyz;
    float xxxx,xyyy,xxxy,yyyy,xxxz,yyyz,xxyy,xxyz,xyyz;
    } FMOMR;

typedef struct flocReduced {
    float m;
    float x,y,z;
    float xx,yy,xy,xz,yz;
    float xxx,xyy,xxy,yyy,xxz,yyz,xyz;
    float xxxx,xyyy,xxxy,yyyy,xxxz,yyyz,xxyy,xxyz,xyyz;
    float xxxxx,xyyyy,xxxxy,yyyyy,xxxxz,yyyyz,xxxyy,xxyyy,xxxyz,xyyyz,xxyyz;
    } FLOCR;

/*
** Multipole moment generating functions.
*/
void momMakeMomc(MOMC *mc,momFloat m,momFloat x,momFloat y,momFloat z);
momFloat momMakeMomr(MOMR *mr,momFloat m,momFloat x,momFloat y,momFloat z);
float momMakeFmomr(FMOMR *mr,float m,float u,float x,float y,float z);
void momOldMakeMomr(MOMR *mr,momFloat m,momFloat x,momFloat y,momFloat z);
void momMomr2Momc(MOMR *ma,MOMC *mc);
void momReduceMomc(MOMC *mc,MOMR *mr);
/*
** Shifting functions.
*/
void momShiftMomc(MOMC *m,momFloat x,momFloat y,momFloat z);
void momShiftMomr(MOMR *m,momFloat x,momFloat y,momFloat z);
void momShiftFmomr(FMOMR *m,float u,float x,float y,float z);
double momShiftLocr(LOCR *l,momFloat x,momFloat y,momFloat z);
double momShiftFlocr(FLOCR *l,float v,float x,float y,float z);

/*
** All the variants of EvalMomr...
*/
void momEvalMomr(MOMR *m,momFloat dir,momFloat x,momFloat y,momFloat z,momFloat *fPot,momFloat *ax,momFloat *ay,momFloat *az,momFloat *magai);
void momGenEvalMomr(MOMR *m,momFloat g0,momFloat g1,momFloat g2,momFloat g3,momFloat g4,momFloat g5,momFloat x,momFloat y,momFloat z,momFloat *fPot,momFloat *ax,momFloat *ay,momFloat *az,momFloat *magai);
void momEvalFmomrcm(FMOMR *m,float u,float dir,float x,float y,float z,float *fPot,float *ax,float *ay,float *az,float *magai);
/*
** These are the prefered versions that should be used in pkdgrav2.
*/
double momLocrAddMomr5cm(LOCR *l,MOMR *m,momFloat dir,momFloat x,momFloat y,momFloat z,double *tax,double *tay,double *taz);
double momLocrAddMono5(LOCR *l,momFloat m,momFloat dir,momFloat x,momFloat y,momFloat z,double *tax,double *tay,double *taz);
double momFlocrAddFmomr5cm(FLOCR *l,float v,FMOMR *m,float u,float dir,float x,float y,float z,float *tax,float *tay,float *taz);
double momFlocrAddMono5(FLOCR *l,float v,float m,float dir,float x,float y,float z,float *tax,float *tay,float *taz);
/*
** All the functions for evaluating local expansions.
*/
void momEvalLocr(LOCR *l,momFloat x,momFloat y,momFloat z,
		 momFloat *fPot,momFloat *ax,momFloat *ay,momFloat *az);
void momEvalFlocr(FLOCR *l,float v,float x,float y,float z,
		  float *fPot,float *ax,float *ay,float *az);

/*
** Basic arithmetic functions.
*/
void momClearMomc(MOMC *l);
void momClearMomr(MOMR *l);
void momClearFmomr(FMOMR *l);
void momClearLocr(LOCR *l);
void momClearFlocr(FLOCR *l);
void momAddMomc(MOMC *mc,MOMC *ma);
void momAddMomr(MOMR *mr,MOMR *ma);
void momAddFmomr(FMOMR *mr,FMOMR *ma);
void momScaledAddFmomr(FMOMR *mr,float ur,FMOMR *ma,float ua);
void momRescaleFmomr(FMOMR *mr,float unew,float uold);
void momMulAddMomc(MOMC *mc,momFloat m,MOMC *ma);
void momMulAddMomr(MOMR *mr,momFloat m,MOMR *ma);
void momMulAddFmomr(FMOMR *mr,float ur,float m,FMOMR *ma,float ua);
void momAddLocr(LOCR *lr,LOCR *la);
void momAddFlocr(FLOCR *lr,FLOCR *la);
void momScaledAddFlocr(FLOCR *lr,float vr,FLOCR *la,float va);
void momRescaleFlocr(FLOCR *lr,float vnew,float vold);
void momSubMomc(MOMC *mc,MOMC *ma);
void momSubMomr(MOMR *mr,MOMR *ma);
/*
** Printing functions:
*/
void momPrintMomc(MOMC *m);
void momPrintMomr(MOMR *m);
void momPrintFmomr(FMOMR *m,float u);
void momPrintLocr(LOCR *l);
void momPrintFlocr(FLOCR *l,float v);


#endif
