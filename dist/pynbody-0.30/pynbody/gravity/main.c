#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <time.h>
#include <unistd.h>
#include <sys/times.h>
#include <signal.h>

#include <sched.h>
#include <errno.h>

#include "kd.h"

/*==========================================================================*/
/* Debugging tools                                                          */
/*==========================================================================*/
#define DBG_LEVEL 0
#define DBG(lvl) if (DBG_LEVEL >= lvl) 

/*==========================================================================*/
/* Memory allocation wrappers.                                              */
/*==========================================================================*/

#if DBG_LEVEL >= 10000
long total_alloc = 0;
#define CALLOC(type, num) \
    (total_alloc += sizeof(type) * (num), \
    fprintf(stderr, "c'allocating %ld bytes [already alloc'd: %ld].\n", sizeof(type) * (num), total_alloc), \
    ((type *)calloc((num), sizeof(type))))
#else
#define CALLOC(type, num) ((type *)calloc((num), sizeof(type)))
#endif

#define MALLOC(type, num) ((type *)malloc((num) * sizeof(type)))


/*==========================================================================*/
/* Prototypes.                                                              */
/*==========================================================================*/

PyObject *treeinit(PyObject *self, PyObject *args);
PyObject *kdfree(PyObject *self, PyObject *args);

PyObject *calculate(PyObject *self, PyObject *args);

static PyMethodDef grav_methods[] = 
{
    {"treeinit", treeinit, METH_VARARGS, "treeinit"},
    {"free", kdfree, METH_VARARGS, "free"},

    {"calculate",  calculate,  METH_VARARGS, "calculate"},

    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initgrav()
{
    (void)Py_InitModule("grav", grav_methods);
}

/*==========================================================================*/
/* kdinit                                                                   */
/*==========================================================================*/
PyObject *treeinit(PyObject *self, PyObject *args)
{
    int nBucket, nbodies, nTreeBitsLo=14, nTreebitsHi=18;
    int i, j;
    KD kd = (KD)malloc(sizeof(struct kdContext));

    PyObject *pos;  // Nx3 Numpy array of positions
    PyObject *mass; // Nx1 Numpy array of masses
    double dTheta=0.7 * 2.0/sqrt(3.0);
    float fPeriod[3], fTemp, fSoft;
    for (i=0; i<3; i++) fPeriod[i] = 1.0;

    if (!PyArg_ParseTuple(args, "OOiif", &pos, &mass, &nBucket, &nbodies, &fSoft))
        return NULL;

    /*nbodies = PyArray_DIM(mass, 0);*/
    kdInitialize(kd, nbodies, nBucket, dTheta, nTreeBitsLo, nTreebitsHi, fPeriod);
    /*
    ** Allocate particles.
    */
    
    kd->pStorePRIVATE = (PARTICLE *)malloc(nbodies*kd->iParticleSize);
    assert(kd->pStorePRIVATE != NULL);

    Py_BEGIN_ALLOW_THREADS

    for (i=0; i < nbodies; i++)
    {
	for (j=0; j < 3; j++) {
	    fTemp = (float)*((double *)PyArray_GETPTR2(pos, i, j));
	    kd->pStorePRIVATE[i].r[j] = fTemp;
	    kd->pStorePRIVATE[i].a[j] = 0;
	    }
	fTemp = (float)*((double *)PyArray_GETPTR1(mass, i));
        kd->pStorePRIVATE[i].fMass = fTemp;
        kd->pStorePRIVATE[i].fPot = 0;
        kd->pStorePRIVATE[i].fSoft = fSoft;
    }

    kdTreeBuild(kd, nBucket);


    Py_END_ALLOW_THREADS

    return PyCObject_FromVoidPtr((void *)kd, NULL);
}

/*==========================================================================*/
/* kdfree                                                                   */
/*==========================================================================*/
PyObject *kdfree(PyObject *self, PyObject *args)
{
    KD kd;
    PyObject *kdobj;

    PyArg_ParseTuple(args, "O", &kdobj);
    kd = PyCObject_AsVoidPtr(kdobj);

    kdFinish(kd);

    return Py_None;
}


/*==========================================================================*/
/* calculate                                                                 */
/*==========================================================================*/
PyObject *calculate(PyObject *self, PyObject *args)
{
    long i, j;

    KD kd;
    int nReps=0, bPeriodic=0, bEwald=0;
    float fSoft;
    int nPos;
    PyObject *kdobj, *acc, *pot, *pos;
    PARTICLE *testParticles;
  
    PyArg_ParseTuple(args, "OOOOdi", &kdobj, &pos, &acc, &pot, &fSoft, &nPos);
    kd  = PyCObject_AsVoidPtr(kdobj);
    if (kd == NULL) return NULL;
    
    testParticles = (PARTICLE *)malloc(nPos*sizeof(PARTICLE));
    assert(testParticles != NULL);
    for (i=0; i< nPos; i++) {
	for (j=0; j < 3; j++) {
	    testParticles[i].r[j] = 
		(float)*((double *)PyArray_GETPTR2(pos, i, j));
	    testParticles[i].a[j] = 0;
	    }
	    testParticles[i].fMass = 0;
	    testParticles[i].fPot = 0;
	    testParticles[i].fSoft = fSoft;
	    testParticles[i].iOrder = i;
	}
    Py_BEGIN_ALLOW_THREADS

	kdGravWalk(kd, nReps, bPeriodic && bEwald, testParticles, nPos);

    Py_END_ALLOW_THREADS
    
    for (i=0; i < nPos; i++) {
	PyArray_SETITEM(pot, PyArray_GETPTR1(pot,i), PyFloat_FromDouble(testParticles[i].fPot));
	for (j=0; j < 3; j++) {
	    PyArray_SETITEM(acc, PyArray_GETPTR2(acc, i, j), 
			    PyFloat_FromDouble(testParticles[i].a[j]));
	    }
	}


    Py_DECREF(kd);
    /*
    Py_DECREF(pos);
    Py_DECREF(pot);
    Py_DECREF(acc);
    */
    Py_INCREF(Py_None);
    return Py_None;
}

