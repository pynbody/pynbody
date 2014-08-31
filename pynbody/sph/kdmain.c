#include <Python.h>
#include <numpy/arrayobject.h>
#include "../bc_modules/capsulethunk.h"

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
#include "smooth.h"

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

PyObject *kdinit(PyObject *self, PyObject *args);
PyObject *kdfree(PyObject *self, PyObject *args);

PyObject *nn_start(PyObject *self, PyObject *args);
PyObject *nn_next(PyObject *self, PyObject *args);
PyObject *nn_stop(PyObject *self, PyObject *args);
PyObject *nn_rewind(PyObject *self, PyObject *args);

PyObject *populate(PyObject *self, PyObject *args);

/*==========================================================================*/
#define PROPID_HSM      1
#define PROPID_RHO      2
#define PROPID_MEANVEL  3
#define PROPID_VELDISP  4
/*==========================================================================*/

static PyMethodDef kdmain_methods[] =
{
    {"init", kdinit, METH_VARARGS, "init"},
    {"free", kdfree, METH_VARARGS, "free"},

    {"nn_start",  nn_start,  METH_VARARGS, "nn_start"},
    {"nn_next",   nn_next,   METH_VARARGS, "nn_next"},
    {"nn_stop",   nn_stop,   METH_VARARGS, "nn_stop"},
    {"nn_rewind", nn_rewind, METH_VARARGS, "nn_rewind"},

    {"populate",  populate,  METH_VARARGS, "populate"},

    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION>=3
static struct PyModuleDef ourdef = {
  PyModuleDef_HEAD_INIT,
  "kdmain",
  "KDTree module for pynbody",
  -1,
  kdmain_methods,
  NULL, NULL, NULL, NULL };
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION>=3
PyInit_kdmain(void)
#else
initkdmain(void)
#endif
{
  #if PY_MAJOR_VERSION>=3
    return PyModule_Create(&ourdef);
  #else
    (void)Py_InitModule("kdmain", kdmain_methods);
  #endif
}

/*==========================================================================*/
/* kdinit                                                                   */
/*==========================================================================*/
PyObject *kdinit(PyObject *self, PyObject *args)
{
    int nBucket;
    int i;

    PyObject *pos;  // Nx3 Numpy array of positions
    PyObject *mass; // Nx1 Numpy array of masses

    if (!PyArg_ParseTuple(args, "OOi", &pos, &mass, &nBucket))
        return NULL;

    KD kd = malloc(sizeof(*kd));
    kdInit(&kd, nBucket);

    int nbodies = PyArray_DIM(pos, 0);

    kd->nParticles = nbodies;
    kd->nActive = nbodies;

    kd->pNumpyPos = pos;
    kd->pNumpyMass = mass;
    kd->pNumpySmooth = NULL;
    kd->pNumpyDen = NULL;

    Py_INCREF(pos);
    Py_INCREF(mass);


    Py_BEGIN_ALLOW_THREADS


    // Allocate particles
    kd->p = (PARTICLE *)malloc(kd->nActive*sizeof(PARTICLE));
    assert(kd->p != NULL);

    for (i=0; i < nbodies; i++)
    {
        kd->p[i].iOrder = i;
        kd->p[i].iMark = 1;
        /*
        kd->p[i].r[0] = (float)*((double *)PyArray_GETPTR2(pos, i, 0));
        kd->p[i].r[1] = (float)*((double *)PyArray_GETPTR2(pos, i, 1));
        kd->p[i].r[2] = (float)*((double *)PyArray_GETPTR2(pos, i, 2));
        */
    }

    kdBuildTree(kd);

    Py_END_ALLOW_THREADS

    return PyCapsule_New((void *)kd, NULL, NULL);
}

/*==========================================================================*/
/* kdfree                                                                   */
/*==========================================================================*/
PyObject *kdfree(PyObject *self, PyObject *args)
{
    KD kd;
    PyObject *kdobj;

    PyArg_ParseTuple(args, "O", &kdobj);
    kd = PyCapsule_GetPointer(kdobj, NULL);

    kdFinish(kd);
    Py_XDECREF(kd->pNumpyPos);
    Py_XDECREF(kd->pNumpyMass);
    Py_XDECREF(kd->pNumpySmooth);
    Py_XDECREF(kd->pNumpyDen);
    return Py_None;
}

/*==========================================================================*/
/* nn_start                                                                 */
/*==========================================================================*/
PyObject *nn_start(PyObject *self, PyObject *args)
{
    KD kd;
    SMX smx;

    PyObject *kdobj;
    /* Nx1 Numpy arrays for smoothing length and density for calls that pass
       in those values from existing arrays
    */
    PyObject *smooth = NULL, *rho=NULL, *mass=NULL;

    int nSmooth, nProcs;
    long i;
    float hsm;

    PyArg_ParseTuple(args, "Oii|OOO", &kdobj, &nSmooth, &nProcs, &smooth, &rho, &mass);
    kd = PyCapsule_GetPointer(kdobj, NULL);

#define BIGFLOAT ((float)1.0e37)

    float fPeriod[3] = {BIGFLOAT, BIGFLOAT, BIGFLOAT};

    if(!smInit(&smx, kd, nSmooth, fPeriod)) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create smoothing context");
        return NULL;
    }

    smSmoothInitStep(smx, nProcs);

    if(mass!=NULL) {
      if(checkArray(mass)) return NULL;
      Py_XDECREF(kd->pNumpyMass);
      kd->pNumpyMass = mass;
      Py_INCREF(mass);
    }
    if(rho!=NULL) {
      if(checkArray(rho)) return NULL;
      Py_XDECREF(kd->pNumpyDen);
      kd->pNumpyDen = rho;
      Py_INCREF(rho);
    }
    if(smooth!=NULL) {
      if(checkArray(smooth)) return NULL;
      Py_XDECREF(kd->pNumpySmooth);
      kd->pNumpySmooth = smooth;
      Py_INCREF(smooth);
    }

    return PyCapsule_New(smx, NULL, NULL);
}

/*==========================================================================*/
/* nn_next                                                                 */
/*==========================================================================*/
PyObject *nn_next(PyObject *self, PyObject *args)
{
    long nCnt, i;

    KD kd;
    SMX smx;

    PyObject *kdobj, *smxobj;
    PyObject *nnList;
    PyObject *nnDist;
    PyObject *retList;

    PyArg_ParseTuple(args, "OO", &kdobj, &smxobj);
    kd  = PyCapsule_GetPointer(kdobj, NULL);
    smx = PyCapsule_GetPointer(smxobj, NULL);

    Py_BEGIN_ALLOW_THREADS

    nCnt = smSmoothStep(smx, NULL,0);

    Py_END_ALLOW_THREADS

    if (nCnt != 0)
    {
        nnList = PyList_New(nCnt); // Py_INCREF(nnList);
        nnDist = PyList_New(nCnt); // Py_INCREF(nnDist);
        retList = PyList_New(4);   Py_INCREF(retList);

        for (i=0; i < nCnt; i++)
        {
            PyList_SetItem(nnList, i, PyLong_FromLong(smx->pList[i]));
            PyList_SetItem(nnDist, i, PyFloat_FromDouble(smx->fList[i]));
        }

        PyList_SetItem(retList, 0, PyLong_FromLong(smx->pi));
        PyList_SetItem(retList, 1, PyFloat_FromDouble(smx->pfBall2[smx->pi]));
        PyList_SetItem(retList, 2, nnList);
        PyList_SetItem(retList, 3, nnDist);

        return retList;
    }

    return Py_None;
}

/*==========================================================================*/
/* nn_stop                                                                 */
/*==========================================================================*/
PyObject *nn_stop(PyObject *self, PyObject *args)
{
    KD kd;
    SMX smx;

    PyObject *kdobj, *smxobj;

    PyArg_ParseTuple(args, "OO", &kdobj, &smxobj);
    kd  = PyCapsule_GetPointer(kdobj,NULL);
    smx = PyCapsule_GetPointer(smxobj,NULL);

    smFinish(smx);

    return Py_None;
}

/*==========================================================================*/
/* nn_rewind                                                                */
/*==========================================================================*/
PyObject *nn_rewind(PyObject *self, PyObject *args)
{
    SMX smx;
    PyObject *smxobj;

    PyArg_ParseTuple(args, "O", &smxobj);
    smx = PyCapsule_GetPointer(smxobj, NULL);
    smSmoothInitStep(smx, 1);

    return PyCapsule_New(smx, NULL, NULL);
}

/*==========================================================================*/
/* populate                                                                 */
/*==========================================================================*/
int checkArray(PyObject *check) {

  if(check==NULL) {
    PyErr_SetString(PyExc_ValueError, "Unspecified array in kdtree");
    return 1;
  }

  PyArray_Descr *descr = PyArray_DESCR(check);
  if(descr==NULL || descr->kind!='f' || descr->elsize!=sizeof(double)) {
    PyErr_SetString(PyExc_TypeError, "Incorrect numpy data type to kdtree - must match C double");
    return 1;
  }
  return 0;

}

PyObject *populate(PyObject *self, PyObject *args)
{
    long i,nCnt;
    long procid;
    KD kd;
    SMX smx_global, smx_local;
    int propid, j;
    float ri[3];

    PyObject *kdobj, *smxobj;
    PyObject *dest; // Nx1 Numpy array for the property



    PyArg_ParseTuple(args, "OOii", &kdobj, &smxobj, &propid, &procid);
    kd  = PyCapsule_GetPointer(kdobj, NULL);
    smx_global = PyCapsule_GetPointer(smxobj, NULL);
    #define BIGFLOAT ((float)1.0e37)

    long nbodies = PyArray_DIM(kd->pNumpyPos, 0);

  /*
    if(n_particles>nbodies) {
      PyErr_SetString(PyExc_ValueError, "Trying to process more particles than are in simulation?");
      return NULL;
    }
    */



    if (checkArray(kd->pNumpySmooth)) return NULL;
    if(propid!=PROPID_HSM) {
      if (checkArray(kd->pNumpyDen)) return NULL;
      if (checkArray(kd->pNumpyMass)) return NULL;
      if (checkArray(kd->pNumpySmooth)) return NULL;
    }


    smx_local = smInitThreadLocalCopy(smx_global);
    smx_local->warnings=false;
    smx_local->pi = 0;

    int total_particles=0;

    switch(propid)
    {
        case PROPID_HSM:

          Py_BEGIN_ALLOW_THREADS
            for (i=0; i < nbodies; i++)
              {
                nCnt = smSmoothStep(smx_local, NULL, procid);
                if(nCnt==-1)
                  break; // nothing more to do
                total_particles+=1;
              }
          Py_END_ALLOW_THREADS

          break;

    case PROPID_RHO:

      i=smGetNext(smx_local);
      Py_BEGIN_ALLOW_THREADS
      while(i<nbodies)
        {
            for(int j=0; j<3; ++j) {
              ri[j] = GET2(kd->pNumpyPos,kd->p[i].iOrder,j);
            }
            // if(i<10) printf("i=%d pfB=%f ri=%f %f %f\n",i,smx_local->pfBall2[i],ri[0],ri[1],ri[2]);
            nCnt = smBallGather(smx_local,smx_local->pfBall2[i],ri);
            smDensity(smx_local, i, nCnt, smx_local->pList,smx_local->fList);
            i=smGetNext(smx_local);
        }
      Py_END_ALLOW_THREADS

/*
    case PROPID_MEANVEL:
      i=smGetNext(smx_local);
      Py_BEGIN_ALLOW_THREADS
      while(i<nbodies)
        {
          nCnt = smBallGather(smx_local,smx_local->pfBall2[i],smx_local->kd->p[i].r);
          smMeanVel(smx_local, i, nCnt, smx_local->pList,smx_local->fList);
          for (j=0;j<3;j++)
            SET2(kd->p[i].iOrder,j,kd->p[i].vMean[j]);
          i=smGetNext(smx_local);
        }
      Py_END_ALLOW_THREADS

      break;

    case PROPID_VELDISP:


      Py_BEGIN_ALLOW_THREADS
      for (i=0; i < nbodies; i++)
        {

          nCnt = smBallGather(smx_local,smx_local->pfBall2[i],smx_local->kd->p[i].r);
          smMeanVel(smx_local, i, nCnt, smx_local->pList,smx_local->fList);
          smDivv(smx_local, i, nCnt, smx_local->pList, smx_local->fList);
        }

      smReset(smx_local);

      for (i=0; i < nbodies; i++)
        {

          nCnt = smBallGather(smx_local,smx_local->pfBall2[i],smx_local->kd->p[i].r);
          smVelDispNBSym(smx_local, i, nCnt, smx_local->pList,smx_local->fList);
        }

      Py_END_ALLOW_THREADS



      for (i=0; i < nbodies; i++)
        {
          PyArray_SETITEM(dest, PyArray_GETPTR1(dest, kd->p[i].iOrder),
                          PyFloat_FromDouble(sqrt(kd->p[i].fVel2)));
        }

      break;

        default:
            break;
      */
    }

    smFinishThreadLocalCopy(smx_local);
    return Py_None;
}
