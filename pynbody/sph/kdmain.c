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
    PyObject *vel;  // Nx3 Numpy array of velocities
    PyObject *mass; // Nx1 Numpy array of masses

    if (!PyArg_ParseTuple(args, "OOOi", &pos, &vel, &mass, &nBucket))
        return NULL;

    KD kd = malloc(sizeof(*kd));
    kdInit(&kd, nBucket);

    int nbodies = PyArray_DIM(pos, 0);

    Py_BEGIN_ALLOW_THREADS

    kd->nParticles = nbodies;
    kd->nDark = kd->nParticles;
    kd->nGas = 0;
    kd->nStar = 0;
    kd->fTime = 0;
    kd->nActive = 0;
    kd->nActive += kd->nDark;
    kd->nActive += kd->nGas;
    kd->nActive += kd->nStar;
    kd->bDark = 1;
    kd->bGas = 0;
    kd->bStar = 0;
    /*
    ** Allocate particles.
    */
    kd->p = (PARTICLE *)malloc(kd->nActive*sizeof(PARTICLE));
    assert(kd->p != NULL);

    

    for (i=0; i < nbodies; i++)
    {
        kd->p[i].iOrder = i;
        kd->p[i].iMark = 1;
        kd->p[i].r[0] = (float)*((double *)PyArray_GETPTR2(pos, i, 0));
        kd->p[i].r[1] = (float)*((double *)PyArray_GETPTR2(pos, i, 1));
        kd->p[i].r[2] = (float)*((double *)PyArray_GETPTR2(pos, i, 2));
        kd->p[i].v[0] = (float)*((double *)PyArray_GETPTR2(vel, i, 0));
        kd->p[i].v[1] = (float)*((double *)PyArray_GETPTR2(vel, i, 1));
        kd->p[i].v[2] = (float)*((double *)PyArray_GETPTR2(vel, i, 2));
        kd->p[i].fMass = (float)*((double *)PyArray_GETPTR1(mass, i));
        kd->p[i].fDensity = 0;
        kd->p[i].fSmooth = 0;
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
    PyObject *smooth = NULL, *rho=NULL; 
    
    int nSmooth;
    long i;
    float hsm;

    PyArg_ParseTuple(args, "Oi|OO", &kdobj, &nSmooth, &smooth, &rho);
    kd = PyCapsule_GetPointer(kdobj, NULL);

#define BIGFLOAT ((float)1.0e37)

    float fPeriod[3] = {BIGFLOAT, BIGFLOAT, BIGFLOAT};

    smInit(&smx, kd, nSmooth, fPeriod);
    smSmoothInitStep(smx);

    if(smooth != NULL) {
      
      for (i=0;i<smx->kd->nActive;i++) {
        hsm = (float)*((double *)PyArray_GETPTR1(smooth, kd->p[i].iOrder));
        smx->pfBall2[i]=4.0*hsm*hsm;
      }
    }
    
    if(rho != NULL) 
      for (i=0;i<kd->nActive;i++) kd->p[i].fDensity = (float)*((double *)PyArray_GETPTR1(rho, kd->p[i].iOrder));
    
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

    nCnt = smSmoothStep(smx, NULL);

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
    smSmoothInitStep(smx);
    
    return PyCapsule_New(smx, NULL, NULL);
}

/*==========================================================================*/
/* populate                                                                 */
/*==========================================================================*/
PyObject *populate(PyObject *self, PyObject *args)
{
  long i,nCnt;

    KD kd;
    SMX smx;
    int propid, j;

    PyObject *kdobj, *smxobj;
    PyObject *dest; // Nx1 Numpy array for the property

    
  
    PyArg_ParseTuple(args, "OOOi", &kdobj, &smxobj, &dest, &propid);
    kd  = PyCapsule_GetPointer(kdobj, NULL);
    smx = PyCapsule_GetPointer(smxobj, NULL);
    
    smx->warnings=false;
    
    
    long nbodies = PyArray_DIM(dest, 0);
   
    PyArray_Descr *descr = PyArray_DESCR(dest);
   
#define SET(nn, val) *((double*)PyArray_GETPTR1(dest, nn)) = val

    if(descr->kind!='f' || descr->elsize!=sizeof(double)) {
      PyErr_SetString(PyExc_TypeError, "Incorrect numpy data type to kdtree - must match C double");
      return NULL;
    }

    switch(propid)
    {
        case PROPID_HSM:

          Py_BEGIN_ALLOW_THREADS
            for (i=0; i < nbodies; i++)
              {
                nCnt = smSmoothStep(smx, NULL);
                SET(kd->p[smx->pi].iOrder, kd->p[smx->pi].fSmooth);
                // *((double*)PyArray_GETPTR1(dest, kd->p[smx->pi].iOrder)) = kd->p[smx->pi].fSmooth;
              }
          Py_END_ALLOW_THREADS
                 
          break;
            
    case PROPID_RHO:
      
      Py_BEGIN_ALLOW_THREADS
      for (i=0; i < nbodies; i++)
        {          
            nCnt = smBallGather(smx,smx->pfBall2[i],smx->kd->p[i].r);
            smDensitySym(smx, i, nCnt, smx->pList,smx->fList);
            
        }
      Py_END_ALLOW_THREADS
      for(i=0;i<nbodies;i++)  SET(kd->p[i].iOrder, kd->p[i].fDensity);
      break;
      
    case PROPID_MEANVEL:
      Py_BEGIN_ALLOW_THREADS
      for (i=0; i < nbodies; i++)
        {
          
          nCnt = smBallGather(smx,smx->pfBall2[i],smx->kd->p[i].r);
          smMeanVelSym(smx, i, nCnt, smx->pList,smx->fList);
        } 
      Py_END_ALLOW_THREADS

      /* using a symmetric kernel, so need to complete the smMeanVelSym for all
         particles before outputting the values */

      for (i=0; i < nbodies; i++)
        {
          for (j=0;j<3;j++)
            PyArray_SETITEM(dest, PyArray_GETPTR2(dest, kd->p[i].iOrder,j), 
                            PyFloat_FromDouble(kd->p[i].vMean[j]));
        }

      break;

    case PROPID_VELDISP:

      /* when using a symmetric kernel, the dependencies (in this case mean velocity 
         and div_v have to be calculated completely before using for v_disp */      

      Py_BEGIN_ALLOW_THREADS
      for (i=0; i < nbodies; i++)
        {
          
          nCnt = smBallGather(smx,smx->pfBall2[i],smx->kd->p[i].r);
          

          smMeanVelSym(smx, i, nCnt, smx->pList,smx->fList);
          smDivvSym(smx, i, nCnt, smx->pList, smx->fList);
        }


      for (i=0; i < nbodies; i++)
        {

          nCnt = smBallGather(smx,smx->pfBall2[i],smx->kd->p[i].r);


          smVelDispNBSym(smx, i, nCnt, smx->pList,smx->fList);
        }
      
      Py_END_ALLOW_THREADS

      /* using a symmetric kernel, so need to complete the smVelDispNBSym for all
         particles before outputting the values */

      for (i=0; i < nbodies; i++)
        {
          PyArray_SETITEM(dest, PyArray_GETPTR1(dest, kd->p[i].iOrder), 
                          PyFloat_FromDouble(sqrt(kd->p[i].fVel2)));
        }

      break;

        default:
            break;
    }
    return Py_None;
}

