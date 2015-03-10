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
    fprintf(stderr, "c"allocating %ld bytes [already alloc"d: %ld].\n", sizeof(type) * (num), total_alloc), \
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

PyObject *domain_decomposition(PyObject *self, PyObject *args);
PyObject *set_arrayref(PyObject *self, PyObject *args);
PyObject *get_arrayref(PyObject *self, PyObject *args);
PyObject *has_threading(PyObject *self, PyObject *args);

template<typename T>
int checkArray(PyObject *check, const char *name);

int getBitDepth(PyObject *check);

/*==========================================================================*/
#define PROPID_HSM      1
#define PROPID_RHO      2
#define PROPID_QTYMEAN_1D    3
#define PROPID_QTYMEAN_ND    4
#define PROPID_QTYDISP_1D    5
#define PROPID_QTYDISP_ND    6
/*==========================================================================*/

static PyMethodDef kdmain_methods[] =
{
    {"init", kdinit, METH_VARARGS, "init"},
    {"free", kdfree, METH_VARARGS, "free"},

    {"nn_start",  nn_start,  METH_VARARGS, "nn_start"},
    {"nn_next",   nn_next,   METH_VARARGS, "nn_next"},
    {"nn_stop",   nn_stop,   METH_VARARGS, "nn_stop"},
    {"nn_rewind", nn_rewind, METH_VARARGS, "nn_rewind"},

    {"set_arrayref", set_arrayref, METH_VARARGS, "set_arrayref"},
    {"get_arrayref", get_arrayref, METH_VARARGS, "get_arrayref"},
    {"domain_decomposition", domain_decomposition, METH_VARARGS, "domain_decomposition"},

    {"populate",  populate,  METH_VARARGS, "populate"},

    {"has_threading",  has_threading,  METH_VARARGS, "populate"},

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

PyObject *has_threading(PyObject *self, PyObject *args)
{
#ifdef KDT_THREADING
    return Py_True;
#else
    return Py_False;
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

    int bitdepth = getBitDepth(pos);
    if(bitdepth==0) {
        PyErr_SetString(PyExc_ValueError, "Unsupported array dtype for kdtree");
        return NULL;
    }
    if(bitdepth!=getBitDepth(mass)) {
        PyErr_SetString(PyExc_ValueError, "pos and mass arrays must have matching dtypes for kdtree");
        return NULL;
    }

    if(bitdepth==64) {
        if(checkArray<double>(pos, "pos")) return NULL;
        if(checkArray<double>(mass, "mass")) return NULL;
    } else {
        if(checkArray<float>(pos, "pos")) return NULL;
        if(checkArray<float>(mass, "mass")) return NULL;
    }

    KD kd = (KD)malloc(sizeof(*kd));
    kdInit(&kd, nBucket);

    int nbodies = PyArray_DIM(pos, 0);

    kd->nParticles = nbodies;
    kd->nActive = nbodies;
    kd->nBitDepth = bitdepth;
    kd->pNumpyPos = pos;
    kd->pNumpyMass = mass;
    kd->pNumpySmooth = NULL;
    kd->pNumpyDen = NULL;
    kd->pNumpyQty = NULL;
    kd->pNumpyQtySmoothed = NULL;

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
    }

    if(bitdepth==64)
        kdBuildTree<double>(kd);
    else
        kdBuildTree<float>(kd);

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
    kd = (KD)PyCapsule_GetPointer(kdobj, NULL);

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

    PyArg_ParseTuple(args, "Oi", &kdobj, &nSmooth);
    kd = (KD)PyCapsule_GetPointer(kdobj, NULL);

#define BIGFLOAT ((float)1.0e37)

    float fPeriod[3] = {BIGFLOAT, BIGFLOAT, BIGFLOAT};

    if(nSmooth>PyArray_DIM(kd->pNumpyPos,0)) {
        PyErr_SetString(PyExc_ValueError, "Number of smoothing particles exceeds number of particles in tree");
        return NULL;
    }
    if(!smInit(&smx, kd, nSmooth, fPeriod)) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create smoothing context");
        return NULL;
    }

    smSmoothInitStep(smx, nProcs);



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
    kd  = (KD)PyCapsule_GetPointer(kdobj, NULL);
    smx = (SMX)PyCapsule_GetPointer(smxobj, NULL);

    Py_BEGIN_ALLOW_THREADS

    if(kd->nBitDepth==32)
        nCnt = smSmoothStep<float>(smx,0);
    else
        nCnt = smSmoothStep<double>(smx,0);

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
        if(kd->nBitDepth==32)
            PyList_SetItem(retList, 1, PyFloat_FromDouble(
                        (double)GET<float>(smx->kd->pNumpySmooth, smx->kd->p[smx->pi].iOrder)));
        else
            PyList_SetItem(retList, 1, PyFloat_FromDouble(
                        GET<double>(smx->kd->pNumpySmooth, smx->kd->p[smx->pi].iOrder)));

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
    kd  = (KD)PyCapsule_GetPointer(kdobj,NULL);
    smx = (SMX)PyCapsule_GetPointer(smxobj,NULL);

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
    smx = (SMX)PyCapsule_GetPointer(smxobj, NULL);
    smSmoothInitStep(smx, 1);

    return PyCapsule_New(smx, NULL, NULL);
}


int getBitDepth(PyObject *check) {

  if(check==NULL) {
    return 0;
  }

  PyArray_Descr *descr = PyArray_DESCR(check);
  if(descr!=NULL && descr->kind=='f' && descr->elsize==sizeof(float))
      return 32;
  else if(descr!=NULL && descr->kind=='f' && descr->elsize==sizeof(double))
      return 64;
  else
      return 0;

}

template<typename T>
const char* c_name() {
    return "unknown";
}

template<>
const char* c_name<double>() {
    return "double";
}

template<>
const char* c_name<float>() {
    return "float";
}

template<typename T>
int checkArray(PyObject *check, const char* name) {

  if(check==NULL) {
    PyErr_SetString(PyExc_ValueError, "Unspecified array in kdtree");
    return 1;
  }

  PyArray_Descr *descr = PyArray_DESCR(check);
  if(descr==NULL || descr->kind!='f' || descr->elsize!=sizeof(T)) {
    PyErr_Format(PyExc_TypeError, "Incorrect numpy data type for %s passed to kdtree - must match C %s",name,c_name<T>());
    return 1;
  }
  return 0;

}



PyObject *set_arrayref(PyObject *self, PyObject *args) {
    int arid;
    PyObject *kdobj, *arobj, **existing;
    KD kd;

    const char *name0="smooth";
    const char *name1="rho";
    const char *name2="mass";
    const char *name3="qty";
    const char *name4="qty_sm";

    const char *name;

    PyArg_ParseTuple(args, "OiO", &kdobj, &arid, &arobj);
    kd  = (KD)PyCapsule_GetPointer(kdobj, NULL);
    if(!kd) return NULL;



    switch(arid) {
    case 0:
        existing = &(kd->pNumpySmooth);
        name = name0;
        break;
    case 1:
        existing = &(kd->pNumpyDen);
        name = name1;
        break;
    case 2:
        existing = &(kd->pNumpyMass);
        name = name2;
        break;
    case 3:
        existing = &(kd->pNumpyQty);
        name = name3;
        break;
    case 4:
        existing = &(kd->pNumpyQtySmoothed);
        name = name4;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Unknown array to set for KD tree");
        return NULL;
    }

    int bitdepth=0;
    if(arid<=2)
        bitdepth=kd->nBitDepth;
    else if(arid==3 || arid==4)
        bitdepth=getBitDepth(arobj);

    if(bitdepth==32) {
        if(checkArray<float>(arobj,name)) return NULL;
    } else if(bitdepth==64) {
        if(checkArray<double>(arobj,name)) return NULL;
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported array dtype for kdtree");
        return NULL;
    }

    Py_XDECREF(*existing);
    (*existing) = arobj;
    Py_INCREF(arobj);
    return Py_None;
}

PyObject *get_arrayref(PyObject *self, PyObject *args) {
    int arid;
    PyObject *kdobj, *arobj, **existing;
    KD kd;

    PyArg_ParseTuple(args, "Oi", &kdobj, &arid);
    kd  = (KD)PyCapsule_GetPointer(kdobj, NULL);
    if(!kd) return NULL;

    switch(arid) {
    case 0:
        existing = &(kd->pNumpySmooth);
        break;
    case 1:
        existing = &(kd->pNumpyDen);
        break;
    case 2:
        existing = &(kd->pNumpyMass);
        break;
    case 3:
        existing = &(kd->pNumpyQty);
        break;
    case 4:
        existing = &(kd->pNumpyQtySmoothed);
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Unknown array to get from KD tree");
        return NULL;
    }

    Py_INCREF(*existing);

    if(*existing==NULL)
        return Py_None;
    else
        return (*existing);

}

PyObject *domain_decomposition(PyObject *self, PyObject *args) {
    int nproc;
    PyObject *smxobj;
    KD kd;

    PyArg_ParseTuple(args, "Oi", &smxobj, &nproc);

    kd  = (KD)PyCapsule_GetPointer(smxobj, NULL);
    if(!kd) return NULL;

    if(kd->nBitDepth==32) {
        if(checkArray<float>(kd->pNumpySmooth, "smooth")) return NULL;
    } else {
        if(checkArray<double>(kd->pNumpySmooth, "smooth")) return NULL;
    }

    if(nproc<0) {
        PyErr_SetString(PyExc_ValueError, "Invalid number of processors");
        return NULL;
    }

    if(kd->nBitDepth==32)
        smDomainDecomposition<float>(kd,nproc);
    else
        smDomainDecomposition<double>(kd,nproc);

    return Py_None;
}

template<typename Tf, typename Tq>
PyObject *typed_populate(PyObject *self, PyObject *args)
{

    long i,nCnt;
    long procid;
    KD kd;
    SMX smx_global, smx_local;
    int propid, j;
    float ri[3];
    float hsm;

    void (*pSmFn)(SMX ,int ,int ,int *,float *)=NULL;

    PyObject *kdobj, *smxobj;
    PyObject *dest; // Nx1 Numpy array for the property



    PyArg_ParseTuple(args, "OOii", &kdobj, &smxobj, &propid, &procid);
    kd  = (KD)PyCapsule_GetPointer(kdobj, NULL);
    smx_global = (SMX)PyCapsule_GetPointer(smxobj, NULL);
    #define BIGFLOAT ((float)1.0e37)

    long nbodies = PyArray_DIM(kd->pNumpyPos, 0);


    if (checkArray<Tf>(kd->pNumpySmooth,"smooth")) return NULL;
    if(propid>PROPID_HSM) {
      if (checkArray<Tf>(kd->pNumpyDen,"rho")) return NULL;
      if (checkArray<Tf>(kd->pNumpyMass,"mass")) return NULL;
    }
    if(propid>PROPID_RHO) {
        if (checkArray<Tq>(kd->pNumpyQty,"qty")) return NULL;
        if (checkArray<Tq>(kd->pNumpyQtySmoothed,"qty_sm")) return NULL;
    }

#ifdef KDT_THREADING
    smx_local = smInitThreadLocalCopy(smx_global);
    smx_local->warnings=false;
    smx_local->pi = 0;
#else
    smx_global = smx_local;
#endif

    smx_global->warnings=false;

    int total_particles=0;


    switch(propid)
    {
        case PROPID_RHO:
            pSmFn = &smDensity<Tf>;
            break;
        case PROPID_QTYMEAN_ND:
            pSmFn = &smMeanQtyND<Tf,Tq>;
            break;
        case PROPID_QTYDISP_ND:
            pSmFn = &smDispQtyND<Tf,Tq>;
            break;
        case PROPID_QTYMEAN_1D:
            pSmFn = &smMeanQty1D<Tf,Tq>;
            break;
        case PROPID_QTYDISP_1D:
            pSmFn = &smDispQty1D<Tf,Tq>;
            break;
    }


    if(propid==PROPID_HSM)
    {
          Py_BEGIN_ALLOW_THREADS
            for (i=0; i < nbodies; i++)
              {
                nCnt = smSmoothStep<Tf>(smx_local, procid);
                if(nCnt==-1)
                  break; // nothing more to do
                total_particles+=1;
              }
          Py_END_ALLOW_THREADS

    } else {

      i=smGetNext(smx_local);

      Py_BEGIN_ALLOW_THREADS
      while(i<nbodies)
        {
            // make a copy of the position of this particle
            for(int j=0; j<3; ++j) {
              ri[j] = GET2<Tf>(kd->pNumpyPos,kd->p[i].iOrder,j);
            }

            // retrieve the existing smoothing length
            hsm = GETSMOOTH(Tf,i);

            // use it to get nearest neighbours
            nCnt = smBallGather<Tf>(smx_local,4*hsm*hsm,ri);

            // calculate the density
            (*pSmFn)(smx_local, i, nCnt, smx_local->pList,smx_local->fList);

            // select next particle in coordination with other threads
            i=smGetNext(smx_local);

            if(smx_global->warnings)
                break;
        }
      Py_END_ALLOW_THREADS
  }


  if(smx_local->warnings) {
#ifdef KDT_THREADING
    smFinishThreadLocalCopy(smx_local);
#endif
    PyErr_SetString(PyExc_RuntimeError,"Buffer overflow in smoothing operation. This probably means that your smoothing lengths are too large compared to the number of neighbours you specified.");
    return NULL;
  } else {
#ifdef KDT_THREADING
    smFinishThreadLocalCopy(smx_local);
#endif
    return Py_None;
  }

}

PyObject *populate(PyObject *self, PyObject *args)
{
    // this is really a shell function that works out what
    // template parameters to adopt

    KD kd;
    PyObject *kdobj, *smxobj;
    int propid, procid, nF, nQ;

    PyArg_ParseTuple(args, "OOii", &kdobj, &smxobj, &propid, &procid);
    kd  = (KD)PyCapsule_GetPointer(kdobj, NULL);


    nF = kd->nBitDepth;
    nQ = 32;

    if(kd->pNumpyQty!=NULL) {
        nQ=getBitDepth(kd->pNumpyQty);
    }

    if(nF==64 && nQ==64)
        return typed_populate<double,double>(self,args);
    else if(nF==64 && nQ==32)
        return typed_populate<double,float>(self,args);
    else if(nF==32 && nQ==32)
        return typed_populate<float,float>(self,args);
    else if(nF==32 && nQ==64)
        return typed_populate<float,double>(self,args);
    else {
        PyErr_SetString(PyExc_ValueError, "Unsupported array dtypes for kdtree");
        return NULL;
    }
}
