#include "kd.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#include "kd.h"
#include "smooth.h"

// For Numpy < 2.0, if build isolation does not work 
// To be tested!
#if NPY_ABI_VERSION < 0x02000000
  #define PyDataType_ELSIZE(descr) ((descr)->elsize)
#endif

/*==========================================================================*/
/* Prototypes.                                                              */
/*==========================================================================*/

PyObject *kdinit(PyObject *self, PyObject *args);
PyObject *kdfree(PyObject *self, PyObject *args);
PyObject *kdbuild(PyObject *self, PyObject *args);
PyObject *kdimport_prebuilt(PyObject *self, PyObject *args);

PyObject *nn_start(PyObject *self, PyObject *args);
PyObject *nn_next(PyObject *self, PyObject *args);
PyObject *nn_stop(PyObject *self, PyObject *args);
PyObject *nn_rewind(PyObject *self, PyObject *args);

PyObject *populate(PyObject *self, PyObject *args);

PyObject *domain_decomposition(PyObject *self, PyObject *args);
PyObject *set_arrayref(PyObject *self, PyObject *args);
PyObject *get_arrayref(PyObject *self, PyObject *args);
PyObject *get_node_count(PyObject *self, PyObject *args);

PyObject *particles_in_sphere(PyObject *self, PyObject *args);

int getBitDepth(PyObject *check);

/*==========================================================================*/
#define PROPID_HSM 1
#define PROPID_RHO 2
#define PROPID_QTYMEAN_1D 3
#define PROPID_QTYMEAN_ND 4
#define PROPID_QTYDISP_1D 5
#define PROPID_QTYDISP_ND 6
#define PROPID_QTYDIV 7
#define PROPID_QTYCURL 8
/*==========================================================================*/

static PyMethodDef kdmain_methods[] = {
    {"init", kdinit, METH_VARARGS, "init"},
    {"free", kdfree, METH_VARARGS, "free"},
    {"build", kdbuild, METH_VARARGS, "build"},
    {"import_prebuilt", kdimport_prebuilt, METH_VARARGS, "import_prebuilt"},

    {"nn_start", nn_start, METH_VARARGS, "nn_start"},
    {"nn_next", nn_next, METH_VARARGS, "nn_next"},
    {"nn_stop", nn_stop, METH_VARARGS, "nn_stop"},
    {"nn_rewind", nn_rewind, METH_VARARGS, "nn_rewind"},

    {"particles_in_sphere", particles_in_sphere, METH_VARARGS,
     "particles_in_sphere"},

    {"set_arrayref", set_arrayref, METH_VARARGS, "set_arrayref"},
    {"get_arrayref", get_arrayref, METH_VARARGS, "get_arrayref"},
    {"get_node_count", get_node_count, METH_VARARGS, "get_node_count"},
    {"domain_decomposition", domain_decomposition, METH_VARARGS,
     "domain_decomposition"},

    {"populate", populate, METH_VARARGS, "populate"},

    {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef ourdef = {PyModuleDef_HEAD_INIT,
                                    "kdmain",
                                    "KDTree module for pynbody",
                                    -1,
                                    kdmain_methods,
                                    NULL,
                                    NULL,
                                    NULL,
                                    NULL};
#endif

PyMODINIT_FUNC
PyInit_kdmain(void)
{
  import_array();
  return PyModule_Create(&ourdef);
}

/* Array checking utility function */

template <typename T> const char *c_name() { return "unknown"; }

template <> const char *c_name<double>() { return "double"; }

template <> const char *c_name<float>() { return "float"; }

template <> const char *c_name<KDNode>() { return "KDNode"; }

template <> const char *c_name<npy_intp>() { return "npy_intp"; }


template <typename T> const char np_kind() { return '?'; }

template <> const char np_kind<double>() { return 'f'; }

template <> const char np_kind<float>() { return 'f'; }

template <> const char np_kind<KDNode>() { return 'V'; }

template <> const char np_kind<npy_intp>() { return 'i'; }

template <typename T> const char py_kind() { return '?'; }

template <> const char py_kind<double>() { return 'd'; }

template <> const char py_kind<float>() { return 'f'; }


template <typename T> int checkArray(PyObject *check, const char *name, npy_intp size=0, bool require_c_contiguous=false) {
  /* Checks that the passed object is a numpy array of the correct type, with the correct size (if specified), and is C-contiguous (if required)
  Returns 0 if the check passes, 1 if it fails (in which case an exception will have been set).
  */

  if (check == nullptr) {
    PyErr_Format(PyExc_ValueError, "An array must be passed for the '%s' argument", name);
    return 1;
  }

  if(!PyArray_Check(check)) {
    PyErr_Format(PyExc_ValueError, "An array must be passed for the '%s' argument", name);
    return 1;
  }

  PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *) check);

  if (descr == NULL || descr->kind != np_kind<T>() ||PyDataType_ELSIZE(descr) != sizeof(T)) {
    PyErr_Format(
        PyExc_TypeError,
        "Incorrect numpy data type for %s passed to kdtree - must match C %s",
        name, c_name<T>());
    return 1;
  }

  if(size > 0 && PyArray_DIM((PyArrayObject *) check, 0) != size) {
    PyErr_Format(PyExc_ValueError, "Array '%s' has the wrong size", name);
    return 1;
  }

  if(require_c_contiguous && (PyArray_FLAGS((PyArrayObject *) check) & NPY_ARRAY_C_CONTIGUOUS) == 0) {
    PyErr_Format(PyExc_ValueError, "Array '%s' must be C-contiguous", name);
    return 1;
  }

  npy_intp expected_bytes = PyArray_SIZE((PyArrayObject *)check) * sizeof(T);
  npy_intp actual_bytes = PyArray_NBYTES((PyArrayObject *)check);
  if (actual_bytes != expected_bytes) {
    PyErr_Format(PyExc_ValueError,
                 "Array '%s' has %zd bytes, but %zd bytes are required for type %s",
                 name, static_cast<Py_ssize_t>(actual_bytes),
                 static_cast<Py_ssize_t>(expected_bytes), c_name<T>());
    return 1;
  }
  
  return 0;
}



/*==========================================================================*/
/* kdinit                                                                   */
/*==========================================================================*/
PyObject *kdinit(PyObject *self, PyObject *args) {
  npy_intp nBucket;
  npy_intp i;

  PyObject *pos;  // Nx3 Numpy array of positions
  PyObject *mass; // Nx1 Numpy array of masses

  if (!PyArg_ParseTuple(args, "OOn", &pos, &mass, &nBucket))
    return NULL;

  int bitdepth = getBitDepth(pos);
  if (bitdepth == 0) {
    PyErr_SetString(PyExc_ValueError, "Unsupported array dtype for kdtree");
    return NULL;
  }
  if (bitdepth != getBitDepth(mass)) {
    PyErr_SetString(PyExc_ValueError,
                    "pos and mass arrays must have matching dtypes for kdtree");
    return NULL;
  }

  if (bitdepth == 64) {
    if (checkArray<double>(pos, "pos"))
      return NULL;
    if (checkArray<double>(mass, "mass"))
      return NULL;
  } else {
    if (checkArray<float>(pos, "pos"))
      return NULL;
    if (checkArray<float>(mass, "mass"))
      return NULL;
  }

  KDContext *kd = new KDContext();
  kd->nBucket = nBucket;

  npy_intp nbodies = PyArray_DIM((PyArrayObject *) pos, 0);

  kd->nParticles = nbodies;
  kd->nActive = nbodies;
  kd->nBitDepth = bitdepth;
  kd->pNumpyPos = (PyArrayObject *) pos;
  kd->pNumpyMass = (PyArrayObject *) mass;

  Py_INCREF(pos);
  Py_INCREF(mass);


  kdCountNodes(kd);

  return PyCapsule_New((void *)kd, NULL, NULL);
}

PyObject * get_node_count(PyObject *self, PyObject *args) {
  PyObject *kdobj;
  if(!PyArg_ParseTuple(args, "O", &kdobj))
    return NULL;
  KDContext *kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));
  return PyLong_FromSsize_t(kd->nNodes);
}

PyObject * build_or_import(PyObject *self, PyObject *args, bool import_mode) {
  PyObject *kdNodeArray; // Length-N_node Numpy array (uninitialized) for KDNodes
  PyObject *orderArray;  // Length-N Numpy array (uninitialized) for particle ordering map
  PyObject *kdobj;
  int num_threads;


  if (!PyArg_ParseTuple(args, "OOOi", &kdobj, &kdNodeArray, &orderArray, &num_threads))
    return nullptr;


  KDContext *kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));
  if (kd == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Invalid KDContext object");
    return nullptr;
  }

  if (checkArray<KDNode>(kdNodeArray, "kdNodes", kd->nNodes, true)) {
    return nullptr;
  }

  if (checkArray<npy_intp>(orderArray, "orderArray", kd->nParticles, true)) {
    return nullptr;
  }

  kd->kdNodes = static_cast<KDNode*>(PyArray_DATA((PyArrayObject *) kdNodeArray));
  kd->kdNodesPyArrayObject = (PyArrayObject *) kdNodeArray;

  kd->particleOffsets = static_cast<npy_intp*>(PyArray_DATA((PyArrayObject *) orderArray));
  kd->pNumpyParticleOffsets = (PyArrayObject *) orderArray;

  Py_INCREF(kd->kdNodesPyArrayObject);
  Py_INCREF(kd->pNumpyParticleOffsets);


  if(!import_mode) {
    Py_BEGIN_ALLOW_THREADS;
    for (npy_intp i = 0; i < kd->nParticles; i++) {
      kd->particleOffsets[i] = i;
    }

    if (kd->nBitDepth == 64)
      kdBuildTree<double>(kd, num_threads);
    else
      kdBuildTree<float>(kd, num_threads);
    Py_END_ALLOW_THREADS;
  }


  Py_INCREF(Py_None);
  return Py_None;

}

PyObject * kdimport_prebuilt(PyObject *self, PyObject *args) {
  return build_or_import(self, args, true);
}

PyObject * kdbuild(PyObject *self, PyObject *args) {
  return build_or_import(self, args, false);
}

/*==========================================================================*/
/* kdfree                                                                   */
/*==========================================================================*/
PyObject *kdfree(PyObject *self, PyObject *args) {
  KDContext *kd;
  PyObject *kdobj;

  PyArg_ParseTuple(args, "O", &kdobj);
  kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));

  if (kd == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Invalid KDContext object");
    return nullptr;
  }

  Py_XDECREF(kd->pNumpyPos);
  Py_XDECREF(kd->pNumpyMass);
  Py_XDECREF(kd->pNumpySmooth);
  Py_XDECREF(kd->pNumpyDen);
  Py_XDECREF(kd->kdNodesPyArrayObject);
  Py_XDECREF(kd->pNumpyParticleOffsets);

  delete kd;

  Py_INCREF(Py_None);
  return Py_None;
}


/*==========================================================================*/
/* nn_start                                                                 */
/*==========================================================================*/
template<typename T> struct typed_nn_start {
  static PyObject *call(PyObject *self, PyObject *args) {
    KDContext* kd;
    SmoothingContext<T> * smx;

    PyObject *kdobj;
    /* Nx1 Numpy arrays for smoothing length and density for calls that pass
      in those values from existing arrays
    */

    int nSmooth;
    double period;

    PyArg_ParseTuple(args, "Oi|d", &kdobj, &nSmooth, &period);
    kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));

    if (period <= 0)
      period = std::numeric_limits<double>::max();



    double fPeriod[3] = {period, period, period};

    if (nSmooth > PyArray_DIM(kd->pNumpyPos, 0)) {
      PyErr_SetString(
          PyExc_ValueError,
          "Number of smoothing particles exceeds number of particles in tree");
      return NULL;
    }

    smCheckPeriodicityAndWarn(kd, fPeriod);

    smx = smInit<T>(kd, nSmooth, period);
    if (smx == nullptr) return nullptr; // smInit sets the error message
    smSmoothInitStep(smx);
    return PyCapsule_New(smx, NULL, NULL);

  }
};




/*==========================================================================*/
/* nn_next                                                                 */
/*==========================================================================*/
template<typename T> struct typed_nn_next {
  static PyObject *call(PyObject *self, PyObject *args) {
    npy_intp nCnt, i, pj;

    KDContext* kd;
    SmoothingContext<T> * smx;

    PyObject *kdobj, *smxobj;
    PyObject *nnList;
    PyObject *nnDist;
    PyObject *retList;

    PyArg_ParseTuple(args, "OO", &kdobj, &smxobj);
    kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));
    smx = static_cast<SmoothingContext<T>*>(PyCapsule_GetPointer(smxobj, NULL));

    if(smx==nullptr) {
      PyErr_SetString(PyExc_ValueError, "Invalid smoothing context object");
      return nullptr;
    }

    Py_BEGIN_ALLOW_THREADS;

    nCnt = smSmoothStep<T>(smx, 0);

    Py_END_ALLOW_THREADS;

    if (nCnt > 0) {
      nnList = PyList_New(nCnt); // Py_INCREF(nnList);
      nnDist = PyList_New(nCnt); // Py_INCREF(nnDist);
      retList = PyList_New(4);
      Py_INCREF(retList);

      for (i = 0; i < nCnt; i++) {
        pj = smx->pList[i];
        PyList_SetItem(nnList, i, PyLong_FromSsize_t(smx->kd->particleOffsets[pj]));
        PyList_SetItem(nnDist, i, PyFloat_FromDouble(smx->fList[i]));
      }

      PyList_SetItem(retList, 0, PyLong_FromSsize_t(smx->kd->particleOffsets[smx->pi]));
      PyList_SetItem(retList, 1, PyFloat_FromDouble(GETSMOOTH(T, smx->pi)));
      PyList_SetItem(retList, 2, nnList);
      PyList_SetItem(retList, 3, nnDist);

      return retList;
    }

    Py_INCREF(Py_None);
    return Py_None;
  }
};




/*==========================================================================*/
/* nn_stop                                                                 */
/*==========================================================================*/
template<typename T> struct typed_nn_stop {
  static PyObject *call(PyObject *self, PyObject *args) {
    KDContext* kd;
    SmoothingContext<T> * smx;

    PyObject *kdobj, *smxobj;

    PyArg_ParseTuple(args, "OO", &kdobj, &smxobj);
    kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));
    smx = static_cast<SmoothingContext<T>*>(PyCapsule_GetPointer(smxobj, NULL));
    if(smx==nullptr) {
      PyErr_SetString(PyExc_ValueError, "Invalid smoothing context object");
      return nullptr;
    }
    delete smx;

    Py_INCREF(Py_None);
    return Py_None;
  }
};




/*==========================================================================*/
/* nn_rewind                                                                */
/*==========================================================================*/
template<typename T> struct typed_nn_rewind {
  static PyObject *call(PyObject *self, PyObject *args) {
    SmoothingContext<T> * smx;
    PyObject *smxobj;

    PyArg_ParseTuple(args, "O", &smxobj);
    smx = static_cast<SmoothingContext<T> * >(PyCapsule_GetPointer(smxobj, nullptr));
    if(smx==nullptr) {
      PyErr_SetString(PyExc_ValueError, "Invalid smoothing context object");
      return nullptr;
    }
    smSmoothInitStep(smx);

    return PyCapsule_New(smx, NULL, NULL);
  }
};



int getBitDepth(PyObject *check) {

  if (check == NULL) {
    return 0;
  }

  PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *) check);
  if (descr != NULL && descr->kind == 'f' && PyDataType_ELSIZE(descr) == sizeof(float))
    return 32;
  else if (descr != NULL && descr->kind == 'f' &&
           PyDataType_ELSIZE(descr) == sizeof(double))
    return 64;
  else
    return 0;
}


PyObject *set_arrayref(PyObject *self, PyObject *args) {
  int arid;
  PyObject *kdobj, *arobj;
  PyArrayObject **existing;
  KDContext* kd;

  const char *name0 = "smooth";
  const char *name1 = "rho";
  const char *name2 = "mass";
  const char *name3 = "qty";
  const char *name4 = "qty_sm";

  const char *name;

  PyArg_ParseTuple(args, "OiO", &kdobj, &arid, &arobj);
  kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));
  if (!kd)
    return NULL;

  switch (arid) {
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

  int bitdepth = 0;
  if (arid <= 2)
    bitdepth = kd->nBitDepth;
  else if (arid == 3 || arid == 4)
    bitdepth = getBitDepth(arobj);

  if (bitdepth == 32) {
    if (checkArray<float>(arobj, name))
      return NULL;
  } else if (bitdepth == 64) {
    if (checkArray<double>(arobj, name))
      return NULL;
  } else {
    PyErr_SetString(PyExc_ValueError, "Unsupported array dtype for kdtree");
    return NULL;
  }

  Py_XDECREF(*existing);
  (*existing) = (PyArrayObject *) arobj;
  Py_INCREF(arobj);

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject *get_arrayref(PyObject *self, PyObject *args) {
  int arid;
  PyObject *kdobj, *arobj;
  PyArrayObject **existing;
  KDContext* kd;

  PyArg_ParseTuple(args, "Oi", &kdobj, &arid);
  kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));
  if (!kd)
    return NULL;

  switch (arid) {
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

  if (*existing == NULL) {
    Py_INCREF(Py_None);
    return Py_None;
  } else
    return ((PyObject *) *existing);
}

PyObject *domain_decomposition(PyObject *self, PyObject *args) {
  int nproc;
  PyObject *smxobj;
  KDContext* kd;

  PyArg_ParseTuple(args, "Oi", &smxobj, &nproc);

  kd = static_cast<KDContext*>(PyCapsule_GetPointer(smxobj, NULL));
  if (!kd)
    return NULL;

  if (kd->nBitDepth == 32) {
    if (checkArray<float>((PyObject *) kd->pNumpySmooth, "smooth"))
      return NULL;
  } else {
    if (checkArray<double>((PyObject *) kd->pNumpySmooth, "smooth"))
      return NULL;
  }

  if (nproc < 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid number of processors");
    return NULL;
  }

  if (kd->nBitDepth == 32)
    smDomainDecomposition<float>(kd, nproc);
  else
    smDomainDecomposition<double>(kd, nproc);

  Py_INCREF(Py_None);
  return Py_None;
}

template <typename Tf, typename Tq> struct typed_particles_in_sphere {
  static PyObject *call(PyObject *self, PyObject *args) {
    SmoothingContext<Tf> * smx;
    KDContext* kd;
    Tf r;
    Tf ri[3];

    std::string pytype_s = "OO" + std::string(4, py_kind<Tf>());
    const char* pytype = pytype_s.c_str();

    PyObject *kdobj = nullptr, *smxobj = nullptr;


    PyArg_ParseTuple(args, pytype, &kdobj, &smxobj, &ri[0], &ri[1], &ri[2],
                     &r);

    kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));
    smx = (SmoothingContext<Tf> *)PyCapsule_GetPointer(smxobj, NULL);

    initParticleList(smx);
    smBallGather<Tf, smBallGatherStoreResultInList>(smx, r * r, ri);

    return getReturnParticleList(smx);
  }
};

template <typename Tf, typename Tq> struct typed_populate {
  static PyObject *call(PyObject *self, PyObject *args) {

    npy_intp i, nCnt;
    int procid;
    KDContext* kd;
    SmoothingContext<Tf> *smx_global, *smx_local;
    int propid;
    Tf ri[3];
    Tf hsm;
    int kernel_id;

    void (*pSmFn)(SmoothingContext<Tf> *, npy_intp, int) = NULL;

    PyObject *kdobj, *smxobj;

    PyArg_ParseTuple(args, "OOiii", &kdobj, &smxobj, &propid, &procid,
                     &kernel_id);
    kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, NULL));
    smx_global = (SmoothingContext<Tf> *)PyCapsule_GetPointer(smxobj, NULL);

    smx_global->setupKernel(kernel_id);

    npy_intp nbodies = PyArray_DIM(kd->pNumpyPos, 0);

    if (checkArray<Tf>((PyObject *) kd->pNumpySmooth, "smooth"))
      return NULL;

    if (propid > PROPID_HSM) {
      if (checkArray<Tf>((PyObject *) kd->pNumpyDen, "rho"))
        return NULL;
      if (checkArray<Tf>((PyObject *) kd->pNumpyMass, "mass"))
        return NULL;
    }
    if (propid > PROPID_RHO) {
      if (checkArray<Tq>((PyObject *) kd->pNumpyQty, "qty"))
        return NULL;
      if (checkArray<Tq>((PyObject *) kd->pNumpyQtySmoothed, "qty_sm"))
        return NULL;
    }

    smx_local = smInitThreadLocalCopy(smx_global);
    smx_local->warnings = false;
    smx_local->pi = 0;

    smx_global->warnings = false;

    npy_intp total_particles = 0;

    switch (propid) {
    case PROPID_RHO:
      pSmFn = &smDensity<Tf>;
      break;
    case PROPID_QTYMEAN_ND:
      pSmFn = &smMeanQtyND<Tf, Tq>;
      break;
    case PROPID_QTYDISP_ND:
      pSmFn = &smDispQtyND<Tf, Tq>;
      break;
    case PROPID_QTYMEAN_1D:
      pSmFn = &smMeanQty1D<Tf, Tq>;
      break;
    case PROPID_QTYDISP_1D:
      pSmFn = &smDispQty1D<Tf, Tq>;
      break;
    case PROPID_QTYDIV:
      pSmFn = &smDivQty<Tf, Tq>;
      break;
    case PROPID_QTYCURL:
      pSmFn = &smCurlQty<Tf, Tq>;
      break;
    }

    if (propid == PROPID_HSM) {
      Py_BEGIN_ALLOW_THREADS;
      for (i = 0; i < nbodies; i++) {
        nCnt = smSmoothStep<Tf>(smx_local, procid);
        if (nCnt == -1)
          break; // nothing more to do
        total_particles += 1;
      }
      Py_END_ALLOW_THREADS;

    } else {

      i = smGetNext(smx_local);

      Py_BEGIN_ALLOW_THREADS;
      while (i < nbodies) {
        // make a copy of the position of this particle
        for (int j = 0; j < 3; ++j) {
          ri[j] = GET2<Tf>(kd->pNumpyPos, kd->particleOffsets[i], j);
        }

        // retrieve the existing smoothing length
        hsm = GETSMOOTH(Tf, i);

        // use it to get nearest neighbours
        nCnt = smBallGather<Tf, smBallGatherStoreResultInSmx>(smx_local, 4 * hsm * hsm, ri);

        // calculate the density
        (*pSmFn)(smx_local, i, nCnt);

        // select next particle in coordination with other threads
        i = smGetNext(smx_local);

        if (smx_global->warnings)
          break;
      }
      Py_END_ALLOW_THREADS;

    }

    smFinishThreadLocalCopy(smx_local);
    if (smx_local->warnings) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Buffer overflow in smoothing operation. This probably "
                      "means that your smoothing lengths are too large "
                      "compared to the number of neighbours you specified.");
      return NULL;
    } else {
      Py_INCREF(Py_None);
      return Py_None;
    }
  }
};

template <template <typename, typename> class func>
PyObject *type_dispatcher_2(PyObject *self, PyObject *args) {
  PyObject *kdobj = PyTuple_GetItem(args, 0);
  if (kdobj == nullptr) {
    PyErr_SetString(PyExc_ValueError, "First argument must be a kdtree object");
    return nullptr;
  }
  KDContext* kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, nullptr));
  int nF = kd->nBitDepth;
  int nQ = 32;

  if (kd->pNumpyQty != NULL) {
    nQ = getBitDepth((PyObject *) kd->pNumpyQty);
  }

  if (nF == 64 && nQ == 64)
    return func<double, double>::call(self, args);
  else if (nF == 64 && nQ == 32)
    return func<double, float>::call(self, args);
  else if (nF == 32 && nQ == 32)
    return func<float, float>::call(self, args);
  else if (nF == 32 && nQ == 64)
    return func<float, double>::call(self, args);
  else {
    PyErr_SetString(PyExc_ValueError, "Unsupported dtype combination");
    return nullptr;
  }
}

template <template <typename> class func>
PyObject *type_dispatcher_1(PyObject *self, PyObject *args) {
  PyObject *kdobj = PyTuple_GetItem(args, 0);
  if (kdobj == nullptr) {
    PyErr_SetString(PyExc_ValueError, "First argument must be a kdtree object");
    return nullptr;
  }
  KDContext* kd = static_cast<KDContext*>(PyCapsule_GetPointer(kdobj, nullptr));
  int nF = kd->nBitDepth;

  if (nF == 64)
    return func<double>::call(self, args);
  else if (nF == 32)
    return func<float>::call(self, args);
  else {
    PyErr_SetString(PyExc_ValueError, "Unsupported dtype combination");
    return nullptr;
  }
}

PyObject *nn_start(PyObject *self, PyObject *args) {
  return type_dispatcher_1<typed_nn_start>(self, args);
}

PyObject *nn_next(PyObject *self, PyObject *args) {
  return type_dispatcher_1<typed_nn_next>(self, args);
}

PyObject *nn_stop(PyObject *self, PyObject *args) {
  return type_dispatcher_1<typed_nn_stop>(self, args);
}


PyObject *nn_rewind(PyObject *self, PyObject *args) {
  return type_dispatcher_1<typed_nn_rewind>(self, args);
}



PyObject *populate(PyObject *self, PyObject *args) {
  return type_dispatcher_2<typed_populate>(self, args);
}

PyObject *particles_in_sphere(PyObject *self, PyObject *args) {
  return type_dispatcher_2<typed_particles_in_sphere>(self, args);
}
