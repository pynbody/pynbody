"""
Efficient 3D KDTree implementation for fast geometrical calculations such as neighbour lists and smoothing lengths.

"""
import logging
import time
import warnings

import numpy as np

from .. import array as ar, config, util
from . import kdmain

logger = logging.getLogger("pynbody.kdtree")

# Boundary type definition must exactly match the C++ definition (see kd.h)
Boundary = np.dtype([
    ('fMin', np.float32, (3,)),
    ('fMax', np.float32, (3,))
])

# KDNode type definition must exactly match the C++ definition (see kd.h)
KDNode = np.dtype([
    ('fSplit', np.float32),
    ('bnd', Boundary),
    ('iDim', np.int32),
    ('pLower', np.intp),
    ('pUpper', np.intp)
])

class KDTree:
    """KDTree can be used for smoothing, interpolating and geometrical queries.


    You are unlikely to need to interact with the KDTree directly, instead using the higher-level
    SPH functionality such as

    >>> f = pynbody.load("my_snapshot")
    >>> f.gas['smooth'] # will build KDTree and calculate smoothing lengths for gas, if not available on disk
    >>> f.gas['rho'] # will calculate densities for gas, if not available on disk
    >>> f.dm['rho'] # will calculate densities for dark matter, if not available on disk

    KDTree can also be used to accelerate geometrical queries on a snapshot, e.g.:

    >>> f = pynbody.load('my_snapshot')
    >>> f.build_kdtree()
    >>> f[pynbody.filt.Sphere('10 kpc')]

    See :ref:`performance of filters <filters_tutorial_performance_implications>` for more information.

    Most KDTree operations proceed in parallel, using the number of threads specified in the
    pynbody configuration.

    Performance statistics can be tested using the ``performance_kdtree.py`` script in the tests folder.

    Note that the KDTree takes into account periodicity of cosmological volumes if it can. However, as soon as a
    snapshot has been rotated, this may become impossible. In this case, the KDTree will issue a warning,
    and will not use periodicity. This is to avoid incorrect results due to the periodicity assumption,
    but it will lead to artefacts on the edge of the box. If you are working with a cosmological simulation,
    it is best to perform any KDTree operations (such as smoothing) before rotating the snapshot, e.g.:

    >>> f = pynbody.load('my_snapshot')
    >>> f['rho']; f['smooth'] # force KDTree to be built and density estimations made
    >>> pynbody.analysis.faceon(f) # rotate the snapshot
    >>> pynbody.plot.image(f.g, qty='rho', width='10 kpc')

    """

    PROPID_HSM = 1
    PROPID_RHO = 2
    PROPID_QTYMEAN_1D = 3
    PROPID_QTYMEAN_ND = 4
    PROPID_QTYDISP_1D = 5
    PROPID_QTYDISP_ND = 6
    PROPID_QTYDIV  = 7
    PROPID_QTYCURL = 8

    def __init__(self, pos, mass, leafsize=32, boxsize=None, num_threads=None, shared_mem=False):
        """Create a KDTree

        Most users will not need to call this directly, but will instead use the higher-level functions, either
        via accessing SPH properties on a snapshot, or calling :meth:`pynbody.snapshot.simsnap.SimSnap.build_tree`.

        Parameters
        ----------
        pos : pynbody.array.SimArray
            Particles positions.
        mass : pynbody.array.SimArray
            Particles mass.
        leafsize : int, optional
            The number of particles in leaf nodes (default 32).
        boxsize : float, optional
            Boxsize (default None)
        num_threads : int, optional
            Number of threads to use when building tree (if None, use configured/detected number of processors).
        shared_mem : bool, optional
            Whether to keep kdtree in shared memory so that it can be shared between processes (default False).
        """

        num_threads = self._set_num_threads(num_threads)

        # get a power of 2 for num_threads to pass to the constructor, because
        # otherwise the workload will not be balanced across threads and they
        # will be wasted
        num_threads_init = 2 ** int(np.log2(num_threads))


        self.leafsize = int(leafsize)
        self.kdtree = kdmain.init(pos, mass, self.leafsize)
        nodes = kdmain.get_node_count(self.kdtree)

        # zeroing the KDNode elements below isn't strictly necessary but helps with testing/debugging since some
        # of the memory won't be used

        if shared_mem:
            from ..array import shared
            self.kdnodes = shared.make_shared_array(nodes, KDNode, zeros=True)
            self.particle_offsets = shared.make_shared_array(len(pos), np.intp)
        else:
            self.kdnodes = np.zeros(nodes, dtype=KDNode)
            self.particle_offsets = np.empty(len(pos), dtype=np.intp)

        kdmain.build(self.kdtree, self.kdnodes, self.particle_offsets, num_threads_init)

        self.boxsize = boxsize
        self._pos = pos
        self.set_kernel(config['sph'].get('Kernel', 'CubicSpline'))

    def _set_num_threads(self, num_threads):
        if num_threads is None:
            num_threads = int(config["number_of_threads"])
        self.num_threads = num_threads
        return num_threads

    def serialize(self):
        """Produce a serialized description of the tree"""
        return self.leafsize, self.boxsize, self.kdnodes, self.particle_offsets, self._kernel_id

    @classmethod
    def deserialize(cls, pos, mass, serialized_state, num_threads = None, boxsize=None):
        """Reconstruct a KDTree from a serialized state."""
        self = object.__new__(cls)
        self._set_num_threads(num_threads)
        leafsize, boxsize_s, kdnodes, particle_offsets, kernel_id = serialized_state
        if boxsize is not None and abs(boxsize-boxsize_s) > 1e-6:
            warnings.warn("Boxsize in serialized state does not match boxsize passed to deserialize")
        self.kdtree = kdmain.init(pos, mass, leafsize)
        self.leafsize = leafsize
        nodes = kdmain.get_node_count(self.kdtree)
        if len(kdnodes) != nodes:
            raise ValueError("Number of nodes in serialized state does not match number of nodes in kdtree")
        self.kdnodes = kdnodes
        if len(particle_offsets) != len(pos):
            raise ValueError("Number of particle offsets in serialized state does not match number of particles")
        self.particle_offsets = particle_offsets
        self.boxsize = boxsize
        self._pos = pos
        self._kernel_id = kernel_id

        kdmain.import_prebuilt(self.kdtree, self.kdnodes, self.particle_offsets, 0)

        return self



    def particles_in_sphere(self, center, radius):
        """Find particles within a sphere.

        Most users will not need to call this directly, but will instead use :class:`pynbody.filt.Sphere`.

        Parameters
        ----------
        center : array_like
            Center of the sphere.
        radius : float
            Radius of the sphere.

        Returns
        -------
        indices : array_like
            Indices of the particles within the sphere.
        """
        smx = kdmain.nn_start(self.kdtree, 1, self.boxsize)

        particle_ids = kdmain.particles_in_sphere(self.kdtree, smx, center[0], center[1], center[2], radius)

        kdmain.nn_stop(self.kdtree, smx)
        return particle_ids

    def nn(self, nn=None):
        """Generator of neighbour list.

        Parameters
        ----------
        nn : int, None
            number of neighbours. If None, default to 64.

        Yields
        ------
        nbr_list : list[int, float, list[int], list[float]]

            Information about the neighbours of a particle.

            List with four elements:

            * ``nbr_list[0]`` (``int``): the index of the particle in the snapshot's arrays.
            * ``nbr_list[1]`` (``float``): the smoothing length of the particle
            * ``nbr_list[2]`` (``list[int]``): list of the indexes of the `nn` neighbouring particles.
            * ``nbr_list[3]`` (``list[float]``): list of distances squared of each neighbouring particles.

            The lists of particle and of the relative distance squared are not sorted.

        """
        if nn is None:
            nn = 64

        smx = kdmain.nn_start(self.kdtree, int(nn), self.boxsize)
        kdmain.domain_decomposition(self.kdtree, 1)

        while True:
            nbr_list = kdmain.nn_next(self.kdtree, smx)
            if nbr_list is None:
                break
            yield nbr_list

        kdmain.nn_stop(self.kdtree, smx)

    def all_nn(self, nn=None):
        """A list of neighbours information for all the particles in the snapshot."""
        return [x for x in self.nn(nn)]

    @staticmethod
    def array_name_to_id(name):
        """Convert the pynbody name of an array to the corresponding integer ID in the C++ code."""
        if name == "smooth":
            return 0
        elif name == "rho":
            return 1
        elif name == "mass":
            return 2
        elif name == "qty":
            return 3
        elif name == "qty_sm":
            return 4
        else:
            raise ValueError("Unknown KDTree array")

    def set_array_ref(self, name, ar):
        """Give the C++ code access to a given array.

        Parameters
        ----------

        name : str
            Name of the array to set (can be 'smooth', 'rho', 'mass', 'qty', 'qty_sm')
        ar : pynbody.array.SimArray
            Array that the C++ code will access. Note that INCREF is called on the array, so it will be kept
            alive as long as the KDTree object is alive (or until another set_array_ref call replaces it.)
        """

        if self.array_name_to_id(name) < 3:
            if not np.issubdtype(self._pos.dtype, ar.dtype):
                raise TypeError(
                    "KDTree requires matching dtypes for %s (%s) and pos (%s) arrays"
                    % (name, ar.dtype, self._pos.dtype)
                )
        kdmain.set_arrayref(self.kdtree, self.array_name_to_id(name), ar)
        assert self.get_array_ref(name) is ar

    def get_array_ref(self, name):
        """Get the current array reference for a given name ('smooth', 'rho', 'mass', 'qty', or 'qty_sm')."""
        return kdmain.get_arrayref(self.kdtree, self.array_name_to_id(name))

    def smooth_operation_to_id(self, name):
        """Convert the name of a smoothing operation to the corresponding integer ID in the C++ code.

        Valid names are:

        * 'hsm' : compute smoothing lengths
        * 'rho' : compute density
        * 'qty_mean' : compute SPH mean of the 'qty' array
        * 'qty_disp' : compute SPH dispersion of the 'qty' array
        * 'qty_div' : compute divergence of the 'qty' array
        * 'qty_curl' : compute curl of the 'qty' array

        """

        if name == "hsm":
            return self.PROPID_HSM
        elif name == "rho":
            return self.PROPID_RHO
        elif name == "qty_mean":
            input_array = self.get_array_ref("qty")
            if len(input_array.shape) == 1:
                return self.PROPID_QTYMEAN_1D
            elif len(input_array.shape) == 2:
                if input_array.shape[1] != 3:
                    raise ValueError("Currently only able to smooth 3D or 1D arrays")
                return self.PROPID_QTYMEAN_ND
        elif name == "qty_disp":
            input_array = self.get_array_ref("qty")
            if len(input_array.shape) == 1:
                return self.PROPID_QTYDISP_1D
            elif len(input_array.shape) == 2:
                if input_array.shape[1] != 3:
                    raise ValueError("Currently only able to smooth 3D or 1D arrays")
                return self.PROPID_QTYDISP_ND
        elif name == "qty_div":
            input_array = self.get_array_ref("qty")
            if len(input_array.shape) != 2 and input_array.shape[1] != 3:
                raise ValueError("Can only compute divergence of 3D arrays")
            return self.PROPID_QTYDIV
        elif name == "qty_curl":
            input_array = self.get_array_ref("qty")
            if len(input_array.shape) != 2 and input_array.shape[1] != 3:
                raise ValueError("Can only compute curl of 3D arrays")
            return self.PROPID_QTYCURL
        else:
            raise ValueError("Unknown smoothing request %s" % name)


    def set_kernel(self, kernel = 'CubicSpline'):
        """Set the kernel for smoothing operations.

        Parameters
        ----------
        kernel : str
            Keyword to specify the smoothing kernel. Options: 'CubicSpline', 'WendlandC2'
        """
        if kernel == 'CubicSpline':
            self._kernel_id = 0
        elif kernel == 'WendlandC2':
            self._kernel_id = 1
        else:
            raise ValueError("Kernel %r not recognised. Please choose either 'CubicSpline' or 'WendlandC2'." % kernel)


    def populate(self, mode, nn):
        """Create the KDTree and perform the operation specified by `mode`.

        Parameters
        --------
        mode : str (see `kdtree.smooth_operation_to_id`)
            Specify operation to perform (compute smoothing lengths, density, SPH mean, or SPH dispersion).
        nn : int
            Number of neighbours to be considered when smoothing.
        """


        if nn is None:
            nn = 64

        smx = kdmain.nn_start(self.kdtree, int(nn), self.boxsize)

        try:
            propid = self.smooth_operation_to_id(mode)

            if propid == self.PROPID_HSM:
                kdmain.domain_decomposition(self.kdtree, self.num_threads)


            if self.num_threads == 1:
                NT = 8
                util.thread_map(
                    kdmain.populate,
                    [self.kdtree] * NT,
                    [smx] * NT,
                    [propid] * NT,
                    list(range(0, NT)),
                    [self._kernel_id] * NT
                )
            else:
                util.thread_map(
                    kdmain.populate,
                    [self.kdtree] * self.num_threads,
                    [smx] * self.num_threads,
                    [propid] * self.num_threads,
                    list(range(0, self.num_threads)),
                    [self._kernel_id] * self.num_threads
                )
        finally:
            # Free C-structures memory
            kdmain.nn_stop(self.kdtree, smx)

    def sph_mean(self, array, nsmooth=64):
        r"""Calculate the SPH mean of a simulation array.

        Given a kernel W, this calculates

        .. math::

          r_i = \sum_{j=0}^{N_{smooth}}\left( \frac{m_j}{\rho_j} q_i W(|x_i - x_j|, s_i) \right)

        where r is the result of the smoothing, m is the mass array, rho is the density array, q is the input array,
        s_i is the smoothing length, and W is the kernel function.

        The actual computation is done in the C++-extension function smooth.cpp::smMeanQty[1,N]D.

        Parameters
        ----------
        array : pynbody.array.SimArray
            Quantity to smooth (compute SPH interpolation at particles position).

        nsmooth:
            Number of neighbours to use when smoothing.

        Returns
        -------
        output : pynbody.array.SimArray
            The SPH mean of the input array.
        """
        output = np.empty_like(array)

        if hasattr(array, "units"):
            output = output.view(ar.SimArray)
            output.units = array.units

        self.set_array_ref("qty", array)
        self.set_array_ref("qty_sm", output)

        logger.info("Smoothing array with %d nearest neighbours" % nsmooth)
        start = time.time()
        self.populate("qty_mean", nsmooth)
        end = time.time()

        logger.info("SPH smooth done in %5.3g s" % (end - start))

        return output

    def sph_dispersion(self, array, nsmooth=64):
        r"""Calculate the SPH dispersion of a simulation array.

        Given a kernel W, this routine computes the smoothed quantity:

        .. math::

          r_i = \sum_{j=0}^{N_{smooth}}\left( \frac{m_j}{\rho_j} q_i W(|x_i - x_j|, s_i) \right)

        Then the squared root of the SPH smoothed variance:

        .. math::

          d_i = \left\{ \sum_{j=0}^{N_{smooth}} \frac{m_j}{\rho_j} (q_i - r_i)^2 W(|x_i - x_j|, s_i) \right\}^{1/2}

        where d_i is the calculated dispersion, m is the mass array, rho is the density array, q is the input array,
        s_i is the smoothing length, and W is the kernel function.

        Actual computation is done in the C++-extension functions smooth.cpp::smDispQty[1,N]D.

        Parameters
        ----------
        array : pynbody.array.SimArray
            Quantity to compute dispersion of.

        nsmooth: int
            Number of neighbours to use when smoothing.

        Returns
        -------
        output : pynbody.array.SimArray
            The dispersion of the input array.
        """
        output = np.empty_like(array)
        if hasattr(array, "units"):
            output = output.view(ar.SimArray)
            output.units = array.units

        self.set_array_ref("qty", array)
        self.set_array_ref("qty_sm", output)

        logger.info("Getting dispersion of array with %d nearest neighbours" % nsmooth)
        start = time.time()
        self.populate("qty_disp", nsmooth)
        end = time.time()

        logger.info("SPH dispersion done in %5.3g s" % (end - start))

        return output

    def _sph_differential_operator(self, array, op, nsmooth=64):
        if op == "div":
            op_label = "divergence"
            output = np.empty(len(array), dtype=array.dtype)
        elif op == "curl":
            op_label = "curl"
            output = np.empty_like(array)
        else:
            raise ValueError("Can only compute curl or divergence")

        if hasattr(array, "units"):
            output = output.view(ar.SimArray)
            output.units = array.units/self._pos.units

        self.set_array_ref("qty", array)
        self.set_array_ref("qty_sm", output)

        logger.info("Getting %s of array with %d nearest neighbours" % (op_label, nsmooth))
        start = time.time()
        self.populate("qty_%s" % op, nsmooth)
        end = time.time()

        logger.info(f"SPH {op_label} done in {end - start:5.3g} s")

        return output

    def sph_curl(self, array, nsmooth=64):
        """Calculate the curl of a given array using SPH smoothing.

        Parameters
        ----------

        array : pynbody.array.SimArray
            Quantity to compute curl of.

        nsmooth : int
            Number of neighbours to use when smoothing.

        Returns
        -------
        pynbody.array.SimArray:
            The curl of the input array.
        """
        return self._sph_differential_operator(array, "curl", nsmooth)

    def sph_divergence(self, array, nsmooth=64):
        """Calculate the divergence of a given array using SPH smoothing.

        Parameters
        ----------

        array : pynbody.array.SimArray
            Quantity to compute curl of.

        nsmooth : int
            Number of neighbours to use when smoothing.

        Returns
        -------
        pynbody.array.SimArray:
            The curl of the input array.
        """
        return self._sph_differential_operator(array, "div", nsmooth)

    def __del__(self):
        if hasattr(self, "kdtree"):
            kdmain.free(self.kdtree)
