"""

KDTree
======

Provides access to nearest neighbour lists and smoothing lengths.

"""
import logging
import time
import warnings

import numpy as np

from .. import array as ar, config
from . import kdmain

logger = logging.getLogger("pynbody.sph.kdtree")


class KDTree:
    """KDTree used for smoothing."""

    PROPID_HSM = 1
    PROPID_RHO = 2
    PROPID_QTYMEAN_1D = 3
    PROPID_QTYMEAN_ND = 4
    PROPID_QTYDISP_1D = 5
    PROPID_QTYDISP_ND = 6
    PROPID_QTYDIV  = 7
    PROPID_QTYCURL = 8

    def __init__(self, pos, mass, leafsize=32, boxsize=None):
        """
        Parameters
        ----------
        pos : pynbody.array.SimArray
            Particles positions.
        mass : pynbody.array.SimArray
            Particles mass.
        leafsize : int, optional
            The number of particles in leaf nodes (default 32).
        boxsize : float, optional
            Boxsize (default None).
        """
        self.kdtree = kdmain.init(pos, mass, int(leafsize))
        self.derived = True
        self.boxsize = boxsize
        self._pos = pos
        self.s_len = len(pos)
        self.flags = {"WRITEABLE": False}

    def nn(self, nn=None):
        """Generator of neighbour list.

        This method provides an interface to the neighbours search of the C-extension kdmain.

        Parameters
        ----------
        nn : int, None
            number of neighbours. If None, default to 64.

        Yields
        ------
        nbr_list : list[int, float, list[int], list[float]]
            Information about the neighbours of a particle.
            List with four elements:
                nbr_list[0] : int
                    the index of the particle in the snapshot's arrays.
                nbr_list[1] : float
                    the smoothing length of the particle (i.e. nbr_list[1] == snap['smooth'][nbr_list[0]]).
                nbr_list[2] : list[int]
                    list of the indexes of the `nn` neighbouring particles.
                nbr_list[3] : list[float]
                    list of distances squared of each neighbouring particles.

            The lists of particle and of the relative distance squared are not sorted.

        """
        if nn is None:
            nn = 64

        smx = kdmain.nn_start(self.kdtree, int(nn), 1, self.boxsize)
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
        if self.array_name_to_id(name) < 3:
            if not np.issubdtype(self._pos.dtype, ar.dtype):
                raise TypeError(
                    "KDTree requires matching dtypes for %s (%s) and pos (%s) arrays"
                    % (name, ar.dtype, self._pos.dtype)
                )
        kdmain.set_arrayref(self.kdtree, self.array_name_to_id(name), ar)
        assert self.get_array_ref(name) is ar

    def get_array_ref(self, name):
        return kdmain.get_arrayref(self.kdtree, self.array_name_to_id(name))

    def smooth_operation_to_id(self, name):
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

    def populate(self, mode, nn, kernel = 'CubicSpline'):
        """Create the KDTree and perform the operation specified by `mode`.

        Parameters
        --------
        mode : str (see `kdtree.smooth_operation_to_id`)
            Specify operation to perform (compute smoothing lengths, density, SPH mean, or SPH dispersion).
        nn : int
            Number of neighbours to be considered when smoothing.
        kernel : str
            Keyword to specify the smoothing kernel. Options: 'CubicSpline', 'WendlandC2'
        """
        from . import _thread_map

        n_proc = config["number_of_threads"]

        if kdmain.has_threading() is False and n_proc > 1:
            n_proc = 1
            warnings.warn(
                "Pynbody is configured to use threading for the KDTree, but pthread support was not available during compilation. Reverting to single thread.",
                RuntimeWarning,
            )

        if nn is None:
            nn = 64

        smx = kdmain.nn_start(self.kdtree, int(nn), n_proc, self.boxsize)

        propid = self.smooth_operation_to_id(mode)

        if propid == self.PROPID_HSM:
            kdmain.domain_decomposition(self.kdtree, n_proc)

        if kernel == 'CubicSpline':
            kernel = 0
        elif kernel == 'WendlandC2':
            kernel = 1
        else:
            raise ValueError(
                "Kernel keyword %s not recognised. Please choose either 'CubicSpline' or 'WendlandC2'." % kernel
            )

        if n_proc == 1:
            kdmain.populate(self.kdtree, smx, propid, 0, kernel)
        else:
            _thread_map(
                kdmain.populate,
                [self.kdtree] * n_proc,
                [smx] * n_proc,
                [propid] * n_proc,
                list(range(0, n_proc)),
                [kernel] * n_proc
            )

        # Free C-structures memory
        kdmain.nn_stop(self.kdtree, smx)

    def sph_mean(self, array, nsmooth=64, kernel = 'CubicSpline'):
        r"""Calculate the SPH mean of a simulation array.

        It's the application of the SPH interpolation formula for computing the smoothed quantity at particles position.
        It uses the cubic spline smoothing kernel W.

            qty_sm[i] = \sum_{j=0}^{nsmooth}( mass[j]/rho[j] * qty[i] * W(|pos[i] - pos[j]|, smooth[i]) )

        Actual computation is done in the C-extension functions smooth.cpp::smMeanQty[1,N]D.

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
        self.populate("qty_mean", nsmooth, kernel)
        end = time.time()

        logger.info("SPH smooth done in %5.3g s" % (end - start))

        return output

    def sph_dispersion(self, array, nsmooth=64, kernel = 'CubicSpline'):
        r"""Calculate the SPH dispersion of a simulation array.

        It uses the cubic spline smoothing kernel W.

        First it computes the smoothed quantity:

            qty_sm[i] = \sum_{j=0}^{nsmooth}( mass[j]/rho[j] * qty[i] * W(|pos[i] - pos[j]|, smooth[i]) )

        Then the squared root of the SPH smoothed variance:

            qty_disp[i] = \sqrt( \sum_{j=0}^{nsmooth}( mass[j]/rho[j] * (qty_sm[i] - qty[j])^2 * W(|pos[i] - pos[j]|, smooth[i]) ) )

        Actual computation is done in the C-extension functions smooth.cpp::smDispQty[1,N]D.

        Parameters
        ----------
        array : pynbody.array.SimArray
            Quantity to compute dispersion of.
        nsmooth:
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
        self.populate("qty_disp", nsmooth, kernel)
        end = time.time()

        logger.info("SPH dispersion done in %5.3g s" % (end - start))

        return output

    def _sph_differential_operator(self, array, op, nsmooth=64, kernel = 'CubicSpline'):
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
        self.populate("qty_%s" % op, nsmooth, kernel)
        end = time.time()

        logger.info(f"SPH {op_label} done in {end - start:5.3g} s")

        return output

    def sph_curl(self, array, nsmooth=64, kernel = 'CubicSpline'):
        return self._sph_differential_operator(array, "curl", nsmooth, kernel)

    def sph_divergence(self, array, nsmooth=64, kernel = 'CubicSpline'):
        return self._sph_differential_operator(array, "div", nsmooth, kernel)

    def __del__(self):
        if hasattr(self, "kdtree"):
            kdmain.free(self.kdtree)
