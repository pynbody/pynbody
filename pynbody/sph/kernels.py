"""SPH kernel details"""

from __future__ import annotations

import numpy as np
import scipy.integrate as integrate


class KernelBase:
    """Base class for SPH kernels"""

    _sample_cache = {}

    def __init__(self):
        self.h_power = 3
        # Return the power of the smoothing length which appears in
        # the denominator of the expression for the general kernel.
        # Will be 3 for 3D kernels, 2 for 2D kernels.

        self.max_d = 2
        # The maximum value of the displacement over the smoothing for
        # which the kernel is non-zero

    def _get_samples_from_cache(self):
        if hash(self) in KernelBase._sample_cache:
            return KernelBase._sample_cache[hash(self)]
        else:
            return None

    def get_samples(self, dtype=np.float32):
        import time
        s = time.time()

        samples = self._get_samples_from_cache()
        if samples is None:
            sample_pts = np.arange(0, 4.01, 0.02)
            samples = np.array([self.get_value(x ** 0.5) for x in sample_pts], dtype=dtype)
            KernelBase._sample_cache[hash(self)] = samples

        return samples

    def get_value(self, d, h=1) -> float:
        """Get the value of the kernel for a given smoothing length.

        Here ``d`` is the separation *in units of the smoothing length*, i.e.
        ``d = r/h``. For a vectorised version taking the separation directly,
        see :meth:`value`.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def value(self, r, h):
        r"""Get the kernel :math:`W(r, h)`, vectorised over ``r`` and ``h``.

        .. versionadded:: 2.6.0

        Unlike :meth:`get_value`, this takes the separation ``r`` directly
        rather than ``r/h``, and both arguments may be arrays. This is the
        form needed when evaluating a kernel for a list of particle pairs,
        e.g. inside a callback passed to
        :meth:`pynbody.kdtree.KDTree.pair_reduce`.

        Parameters
        ----------
        r : array_like
            Separation between particles.
        h : array_like
            Smoothing length. The kernel has support out to ``r = 2h``.

        Returns
        -------
        numpy.ndarray
            The kernel value, normalised such that
            :math:`4\pi \int W(r, h) r^2 dr = 1`.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def gradient(self, r, h):
        r"""Get :math:`dW/dr`, vectorised over ``r`` and ``h``.

        .. versionadded:: 2.6.0

        This is the radial derivative of :meth:`value`, and is negative on
        :math:`0 < r < 2h`. The full gradient with respect to the position of
        particle :math:`i` is

        .. math::

            \nabla_i W(|x_i - x_j|, h) = \frac{dW}{dr} \frac{x_i - x_j}{r}

        Parameters
        ----------
        r : array_like
            Separation between particles.
        h : array_like
            Smoothing length.

        Returns
        -------
        numpy.ndarray
            The radial derivative of the kernel.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def _as_arrays(r, h):
        return (np.asarray(r, dtype=np.float64),
                np.asarray(h, dtype=np.float64))

    def projection(self) -> KernelBase:
        """Return a 2D projection of this kernel"""
        return Kernel2D(self)

    def __hash__(self):
        return hash(self.__class__)

    @classmethod
    def get_c_kernel_id(cls) -> int:
        """Return the C kernel id for this kernel

        This is used to select the appropriate C code for the kernel, and must match
        the kernel id defined in the Kernel::create function in kernels.hpp"""
        raise NotImplementedError("Subclasses must implement this method")

class CubicSplineKernel(KernelBase):
    """A cubic spline kernel. This is the default kernel used by pynbody."""
    def get_value(self, d, h=1):
        if d < 1:
            f = 1. - (3. / 2) * d ** 2 + (3. / 4.) * d ** 3
        elif d < 2:
            f = 0.25 * (2. - d) ** 3
        else:
            f = 0

        return f / (np.pi * h ** 3)

    def value(self, r, h):
        r, h = self._as_arrays(r, h)
        q = r / h
        f = np.where(q < 1.0,
                     1.0 - 1.5 * q ** 2 + 0.75 * q ** 3,
                     0.25 * np.clip(2.0 - q, 0.0, None) ** 3)
        return f / (np.pi * h ** 3)

    def gradient(self, r, h):
        r, h = self._as_arrays(r, h)
        q = r / h
        df = np.where(q < 1.0,
                      -3.0 * q + 2.25 * q ** 2,
                      -0.75 * np.clip(2.0 - q, 0.0, None) ** 2)
        return df / (np.pi * h ** 4)

    @classmethod
    def get_c_kernel_id(cls):
        return 0

class WendlandC2Kernel(KernelBase):
    """A Wendland C2 (quintic) kernel. This is the default kernel used by EAGLE."""

    def get_value(self, d, h=1):
        if d < 2:
            f = (1. - (d / 2.))**4 * (2. * d + 1)
        else:
            f = 0

        return (21. * f) / (16. * np.pi * h ** 3)

    def value(self, r, h):
        r, h = self._as_arrays(r, h)
        q = r / h
        t = np.clip(1.0 - 0.5 * q, 0.0, None)
        return 21.0 * t ** 4 * (2.0 * q + 1.0) / (16.0 * np.pi * h ** 3)

    def gradient(self, r, h):
        # d/dq [ (1-q/2)^4 (2q+1) ] = -5 q (1-q/2)^3
        r, h = self._as_arrays(r, h)
        q = r / h
        t = np.clip(1.0 - 0.5 * q, 0.0, None)
        return -21.0 * 5.0 * q * t ** 3 / (16.0 * np.pi * h ** 4)

    @classmethod
    def get_c_kernel_id(cls):
        return 1


class Kernel2D(KernelBase):
    """A 2D spline kernel, generated by numerically projecting an underlying 3D kernel"""
    def __init__(self, k_orig=CubicSplineKernel()):
        """Create a 2D kernel by projecting a 3D kernel. The 3D kernel is passed as an argument."""
        self.h_power = 2
        self.max_d = k_orig.max_d
        self.k_orig = k_orig

    def projection(self):
        raise ValueError("Cannot project a 2D kernel")

    def get_value(self, d, h=1):
        return 2 * integrate.quad(lambda z: self.k_orig.get_value(np.sqrt(z ** 2 + d ** 2), h), 0, 2*h)[0]

    def get_c_kernel_id(self):
        raise NotImplementedError("2D kernels are not supported in C")

    def __hash__(self):
        return hash((self.__class__, self.k_orig))


def create_kernel_from_c_id(kernel_id: int) -> KernelBase:
    """Create a kernel object from the integer id used by the C++ code.

    .. versionadded:: 2.6.0

    This is the inverse of :meth:`KernelBase.get_c_kernel_id`, and is used to
    recover a python-side kernel from a :class:`pynbody.kdtree.KDTree` whose
    kernel was set from a serialized state.
    """
    for subclass in KernelBase.__subclasses__():
        try:
            if subclass.get_c_kernel_id() == kernel_id:
                return subclass()
        except (NotImplementedError, TypeError):
            # e.g. Kernel2D, which has no C counterpart
            continue
    raise ValueError("No kernel corresponds to C kernel id %r" % kernel_id)


def create_kernel(spec) -> KernelBase:
    """Create a kernel object from a string specification, a type, an existing kernel object, or a None

    This function is used to create a kernel object from a variety of input types. It is used by the
    framework to allow the user flexibility in specifying the kernel type.

    If the input is a string, it is assumed to be the name of a kernel class, and an object of that class
    is created. You can use the name of the class with or without the 'Kernel' suffix, and the case is
    ignored. For example, 'WendlandC2Kernel', 'wendlandc2', and 'WendlandC2' all return a WendlandC2Kernel
    instance.

    If the input is a subclass of KernelBase, it is assumed to be a kernel object, and is returned as is.

    If the input is None, a default kernel is created and returned.

    Returns
    -------
    KernelBase
        A kernel object

    """
    if spec is None:
        from ..configuration import config
        return create_kernel(config['sph'].get('kernel', 'CubicSplineKernel'))
    elif isinstance(spec, type):
        return spec()
    elif isinstance(spec, KernelBase):
        return spec
    elif isinstance(spec, str):
        for subclass in KernelBase.__subclasses__():
            subclass_name = subclass.__name__
            if (subclass_name.lower() == spec.lower() or
                    subclass_name.endswith('Kernel') and subclass_name[:-6].lower() == spec.lower()):
                return subclass()
        else:
            raise ValueError("Unknown kernel '%s'" % spec)
    else:
        raise ValueError("Unknown kernel specification %r" % spec)
