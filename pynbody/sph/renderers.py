"""Core classes for rendering images of SPH simulations.

For most users, the :func:`pynbody.plot.sph.image` function is the most convenient interface for rendering images of SPH
simulations. A slightly lower-level approach is to use :func:`make_render_pipeline` to create a renderer object, which
then produces an image when the :meth:`ImageRendererBase.render` method is called.

For complete control, one may use the :class:`ImageRenderer` class directly.
"""

from __future__ import annotations

import concurrent
import concurrent.futures
import copy
from types import NoneType

import numpy as np
import scipy

from .. import array as array_module, config, snapshot, units
from ..configuration import config_parser, logger
from . import _render, kernels


def _kernel_suitable_for_denoise(kernel):
    if isinstance(kernel, kernels.Kernel2D):
        return False
    else:
        return True

def _auto_denoise(sim, kernel):
    """Returns True if pynbody thinks denoise should be on for best results with this simulation/kernel combination."""

    if not _kernel_suitable_for_denoise(kernel):
        return False
    elif isinstance(sim.ancestor,snapshot.ramses.RamsesSnap):
        return True
    else:
        return False

class RenderPipelineLogicError(RuntimeError):
    pass

class ImageGeometry:
    """A class to store the geometry of an image to be rendered."""
    def __init__(self):
        self.x2 = self.y2 = 100.0
        self.x1 = self.y1 = -100.0
        self.z1 = -np.inf
        self.z2 = np.inf
        self.nx = self.ny = self.nz = config['image-default-resolution']
        # nz used only by 3d grid renderer
        self.z_plane = 0.0
        self.z_range = None
        self.z_camera = None

    @property
    def width(self):
        return self.x2 - self.x1

    def set_width(self, width: float):
        """Set the width of the image to be rendered. The image will be centered on the origin.

        You probably want to access this via the :meth:`ImageRenderer.set_width` method of the renderer, which can
        accept and convert units as needed.
        """
        self.x2 = self.y2 = width / 2
        self.x1 = self.y1 = -width / 2

    def set_camera_z(self, z: float):
        """Set the z position of the camera for the image to be rendered.

        A perspective image will be rendered with the camera at this z position, looking at the z_plane.
        The frustrum is defined by the x1, x2, y1, y2 values, which are at z=0.
        """
        self.z_camera = z

    def set_resolution(self, resolution: int):
        """Set the resolution of the image to be rendered, in pixels."""
        self.nx = self.ny = self.nz = resolution

    def restrict_z_range(self):
        """Restrict the z range to the same as the x range"""
        self.z1 = self.x1
        self.z2 = self.x2

    def copy(self) -> ImageGeometry:
        """Return a copy of this geometry object."""
        return copy.copy(self)

class ReadOnlyGeometry:
    """A wrapper around an :class:`ImageGeometry` object that makes it read-only."""
    def __init__(self, geometry):
        self._geometry = geometry

    def __getattr__(self, key):
        return getattr(self._geometry, key)

    def __setattr__(self, key, value):
        if key == '_geometry':
            super().__setattr__(key, value)
        else:
            raise AttributeError("This object is read-only")

    def __delattr__(self, key):
        raise AttributeError("This object is read-only")

    def copy(self):
        return self._geometry.copy()

class ImageRendererBase:
    """An abstract base class for image renderers"""

    def __init__(self, snap: snapshot.SimSnap):
        self._snapshot = snap

        self._array = None

        self._smooth = 'smooth'

        self._particle_array_slice = None

        self._geometry = ImageGeometry()

        self._out_units = None
        self._is_projected = False

        self.set_kernel()

        self.set_smooth_range()
        self.set_smooth_floor()

    @property
    def geometry(self):
        return self._geometry

    @property
    def is_projected(self):
        return self._is_projected

    def set_width(self, width: float | str | units.UnitBase):
        """Set the width of this renderer to the specified value.

        Parameters
        ----------
        width : float, str, units.UnitBase
            The width of the image to be rendered. If a string or unit, the value will be converted to the units
            of the simulation snapshot.
        """

        width = self._to_position_units(width)

        self._geometry.set_width(width)

    def _to_position_units(self, size):
        if size is None:
            return None
        if isinstance(size, str):
            size = units.Unit(size)
        if isinstance(size, units.UnitBase):
            size = size.in_units(self._snapshot['pos'].units, **self._snapshot.conversion_context())
        size = float(size)
        return size

    def set_smooth_range(self, smooth_min: float = 0.0, smooth_max: float = None):
        """Set the range of smoothing lengths in image pixels to be used in the image rendering.

        Any particles with smoothing lengths outside this range will be ignored.

        This is for use by the approximate renderer to render images at multiple resolutions.

        Note that :meth:`set_smoothing_floor` is a different method that does not exclude particles below the
        minimum, rather setting them to the minimum.

        Parameters
        ----------
        smooth_min : float, optional
            The minimum smoothing length to be used in the image rendering, specified in pixels. The default is
            zero.
        smooth_max : float, optional
            The maximum smoothing length to be used in the image rendering, specified in pixels. If None (default),
            there is no maximum.
        """
        self._smooth_min = smooth_min
        self._smooth_max = np.inf if smooth_max is None else smooth_max

    def set_smooth_floor(self, smooth_floor: float | str | units.UnitBase = 0.0):
        """Set the minimum smoothing length to be used in the image rendering, in position units (not pixels).

        Any particles with smoothing lengths below this value will have their smoothing lengths set to this value.
        The purpose is to exclude particles with very small smoothing lengths from the image rendering, which can
        cause excess noise e.g. in contoured images.
        """
        self._smooth_floor = self._to_position_units(smooth_floor)

    def set_particle_array_slice(self, slice):
        """Set the slice of particles to be rendered.

        This is used by the threaded renderer to split the rendering across multiple threads.

        Parameters
        ----------
        slice : slice
            The slice of particles to be rendered.
        """
        self._particle_array_slice = slice

    def set_kernel(self, kernel_spec: str | type | kernels.KernelBase | NoneType = None):
        """Set the kernel to be used for the image rendering.

        Parameters
        ----------

        kernel_spec :
            The kernel to be used for the image rendering. This can be specified as a string, a kernel class, or a
            kernel instance. If None, the default kernel is assigned. For more information see
            :func:`pynbody.sph.kernels.create_kernel`.

        Notes
        -----

        If a projected image is to be used, a 3D kernel should still be passed. The projection is handled internally.

        """
        kernel = kernels.create_kernel(kernel_spec)
        if isinstance(kernel, kernels.Kernel2D):
            raise ValueError("To obtain a projected image, pass the 3D kernel which will be projected internally.")
        self._kernel = kernel

    def _check_quantity_set(self):
        if self._array is None:
            raise RenderPipelineLogicError("This operation requires the rendering quantity to be set; call "
                                           "set_quantity first.")

    def set_quantity(self, qty):
        """Set the quantity to be rendered.

        Parameters
        ----------
        qty : str | numpy.ndarray
            The quantity to be rendered. If a string, the quantity is taken from the simulation snapshot.
        """
        if isinstance(qty, str):
            qty = self._snapshot[qty]
        self._array = qty

    def set_resolution(self, resolution: int):
        """Set the resolution of the image to be rendered.

        Parameters
        ----------
        resolution : int
            The resolution of the image to be rendered.
        """
        self._geometry.set_resolution(resolution)

    def set_output_units(self, units: str | units.UnitBase | None):
        """Set the output units of the image to be rendered. This will also change the projection status of the image.

        Parameters
        ----------
        units : str | units.UnitBase
            The units to be used for the output image. These are checked for compatibility with the array to be
            rendered, either in projection or in slice. If the units are not compatible in either interpretation,
            a UnitsException is raised. If None, the output units are set to the units of the array to be rendered.
        """
        self._check_quantity_set()
        if units is not None:
            self._is_projected = self._units_imply_projection(units)
        self._out_units = units

    def set_projection(self, is_projected: bool):
        """Set whether the image is to be rendered as a projection or a slice. This will also reset the output units."""
        self._is_projected = is_projected
        self._out_units = None

    def restrict_z_range(self):
        """Restrict the z range of the image to the same as the x range."""
        self._geometry.restrict_z_range()

    def copy(self, share_geometry=False):
        """Return a copy of this renderer.

        If share_geometry is True, the geometry of the image is shared between the original and the copy. Otherwise,
        the geometry is copied as well.
        """
        c = copy.copy(self)
        if not share_geometry:
            c._geometry = self._geometry.copy()
        return c

    def _units_imply_projection(self, units_):
        self._check_quantity_set()
        if units_ is None:
            return False
        try:
            self._array.units.ratio(units_, **self._snapshot.conversion_context())
            # if this fails, perhaps we're requesting a projected image?
            return False
        except units.UnitsException:
            # if the following fails, there's no interpretation this routine
            # can cope with. The error will be allowed to propagate.
            self._array.units.ratio(
                units_ / (self._snapshot['x'].units), **self._snapshot.conversion_context())
            return True

    def render(self) -> np.ndarray:
        """Render the image and return it as a numpy array or SimArray."""
        raise NotImplementedError("Subclasses must implement this method")


    def with_denoising(self, denoise : bool | NoneType = None) -> ImageRendererBase:
        """Return a version of this renderer that may inclue a denoising step.

        Parameters
        ----------
        denoise : bool | None
            Whether to include denoising in the rendering process. If None, denoising is applied only if the image
            is likely to benefit from it. If True, denoising is to be forced on the image; if that is actually
            impossible, this routine raises an exception. If False, denoising is never applied
            and the routine returns a straight-forward copy of the current renderer.
        """
        self._check_quantity_set()
        if denoise is None:
            if self._is_projected:
                denoise = False
            else:
                denoise = _auto_denoise(self._snapshot, self._kernel)

        if denoise:
            return DenoisedImageRenderer(self)
        else:
            return self

    def with_threading(self, num_threads : int | NoneType = None) -> ImageRendererBase:
        """Return a version of this renderer that will use the specified number of threads for rendering.

        If num_threads is None, the number of threads is determined by the configuration file.
        """
        self._check_quantity_set()
        if num_threads is None:
            num_threads = config['number_of_threads']
        return ThreadedImageRenderer(self, num_threads)

    def with_approximate(self, levels : int | NoneType = None, factor = 8) -> ImageRendererBase:
        """Return a version of this renderer that will use the specified number of approximation levels for rendering.

        For more information, see :class:`ApproximateImageRenderer`.

        Note that if the number of levels is less than 2, the original renderer is returned.

        Parameters
        ----------

        levels : int, optional
            The number of approximation levels to use. If None, the number of levels is determined by the size of the
            image and the zoom factor.

        factor : int, optional
            The zoom factor to use between levels of approximation. The default is 8.
        """
        self._check_quantity_set()
        if levels is None:
            levels = int(np.floor(np.log2(self.geometry.nx / 5)/np.log2(factor)))

        if levels<2:
            return self
        return ApproximateImageRenderer(self, levels, factor)

    def with_weighted_projection(self, weighting_array):
        """Return a version of this renderer that will render a weighted projection along the line of sight."""
        self._check_quantity_set()
        if isinstance(weighting_array, str):
            weighting_array = self._snapshot[weighting_array]
        if len(weighting_array) != len(self._snapshot):
            raise ValueError("Weighting array must have the same length as the snapshot")
        return ProjectionAverageImageRenderer(self, weighting_array)

    def with_volume_weighted_projection(self):
        """Return a version of this renderer that will render a volume-weighted projection along the line of sight."""
        self._check_quantity_set()
        return self.with_weighted_projection(np.ones_like(self._array.view(np.ndarray)))

class MultipassImageRenderer(ImageRendererBase):
    """A base class for image rendering using multiple passes to the underlying renderer"""
    def __init__(self, template, n_copies, share_geometry=False): # noqa - no need to call super constructor
        self._subrenderers : list[ImageRendererBase] = [template.copy(share_geometry=share_geometry)
                                                        for i in range(n_copies)]
        self._snapshot = template._snapshot
        if share_geometry:
            self._geometry = template._geometry
        else:
            self._geometry = ReadOnlyGeometry(template._geometry)

        self._shared_geometry = share_geometry
        self._is_projected = template._is_projected
        self._array = template._array
        self._smooth = template._smooth
        self._kernel = template._kernel
        self._out_units = template._out_units
        self._smooth_floor = template._smooth_floor


    def copy(self, share_geometry=False):
        copy = super().copy(share_geometry)
        copy._subrenderers = [r.copy() for r in self._subrenderers]
        if self._shared_geometry:
            # the subrenderers of the new copy should share the geometry of the copy
            # Note this is true even if the copy has a NEW geometry (i.e. if share_geometry=False)
            for r in copy._subrenderers:
                r._geometry = copy._geometry
        return copy

    def set_kernel(self, kernel_spec = None):
        super().set_kernel(kernel_spec)
        for r in self._subrenderers:
            r.set_kernel(kernel_spec)

    def set_quantity(self, qty):
        super().set_quantity(qty)
        for r in self._subrenderers:
            r.set_quantity(qty)

    def set_smooth_floor(self, smooth_floor = 0.0):
        super().set_smooth_floor(smooth_floor)
        for r in self._subrenderers:
            r.set_smooth_floor(smooth_floor)

    def set_output_units(self, units_: str | units.UnitBase):
        super().set_output_units(units_)
        for r in self._subrenderers:
            r.set_output_units(units_)

    def set_projection(self, is_projected: bool):
        super().set_projection(is_projected)
        for r in self._subrenderers:
            r.set_projection(is_projected)

    def with_threading(self, num_threads = None ):
        raise RenderPipelineLogicError("Threading cannot be set for a multipass image render. Try setting the threading status for the individual stages before generating the multipass renderer.")

    def render(self):
        return [r.render() for r in self._subrenderers]

    def set_smooth_range(self, smooth_min: float = 0.0, smooth_max: float = None):
        for r in self._subrenderers:
            r.set_smooth_range(smooth_min, smooth_max)

    def set_particle_array_slice(self, slice):
        for r in self._subrenderers:
            r.set_particle_array_slice(slice)


class DenoisedImageRenderer(MultipassImageRenderer):
    """A class to render images with denoising applied."""
    def __init__(self, base):
        if base._is_projected:
            raise RenderPipelineLogicError("Denoising not supported with projected images")
        super().__init__(base, 2)
        self._subrenderers[1].set_quantity(np.ones(len(self._snapshot), dtype=base._array.dtype))
        self._subrenderers[1].set_output_units(None)

    def render(self):
        result_source_field, result_noise_field = super().render()
        return result_source_field / result_noise_field

    def set_output_units(self, units_: str | units.UnitBase):
        ImageRendererBase.set_output_units(self, units_)
        self._subrenderers[0].set_output_units(units_)

    def with_denoising(self, denoise : bool | NoneType = None) -> ImageRendererBase:
        raise RenderPipelineLogicError("This render pipeline already has denoising applied.")

class ProjectionAverageImageRenderer(MultipassImageRenderer):
    """A class to render projected images."""
    def __init__(self, base: ImageRendererBase, weighting_array: np.ndarray):
        if base._is_projected:
            raise RenderPipelineLogicError("Cannot take a projected average of an already projected image.")
        super().__init__(base, 2)

        self._weight_array = weighting_array
        self.set_quantity(base._array)
        self.set_output_units(base._out_units)

    def with_projection(self, is_projected: bool) -> ImageRendererBase:
        raise RenderPipelineLogicError("This render pipeline already has projection applied.")

    def set_quantity(self, qty):
        super().set_quantity(qty)
        my_array = self._array
        self._subrenderers[0].set_quantity(my_array * self._weight_array)
        self._subrenderers[0].set_projection(True)
        self._subrenderers[1].set_quantity(self._weight_array)
        self._subrenderers[1].set_projection(True)

    def set_output_units(self, units_: str | units.UnitBase | None):
        if units.is_unit(units_) and hasattr(self._weight_array, 'units'):
            units_ *= self._weight_array.units
        ImageRendererBase.set_output_units(self, units_)
        self._subrenderers[0].set_output_units(units_)
        self._subrenderers[1].set_output_units(None)

    def render(self):
        result_source_field, result_weight_field = super().render()

        # note that we erradicate any unit information in the weight field, in case it is NoUnit, by casting to np.ndarray
        result = result_source_field / result_weight_field.view(np.ndarray)

        # now manually fix up the units:
        if hasattr(result_source_field, 'units') and hasattr(result_weight_field, 'units'):
            result.units = result_source_field.units / result_weight_field.units

        return result



class ThreadedImageRenderer(MultipassImageRenderer):
    """A class to render images across multiple threads."""
    def __init__(self, base, num_threads):
        """Create a threaded image renderer, rendering the image across the specified number of threads."""
        super().__init__(base, num_threads, share_geometry = True)
        self._num_threads = num_threads
        for i, r in enumerate(self._subrenderers):
            # render every num_threads particle, starting at i
            r.set_particle_array_slice(slice(i, None, num_threads))

    def render(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._subrenderers)) as executor:
            # logger.info("Rendering image on %d threads..." % self._num_threads)
            results = executor.map(lambda r: r.render(), self._subrenderers)

        return sum(results)


class ApproximateImageRenderer(MultipassImageRenderer):
    """A class to render images using a lower-resolution approximation for large smoothing lengths.

    Profiling shows that this approach is faster than the standard renderer when there are a large number
    of particles with smoothing lengths taking up a significant fraction of the image. However the performance
    gains generally decline with increasing number of processor cores."""

    def __init__(self, base, levels, factor=8):
        """Create an approximate image renderer, using the specified number of levels of approximation.

        Each level is rendered at factor x worse resolution than the previous level, and the smoothing length range is
        adjusted so that in each stage the smoothing length is within a factor of 2 of the resolution of the image.
        """
        super().__init__(base, levels, share_geometry=False)
        for level, renderer in enumerate(self._subrenderers):
            if level == 0:
                renderer.set_smooth_range(0, factor)
            elif level == levels - 1:
                renderer.set_smooth_range(1, None)
            else:
                renderer.set_smooth_range(1, factor)
            renderer.set_resolution(base._geometry.nx // (factor ** level))

    def _apply_zoom(self, images):
        """Apply a zoom to the images, to account for the fact that the images are rendered at different resolutions."""
        zoomed_images = [images[0]]
        for i in images[1:]:
            zoomed_result = scipy.ndimage.zoom(i, len(images[0])/len(i), order=1, grid_mode=True, mode='grid-constant')
            assert zoomed_result.shape == images[0].shape
            zoomed_images.append(zoomed_result)
        return zoomed_images

    def render(self):
        results = self._apply_zoom(super().render())
        summed = sum(results)
        return summed

class ImageRenderer(ImageRendererBase):
    """Implementation for rendering a simulation snapshot to 2d image"""

    def _calculate_wrapping_repeat_array(self, x1, x2):
        if 'boxsize' in self._snapshot.properties:
            boxsize = self._snapshot.properties['boxsize'].in_units(self._snapshot['pos'].units,
                                                                    **self._snapshot.conversion_context())
        else:
            boxsize = None

        if boxsize:
            # work out the tile offsets required to make the image wrap
            num_repeats = int(round((x2 - x1) / (2 * boxsize))) + 1
            repeat_array = np.linspace(-num_repeats * boxsize, num_repeats * boxsize, num_repeats * 2 + 1)
        else:
            repeat_array = [0.0]
        return repeat_array


    def render(self):
        kernel = kernels.create_kernel(self._kernel)

        if self._is_projected:
            kernel = kernel.projection()

        g = self._geometry

        with self._snapshot.immediate_mode:
            if self._particle_array_slice is not None:
                mass, rho, x, y, z, smooth = (self._snapshot[name][self._particle_array_slice]
                                              for name in ('mass', 'rho', 'x', 'y', 'z', self._smooth))
                array = self._array[self._particle_array_slice]
            else:
                mass, rho, x, y, z, smooth = (self._snapshot[name]
                                              for name in ('mass', 'rho', 'x', 'y', 'z', self._smooth))
                array = self._array

        if hasattr(array, 'units'):
            array_units = array.units
        else:
            array_units = 1.0

        native_units = array_units * mass.units / (rho.units * smooth.units ** kernel.h_power)

        if self._out_units is not None:
            conversion = native_units.ratio(self._out_units, **self._snapshot.conversion_context())
            out_units = self._out_units
        else:
            conversion = 1.0
            out_units = native_units

        smooth, array, mass, rho = (q.view(np.ndarray) for q in (smooth, array, mass, rho))

        image = self._call_c_renderer(array, g, kernel, mass, rho, smooth, x, y, z)

        if conversion != 1.0:
            image *= conversion

        image = image.view(array_module.SimArray)
        image.sim = self._snapshot
        image.units = out_units

        return image

    def _call_c_renderer(self, array, geometry, kernel, mass_array, rho_array, smooth_array, x_array, y_array, z_array):
        image = _render.render_image(geometry.nx, geometry.ny, x_array, y_array, z_array, smooth_array, geometry.x1, geometry.x2, geometry.y1, geometry.y2,
                                     geometry.z_camera or 0.0, geometry.z_plane, array, mass_array, rho_array,
                                     self._smooth_min, self._smooth_max, geometry.z1, geometry.z2,
                                     self._smooth_floor, kernel,
                                     self._calculate_wrapping_repeat_array(geometry.x1, geometry.x2),
                                     self._calculate_wrapping_repeat_array(geometry.y1, geometry.y2))
        return image


class Grid3dRenderer(ImageRenderer):
    """Implementation for rendering a simulation snapshot to a 3d grid"""

    def __init__(self, snap: snapshot.SimSnap):
        super().__init__(snap)
        self.geometry.restrict_z_range() # sets z1, z2 - here this is for the grid edges, not the camera

    def set_width(self, width: float | str | units.UnitBase):
        super().set_width(width)
        self.geometry.restrict_z_range() # sets z1, z2 - here this is for the grid edges, not the camera

    def _call_c_renderer(self, array, geometry, kernel, mass_array, rho_array, smooth_array, x_array, y_array, z_array):
        image = _render.to_3d_grid(geometry.nx, geometry.ny, geometry.nz, x_array, y_array, z_array,
                                   smooth_array, geometry.x1, geometry.x2, geometry.y1, geometry.y2, geometry.z1, geometry.z2,
                                   array, mass_array, rho_array, self._smooth_min, self._smooth_max, kernel,
                                   self._calculate_wrapping_repeat_array(geometry.x1, geometry.x2),
                                   self._calculate_wrapping_repeat_array(geometry.y1, geometry.y2),
                                   self._calculate_wrapping_repeat_array(geometry.z1, geometry.z2))
        return image

def make_render_pipeline(sim : snapshot.SimSnap, /,
                         quantity: str | np.ndarray = 'rho',
                         width: float | str | units.UnitBase = 10.0,
                         resolution: int = None,
                         nx: int = None, ny: int = None, nz: int = None,
                         out_units: str | units.UnitBase = None,
                         weight: bool | str | np.ndarray | NoneType = None,
                         restrict_depth: bool = False,
                         kernel: NoneType | str | kernels.KernelBase = None,
                         smooth_floor: float = 0.0,
                         z_camera: float | NoneType = None,
                         threaded: bool | NoneType = None,
                         approximate_fast: bool | NoneType = None,
                         denoise: bool | NoneType = None,
                         grid_3d : bool = False
                         ) -> ImageRendererBase:
    """Generate a renderer object for rendering images of a simulation snapshot.

    Parameters
    ----------
    sim : snapshot.SimSnap
        The simulation snapshot to be rendered.

    quantity : str, numpy.ndarray
        The quantity to be rendered. If a string, the quantity is taken from the simulation snapshot.
        Default is 'rho'.

    width : float, str, units.UnitBase
        The width of the image to be rendered. If a string or unit, the value will be converted to the units
        of the simulation snapshot.

    resolution : int, optional
        The resolution of the image to be rendered, in pixels. The default is None, in which case the
        default resolution from the configuration file is used.

    nx : int, optional
        The x resolution of the image to be rendered, in pixels. The default is None, in which case the
        the resolution keyword is used instead.

    ny : int, optional
        The y resolution of the image to be rendered, in pixels. The default is None, in which case the
        the resolution keyword is used instead.

    nz : int, optional
        The z resolution of the image to be rendered, in pixels (for 3d-grid renderers only). The default is None,
        in which case the the resolution keyword is used instead.

    out_units : str, units.UnitBase, optional
        The units to be used for the output image. These are checked for compatibility with the array to be
        rendered, either in projection or in slice. If None, the output units are set to the units of the array to be
        rendered.

    weight: bool, str, numpy.ndarray, optional
        If True, the image is rendered as a volume-weighted projection. If a string or numpy array, the image is
        rendered as a weighted projection using the specified quantity. If None, the image is rendered as a simple
        slice or projection. The default is None.

    restrict_depth : bool, optional
        Whether to restrict the z range of the image to the same as the x range. The default is False.

    kernel : str, kernels.KernelBase, optional
        The kernel to be used for the image rendering. If None, the default kernel is assigned. For more information
        see :func:`pynbody.sph.kernels.create_kernel`.

    smooth_floor : float, str, units.UnitBase, optional
        The minimum smoothing length to be used in the image rendering, specified in units of the position array.
        Smoothing lengths below this will be boosted to this value. The default is 0.0, i.e. no manipulation of the
        smoothing takes place. This option is most useful for contour rendering, where excessive small-scale detail
        may be undesirable.

    smooth_min : float, str, units.UnitBase, optional
        The minimum smoothing length to be used in the image rendering, specified in units of the position array
        or as a unit. Smoothing lengths below this will be boosted to this value. The default is 0.0, i.e. no
        manipulation of the smoothing takes place. This option is most useful for contour rendering, where
        excessive small-scale detail may be undesirable.

    z_camera : float, optional
        The z position of the camera for the image to be rendered. The default is None. For more information on
        perspective rendering see :meth:`ImageGeometry.set_camera_z`.

    threaded : bool, optional
        Whether to render the image across multiple threads. Yes if true; no if false. The number of threads to be
        used is determined by the configuration file. If None, the use of threading is also determined by the
        configuration file.

    approximate_fast : bool, optional
        Whether to render the image using a lower-resolution approximation for large smoothing lengths. The default
        is None, in which case the use of approximation is determined by the configuration file.

    denoise : bool, optional
        Whether to include denoising in the rendering process. If None, denoising is applied only if the image
        is likely to benefit from it. If True, denoising is to be forced on the image; if that is actually
        impossible, this routine raises an exception. If False, denoising is never applied.

    grid_3d : bool, optional
        If True, the renderer will render a 3D grid instead of a 2D image. The default is False.

    """
    if resolution is None:
        resolution = config['image-default-resolution']

    if approximate_fast is None:
        approximate_fast = config_parser.getboolean('sph', 'approximate-fast-images')

    if threaded is None:
        threaded = config_parser.getboolean('sph', 'threaded-image')

    if isinstance(out_units, str):
        out_units = units.Unit(out_units)

    if isinstance(width, str):
        width = units.Unit(width)

    if units.is_unit(width):
        width = width.in_units(sim['pos'].units, **sim.conversion_context())

    width = float(width)

    if grid_3d:
        renderer = Grid3dRenderer(sim)
    else:
        renderer = ImageRenderer(sim)


    renderer.set_width(width)
    renderer.set_smooth_floor(smooth_floor)
    if restrict_depth:
        renderer.restrict_z_range()

    renderer.set_kernel(kernel)
    renderer.set_resolution(resolution)

    if nx is not None:
        renderer.geometry.nx = nx
    if ny is not None:
        renderer.geometry.ny = ny
        renderer.geometry.y1 *= ny/nx
        renderer.geometry.y2 *= ny/nx
    if nz is not None:
        renderer.geometry.nz = nz
        renderer.geometry.z1 *= nz/nx
        renderer.geometry.z2 *= nz/nx

    renderer.set_quantity(quantity)
    renderer.set_output_units(out_units)

    if z_camera is not None:
        renderer.geometry.set_camera_z(z_camera)

    if threaded:
        renderer = renderer.with_threading()

    if approximate_fast:
        renderer = renderer.with_approximate()

    renderer = renderer.with_denoising(denoise)

    if weight is True:
        renderer = renderer.with_volume_weighted_projection()
    elif weight is not None and weight is not False:
        renderer = renderer.with_weighted_projection(weight)

    return renderer
