"""
interpolate
===========

2D and 3D Interpolation routines written in cython

"""

import numpy as np

from . import _interpolate3d

# this just calls the cython interpolation function, setting the
# interpolation arrays to correct type


def interpolate3d(x, y, z, x_vals, y_vals, z_vals, vals):
    """
    Interpolate on a 3D regular grid.
    Yields results identical to scipy.interpolate.interpn.

    Input
    -----

    x,y,z : points where the interpolation will be performed

    x_vals, y_vals, z_vals : xyz values of the reference grid

    vals : grid values
    """

    # cast x_vals, y_vals and z_vals to float64

    x_vals = x_vals.astype(np.float64)
    y_vals = y_vals.astype(np.float64)
    z_vals = z_vals.astype(np.float64)
    vals = vals.astype(np.float64)

    result_array = np.empty(len(x), dtype=np.float64)

    _interpolate3d.interpolate3d(len(x),
                                 x, y, z,
                                 len(x_vals), x_vals,
                                 len(y_vals), y_vals,
                                 len(z_vals), z_vals,
                                 vals,
                                 result_array)

    return result_array


def interpolate2d(x, y, x_vals, y_vals, vals):
    """
    Interpolate on a 2D regular grid.
    Yields results identical to scipy.interpolate.interpn.

    Input
    -----

    x,y : points where the interpolation will be performed

    x_vals, y_vals : xy values of the reference grid

    vals : grid values
    """

    x_vals = x_vals.astype(np.float64)
    y_vals = y_vals.astype(np.float64)
    z_vals = np.ndarray(1, dtype=np.float64)

    vals = vals.astype(np.float64)
    vals.resize((1,) + vals.shape)

    result_array = np.empty(len(x), dtype=np.float64)

    _interpolate3d.interpolate3d(len(x),
                                 np.ndarray(1, dtype=np.float64), x, y,
                                 0, np.ndarray(1, dtype=np.float64),
                                 len(x_vals), x_vals,
                                 len(y_vals), y_vals,
                                 vals,
                                 result_array)

    return result_array
