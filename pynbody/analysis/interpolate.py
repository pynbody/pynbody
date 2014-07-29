from . import _interpolate3d
import numpy as np

# this just calls the cython interpolation function, setting the 
# interpolation arrays to correct type
def interpolate3d(x,y,z,x_vals,y_vals,z_vals,vals) :
	# cast x_vals, y_vals and z_vals to float64 

	x_vals = x_vals.astype(np.float64)
	y_vals = y_vals.astype(np.float64)
	z_vals = z_vals.astype(np.float64)
	vals = vals.astype(np.float64)

	result_array = np.empty(len(x),dtype=np.float64)

	_interpolate3d.interpolate3d(len(x), 
								 x, y, z, 
								 len(x_vals), x_vals, 
								 len(y_vals), y_vals, 
								 len(z_vals), z_vals, 
								 vals,
								 result_array)

	return result_array

def interpolate2d(x, y, x_vals, y_vals, vals) :
	x_vals = x_vals.astype(np.float64)
	y_vals = y_vals.astype(np.float64)
	z_vals = np.ndarray(1,dtype=np.float64)

	vals = vals.astype(np.float64)
	vals.resize((1,) + vals.shape)

	result_array = np.empty(len(x),dtype=np.float64)

	_interpolate3d.interpolate3d(len(x), 
								 np.ndarray(1,dtype=np.float64), x, y,
								 0, np.ndarray(1,dtype=np.float64),
								 len(x_vals), x_vals,
								 len(y_vals), y_vals,
								 vals,
								 result_array)

	return result_array
	