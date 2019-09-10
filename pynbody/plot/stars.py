"""

stars
=====

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

from ..analysis import profile, angmom, halo
from .. import filt, units, config, array
from .sph import image
from .. import units as _units

from ..sph import render_spherical_image
from ..sph import Kernel2D

import logging
logger = logging.getLogger('pynbody.plot.stars')


def bytscl(arr, mini=0, maxi=10000):
	X = (arr - mini) / (maxi - mini)
	X[X > 1] = 1
	X[X < 0] = 0
	return X


def nw_scale_rgb(r, g, b, scales=[4, 3.2, 3.4]):
	return r * scales[0], g * scales[1], b * scales[2]


def nw_arcsinh_fit(r, g, b, nonlinearity=3):
	radius = r + g + b
	val = np.arcsinh(radius * nonlinearity) / nonlinearity / radius
	return r * val, g * val, b * val


def combine(r, g, b, magnitude_range, brightest_mag=None, mollview=False):
	# flip sign so that brightest pixels have biggest value
	r = -r
	g = -g
	b = -b

	if brightest_mag is None:
		brightest_mag = []

		# find something close to the maximum that is not quite the maximum
		for x in r, g, b:
			if mollview:
				x_tmp = x.flatten()[x.flatten()<0]
				ordered = np.sort(x_tmp.data)
			else:   
				ordered = np.sort(x.flatten())
			brightest_mag.append(ordered[-len(ordered) // 5000])

		brightest_mag = max(brightest_mag)
	else:
		brightest_mag = -brightest_mag

	rgbim = np.zeros((r.shape[0], r.shape[1], 3))
	rgbim[:, :, 0] = bytscl(r, brightest_mag - magnitude_range, brightest_mag)
	rgbim[:, :, 1] = bytscl(g, brightest_mag - magnitude_range, brightest_mag)
	rgbim[:, :, 2] = bytscl(b, brightest_mag - magnitude_range, brightest_mag)
	return rgbim, -brightest_mag

def convert_to_mag_arcsec2(image, mollview=False):
	if not mollview:
		assert image.units=="pc^-2"
	pc2_to_sqarcsec = 2.3504430539466191e-09
	return -2.5*np.log10(image*pc2_to_sqarcsec)

def render(sim, filename=None,
		   r_band='i', g_band='v', b_band='u',
		   r_scale=0.5, g_scale=1.0, b_scale=1.0,
		   dynamic_range=2.0,
		   mag_range=None,
		   width=50,
		   resolution=500,
		   starsize=None,
		   plot=True, axes=None, ret_im=False, clear=True,
		   ret_range=False, with_dust=False, z_range=50.0):
	'''
	Make a 3-color image of stars.

	The colors are based on magnitudes found using stellar Marigo
	stellar population code.  If with_dust is True, a simple dust
	screening is applied.

	Returns: If ret_im=True, an NxNx3 array representing an RGB image

	**Optional keyword arguments:**

	   *filename*: string (default: None)
		 Filename to be written to (if a filename is specified)

	   *r_band*: string (default: 'i')
		 Determines which Johnston filter will go into the image red channel

	   *g_band*: string (default: 'v')
		 Determines which Johnston filter will go into the image green channel

	   *b_band*: string (default: 'b')
		 Determines which Johnston filter will go into the image blue channel

	   *r_scale*: float (default: 0.5)
		 The scaling of the red channel before channels are combined

	   *g_scale*: float (default: 1.0)
		 The scaling of the green channel before channels are combined

	   *b_scale*: float (default: 1.0)
		 The scaling of the blue channel before channels are combined

	   *width*: float in kpc (default:50)
		 Sets the size of the image field in kpc

	   *resolution*: integer (default: 500)
	     Sets the number of pixels on the side of the image

	   *starsize*: float in kpc (default: None)
		 If not None, sets the maximum size of stars in the image

	   *ret_im*: bool (default: False)
		 if True, the NxNx3 image array is returned

	   *ret_range*: bool (default: False)
		 if True, the range of the image in mag arcsec^-2 is returned.

	   *plot*: bool (default: True)
		 if True, the image is plotted

	   *axes*: matplotlib axes object (deault: None)
		 if not None, the axes object to plot to

	   *dynamic_range*: float (default: 2.0)
		 The number of dex in luminosity over which the image brightness ranges

	   *mag_range*: float, float (default: None)
		 If provided, the brightest and faintest surface brightnesses in the range,
		 in mag arcsec^-2. Takes precedence over dynamic_range.

	   *with_dust*: bool, (default: False) 
		 If True, the image is rendered with a simple dust screening model
		 based on Calzetti's law.

	   *z_range*: float, (default: 50.0)
		 If with_dust is True this parameter specifies the z range
		 over which the column density will be calculated.
		 The default value is 50 kpc.

	'''

	if isinstance(width, str) or issubclass(width.__class__, _units.UnitBase):
		if isinstance(width, str):
			width = _units.Unit(width)
		width = width.in_units(sim['pos'].units, **sim.conversion_context())

	if starsize is not None:
		smf = filt.HighPass('smooth', str(starsize) + ' kpc')
		sim.s[smf]['smooth'] = array.SimArray(starsize, 'kpc', sim=sim)

	r = image(sim.s, qty=r_band + '_lum_den', width=width, log=False,
			  units="pc^-2", clear=False, noplot=True, resolution=resolution) * r_scale
	g = image(sim.s, qty=g_band + '_lum_den', width=width, log=False,
			  units="pc^-2", clear=False, noplot=True, resolution=resolution) * g_scale
	b = image(sim.s, qty=b_band + '_lum_den', width=width, log=False,
			  units="pc^-2", clear=False, noplot=True, resolution=resolution) * b_scale

	# convert all channels to mag arcsec^-2

	r=convert_to_mag_arcsec2(r)
	g=convert_to_mag_arcsec2(g)
	b=convert_to_mag_arcsec2(b)

	if with_dust is True:
		# render image with a simple dust absorption correction based on Calzetti's law using the gas content.
		try:
			import extinction                  
		except ImportError:
			warnings.warn(
				"Could not load extinction package. If you want to use this feature, "
				"plaese install the extinction package from here: http://extinction.readthedocs.io/en/latest/" 
				"or <via pip install extinction> or <conda install -c conda-forge extinction>", RuntimeWarning)
			return

		warm = filt.HighPass('temp',3e4)
		cool = filt.LowPass('temp',3e4)
		positive = filt.BandPass('z',-z_range,z_range) #LowPass('z',0)

		column_den_warm = image(sim.g[positive][warm], qty='rho', width=width, log=False,
			  units="kg cm^-2", clear=False, noplot=True,z_camera=z_range)
		column_den_cool = image(sim.g[positive][cool], qty='rho', width=width, log=False,
			  units="kg cm^-2", clear=False, noplot=True,z_camera=z_range)
		mh = 1.67e-27 # units kg

		cool_fac = 0.25 # fudge factor to make dust absorption not too strong
		# get the column density of gas
		col_den = np.divide(column_den_warm,mh)+np.divide(column_den_cool*cool_fac,mh)
		# get absorption coefficient
		a_v = 0.5*col_den*2e-21

		#get the central wavelength for the given band
		wavelength_avail = {'u':3571,'b':4378,'v':5466,'r':6695,'i':8565,'j':12101,
				   'h':16300,'k':21900,'U':3571,'B':4378,'V':5466,'R':6695,'I':8565,'J':12101,
				   'H':16300,'K':21900} #in Angstrom
		# effective wavelength taken from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=Generic&gname2=Johnson
		# and from https://en.wikipedia.org/wiki/Photometric_system for h, k
		
		lr,lg,lb = wavelength_avail[r_band],wavelength_avail[g_band],wavelength_avail[b_band] #in Angstrom
		wave = np.array([lb, lg, lr])

		ext_r = np.empty_like(r)
		ext_g = np.empty_like(g)
		ext_b = np.empty_like(b)
	
		for i in range(len(a_v)):
			for j in range(len(a_v[0])):
				ext = extinction.calzetti00(wave.astype(np.float), a_v[i][j].astype(np.float), 3.1, unit='aa', out=None)
				ext_r[i][j] = ext[2]
				ext_g[i][j] = ext[1]
				ext_b[i][j] = ext[0]

		r = r+ext_r
		g = g+ext_g
		b = b+ext_b

	#r,g,b = nw_scale_rgb(r,g,b)
	#r,g,b = nw_arcsinh_fit(r,g,b)

	if mag_range is None:
		rgbim, mag_max = combine(r, g, b, dynamic_range*2.5)
		mag_min = mag_max + 2.5*dynamic_range
	else:
		mag_max, mag_min = mag_range
		rgbim, mag_max = combine(r, g, b, mag_min - mag_max, mag_max)

	if plot:
		if clear:
			plt.clf()
		if axes is None:
			axes = plt.gca()

		if axes:
			axes.imshow(
				rgbim[::-1, :], extent=(-width / 2, width / 2, -width / 2, width / 2))
			axes.set_xlabel('x [' + str(sim.s['x'].units) + ']')
			axes.set_ylabel('y [' + str(sim.s['y'].units) + ']')
			plt.draw()

	if filename:
		plt.savefig(filename)

	if ret_im:
		return rgbim

	if ret_range:
		return mag_max, mag_min


def mollview(map=None,fig=None,plot=False,filenme=None,
			 rot=None,coord=None,unit='',
			 xsize=800,title='Mollweide view',nest=False,
			 min=None,max=None,flip='astro',
			 remove_dip=False,remove_mono=False,
			 gal_cut=0,
			 format='%g',format2='%g',
			 cbar=True,cmap=None, notext=False,
			 norm=None,hold=False,margins=None,sub=None,
			 return_projected_map=False):
	"""Plot an healpix map (given as an array) in Mollweide projection.
	   Requires the healpy package.

	   This function is taken from the Healpy package and slightly modified.
	
	Parameters
	----------
	map : float, array-like or None
	  An array containing the map, supports masked maps, see the `ma` function.
	  If None, will display a blank map, useful for overplotting.
	fig : figure object or None, optional
	  The figure to use. Default: create a new figure
	plot : bool (default: False)
	  if True the image is plotted 
	filename : string (default: None)
		 Filename to be written to (if a filename is specified)
	rot : scalar or sequence, optional
	  Describe the rotation to apply.
	  In the form (lon, lat, psi) (unit: degrees) : the point at
	  longitude *lon* and latitude *lat* will be at the center. An additional rotation
	  of angle *psi* around this direction is applied.
	coord : sequence of character, optional
	  Either one of 'G', 'E' or 'C' to describe the coordinate
	  system of the map, or a sequence of 2 of these to rotate
	  the map from the first to the second coordinate system.
	unit : str, optional
	  A text describing the unit of the data. Default: ''
	xsize : int, optional
	  The size of the image. Default: 800
	title : str, optional
	  The title of the plot. Default: 'Mollweide view'
	nest : bool, optional
	  If True, ordering scheme is NESTED. Default: False (RING)
	min : float, optional
	  The minimum range value
	max : float, optional
	  The maximum range value
	flip : {'astro', 'geo'}, optional
	  Defines the convention of projection : 'astro' (default, east towards left, west towards right)
	  or 'geo' (east towards right, west towards left)
	remove_dip : bool, optional
	  If :const:`True`, remove the dipole+monopole
	remove_mono : bool, optional
	  If :const:`True`, remove the monopole
	gal_cut : float, scalar, optional
	  Symmetric galactic cut for the dipole/monopole fit.
	  Removes points in latitude range [-gal_cut, +gal_cut]
	format : str, optional
	  The format of the scale label. Default: '%g'
	format2 : str, optional
	  Format of the pixel value under mouse. Default: '%g'
	cbar : bool, optional
	  Display the colorbar. Default: True
	notext : bool, optional
	  If True, no text is printed around the map
	norm : {'hist', 'log', None}
	  Color normalization, hist= histogram equalized color mapping,
	  log= logarithmic color mapping, default: None (linear color mapping)
	hold : bool, optional
	  If True, replace the current Axes by a MollweideAxes.
	  use this if you want to have multiple maps on the same
	  figure. Default: False
	sub : int, scalar or sequence, optional
	  Use only a zone of the current figure (same syntax as subplot).
	  Default: None
	margins : None or sequence, optional
	  Either None, or a sequence (left,bottom,right,top)
	  giving the margins on left,bottom,right and top
	  of the axes. Values are relative to figure (0-1).
	  Default: None
	return_projected_map : bool
	  if True returns the projected map in a 2d numpy array
	See Also
	--------
	gnomview, cartview, orthview, azeqview
	"""
	try:
		from healpy import projaxes as PA
		from healpy import pixelfunc                 
	except ImportError:
		warnings.warn(
			"Could not load healpy package. If you want to use this feature, "
			"plaese install the healpy package from here: http://healpy.readthedocs.io/en/latest/" 
			"or via pip or conda.", RuntimeWarning)
		return

	# Create the figure
	
	if not (hold or sub):
		if fig == None:
			f=plt.figure(figsize=(8.5,5.4))
			extent = (0.02,0.05,0.96,0.9)
		else:
			f=fig
			extent = (0.02,0.05,0.96,0.9)
	elif hold:
		f=plt.gcf()
		left,bottom,right,top = np.array(f.gca().get_position()).ravel()
		extent = (left,bottom,right-left,top-bottom)
		f.delaxes(f.gca())
	else: # using subplot syntax
		f=plt.gcf()
		if hasattr(sub,'__len__'):
			nrows, ncols, idx = sub
		else:
			nrows, ncols, idx = sub//100, (sub%100)//10, (sub%10)
		if idx < 1 or idx > ncols*nrows:
			raise ValueError('Wrong values for sub: %d, %d, %d'%(nrows,
															 ncols,
															 idx))
		c,r = (idx-1)%ncols,(idx-1)//ncols
		if not margins:
			margins = (0.01,0.0,0.0,0.02)
		extent = (c*1./ncols+margins[0], 
			  1.-(r+1)*1./nrows+margins[1],
			  1./ncols-margins[2]-margins[0],
			  1./nrows-margins[3]-margins[1])
		extent = (extent[0]+margins[0],
			  extent[1]+margins[1],
			  extent[2]-margins[2]-margins[0],
			  extent[3]-margins[3]-margins[1])

	# Starting to draw : turn interactive off
	wasinteractive = plt.isinteractive()
	plt.ioff()
	try:
		if map is None:
			map = np.zeros(12)+np.inf
			cbar=False
		map = pixelfunc.ma_to_array(map)
		ax=PA.HpxMollweideAxes(f,extent,coord=coord,rot=rot,
						   format=format2,flipconv=flip)
		f.add_axes(ax)
		if remove_dip:
			map=pixelfunc.remove_dipole(map,gal_cut=gal_cut,
									nest=nest,copy=True,
									verbose=True)
		elif remove_mono:
			map=pixelfunc.remove_monopole(map,gal_cut=gal_cut,nest=nest,
									  copy=True,verbose=True)
		img = ax.projmap(map,nest=nest,xsize=xsize,coord=coord,vmin=min,vmax=max,
			   cmap=cmap,norm=norm)
		if cbar:
			im = ax.get_images()[0]
			b = im.norm.inverse(np.linspace(0,1,im.cmap.N+1))
			v = np.linspace(im.norm.vmin,im.norm.vmax,im.cmap.N)
			if matplotlib.__version__ >= '0.91.0':
				cb=f.colorbar(im,ax=ax,
						  orientation='horizontal',
						  shrink=0.5,aspect=25,ticks=PA.BoundaryLocator(),
						  pad=0.05,fraction=0.1,boundaries=b,values=v,
						  format=format)
			else:
				# for older matplotlib versions, no ax kwarg
				cb=f.colorbar(im,orientation='horizontal',
						  shrink=0.5,aspect=25,ticks=PA.BoundaryLocator(),
						  pad=0.05,fraction=0.1,boundaries=b,values=v,
						  format=format)
			cb.solids.set_rasterized(True)
		ax.set_title(title)
		if not notext:
			ax.text(0.86,0.05,ax.proj.coordsysstr,fontsize=14,
				fontweight='bold',transform=ax.transAxes)
		if cbar:
			cb.ax.text(0.5,-1.0,unit,fontsize=14,
				   transform=cb.ax.transAxes,ha='center',va='center')
		f.sca(ax)
	finally:
		if plot:
			plt.draw()
		if wasinteractive:
			plt.ion()
			#plt.show()
	if return_projected_map:
		return img



def render_mollweide(sim, filename=None,
		   r_band='i', g_band='v', b_band='u',
		   r_scale=0.5, g_scale=1.0, b_scale=1.0,
		   dynamic_range=2.0,
		   mag_range=None,
		   width=25,
		   nside=128,
		   starsize=None,
		   plot=True, axes=None, ret_im=False, clear=True,
		   ret_range=False):
	'''
	Make a 3-color all-sky image of stars in a mollweide projection.
	Adapted from the function pynbody.plot.stars.render 

	The colors are based on magnitudes found using stellar Marigo
	stellar population code.  However there is no radiative transfer
	to account for dust.

	Returns: If ret_im=True, an NxNx3 array representing an RGB image

	**Optional keyword arguments:**

	   *filename*: string (default: None)
		 Filename to be written to (if a filename is specified)

	   *r_band*: string (default: 'i')
		 Determines which Johnston filter will go into the image red channel

	   *g_band*: string (default: 'v')
		 Determines which Johnston filter will go into the image green channel

	   *b_band*: string (default: 'b')
		 Determines which Johnston filter will go into the image blue channel

	   *r_scale*: float (default: 0.5)
		 The scaling of the red channel before channels are combined

	   *g_scale*: float (default: 1.0)
		 The scaling of the green channel before channels are combined

	   *b_scale*: float (default: 1.0)
		 The scaling of the blue channel before channels are combined

	   *width*: float in kpc (default:50)
		 Sets the size of the image field in kpc

	   *starsize*: float in kpc (default: None)
		 If not None, sets the maximum size of stars in the image

	   *ret_im*: bool (default: False)
		 if True, the NxNx3 image array is returned

	   *ret_range*: bool (default: False)
		 if True, the range of the image in mag arcsec^-2 is returned.

	   *plot*: bool (default: True)
		 if True, the image is plotted

	   *axes*: matplotlib axes object (deault: None)
		 if not None, the axes object to plot to

	   *dynamic_range*: float (default: 2.0)
		 The number of dex in luminosity over which the image brightness ranges

	   *mag_range*: float, float (default: None)
		 If provided, the brightest and faintest surface brightnesses in the range,
		 in mag arcsec^-2. Takes precedence over dynamic_range.
	'''
	
	if isinstance(width, str) or issubclass(width.__class__, _units.UnitBase):
		if isinstance(width, str):
			width = _units.Unit(width)
		width = width.in_units(sim['pos'].units, **sim.conversion_context())

	if starsize is not None:
		smf = filt.HighPass('smooth', str(starsize) + ' kpc')
		sim.s[smf]['smooth'] = array.SimArray(starsize, 'kpc', sim=sim)


	r = render_spherical_image(sim.s, qty=r_band + '_lum_den', nside=nside, distance=width, kernel=Kernel2D(),kstep=0.5, denoise=None, out_units="pc^-2", threaded=False)# * r_scale
	r = mollview(r,return_projected_map=True) * r_scale
	f=plt.gcf()
	g = render_spherical_image(sim.s, qty=g_band + '_lum_den', nside=nside, distance=width, kernel=Kernel2D(),kstep=0.5, denoise=None, out_units="pc^-2", threaded=False)# * g_scale
	g = mollview(g,return_projected_map=True,fig=f) * g_scale
	f=plt.gcf()
	b = render_spherical_image(sim.s, qty=b_band + '_lum_den', nside=nside, distance=width, kernel=Kernel2D(),kstep=0.5, denoise=None, out_units="pc^-2", threaded=False)# * b_scale
	b = mollview(b,return_projected_map=True,fig=f) * b_scale
	# convert all channels to mag arcsec^-2
	
	r=convert_to_mag_arcsec2(r, mollview=True)
	g=convert_to_mag_arcsec2(g, mollview=True)
	b=convert_to_mag_arcsec2(b, mollview=True)
	
	if mag_range is None:
		rgbim, mag_max = combine(r, g, b, dynamic_range*2.5, mollview=True)
		mag_min = mag_max + 2.5*dynamic_range
	else:
		mag_max, mag_min = mag_range
		rgbim, mag_max = combine(r, g, b, mag_min - mag_max, mag_max, mollview=True)

	if plot:
		if clear:
			plt.clf()
		if axes is None:
			axes = plt.gca()

		if axes:
			axes.imshow(
				rgbim[::-1, :])#, extent=(-width / 2, width / 2, -width / 2, width / 2)
			axes.axis('off')
			plt.draw()

	if filename:
		plt.savefig(filename)

	if ret_im:
		return rgbim

	if ret_range:
		return mag_max, mag_min


def sfh(sim, filename=None, massform=True, clear=False, legend=False,
		subplot=False, trange=False, bins=100, **kwargs):
	'''
	star formation history

	**Optional keyword arguments:**

	   *trange*: list, array, or tuple
		 size(t_range) must be 2. Specifies the time range.

	   *bins*: int
		 number of bins to use for the SFH

	   *massform*: bool
		 decides whether to use original star mass (massform) or final star mass

	   *subplot*: subplot object
		 where to plot SFH

	   *legend*: boolean
		 whether to draw a legend or not

	   *clear*: boolean
		 if False (default), plot on the current axes. Otherwise, clear the figure first.

	By default, sfh will use the formation mass of the star.  In tipsy, this will be
	taken from the starlog file.  Set massform=False if you want the final (observed)
	star formation history

	**Usage:**

	>>> import pynbody.plot as pp
	>>> pp.sfh(s,linestyle='dashed',color='k')


	'''
	import matplotlib.pyplot as pyplot

	if subplot:
		plt = subplot
	else:
		plt = pyplot

	if "nbins" in kwargs:
		bins = kwargs['nbins']

	if 'nbins' in kwargs:
		bins = kwargs['nbins']
		del kwargs['nbins']

	if ((len(sim.g)>0) | (len(sim.d)>0)): simstars = sim.star
	else: simstars = sim

	if trange:
		assert len(trange) == 2
	else:
		trange = [simstars['tform'].in_units(
			"Gyr").min(), simstars['tform'].in_units("Gyr").max()]
	binnorm = 1e-9 * bins / (trange[1] - trange[0])

	trangefilt = filt.And(filt.HighPass('tform', str(trange[0]) + ' Gyr'),
						  filt.LowPass('tform', str(trange[1]) + ' Gyr'))
	tforms = simstars[trangefilt]['tform'].in_units('Gyr')

	if massform:
		try:
			weight = simstars[trangefilt][
				'massform'].in_units('Msol') * binnorm
		except (KeyError, units.UnitsException):
			warnings.warn(
				"Could not load massform array -- falling back to current stellar masses", RuntimeWarning)
			weight = simstars[trangefilt]['mass'].in_units('Msol') * binnorm
	else:
		weight = simstars[trangefilt]['mass'].in_units('Msol') * binnorm

	if clear:
		plt.clf()
	sfhist, thebins, patches = plt.hist(tforms, weights=weight, bins=bins,
										histtype='step', **kwargs)
	if not subplot:
		# don't set the limits
		#plt.ylim(0.0, 1.2 * np.max(sfhist))
		plt.xlabel('Time [Gyr]', fontsize='large')
		plt.ylabel('SFR [M$_\odot$ yr$^{-1}$]', fontsize='large')
	else:
		plt.set_ylim(0.0, 1.2 * np.max(sfhist))

	# Make both axes have the same start and end point.
	if subplot:
		x0, x1 = plt.get_xlim()
	else:
		x0, x1 = plt.gca().get_xlim()


	# add a z axis on top if it has not been already done by an earlier plot:
	from pynbody.analysis import pkdgrav_cosmo as cosmo
	c = cosmo.Cosmology(sim=sim)

	old_axis = pyplot.gca()

	pz = plt.twiny()
	labelzs = np.arange(5, int(sim.properties['z']) - 1, -1)
	times = [13.7 * c.Exp2Time(1.0 / (1 + z)) / c.Exp2Time(1) for z in labelzs]
	pz.set_xticks(times)
	pz.set_xticklabels([str(x) for x in labelzs])
	pz.set_xlim(x0, x1)
	pz.set_xlabel('$z$')
	pyplot.sca(old_axis)

	if legend:
		plt.legend(loc=1)
	if filename:
		logger.info("Saving %s", filename)
		plt.savefig(filename)

	return array.SimArray(sfhist, "Msol yr**-1"), array.SimArray(thebins, "Gyr")


def schmidtlaw(sim, center=True, filename=None, pretime='50 Myr',
			   diskheight='3 kpc', rmax='20 kpc', compare=True,
			   radial=True, clear=True, legend=True, bins=10, **kwargs):
	'''Schmidt Law

	Plots star formation surface density vs. gas surface density including
	the observed relationships.  Currently, only plots densities found in
	radial annuli.

	**Usage:**

	>>> import pynbody.plot as pp
	>>> pp.schmidtlaw(h[1])

	**Optional keyword arguments:**

	   *center*: bool
		 center and align the input simulation faceon.

	   *filename*: string
		 Name of output file

	   *pretime* (default='50 Myr'): age of stars to consider for SFR

	   *diskheight* (default='3 kpc'): height of gas and stars above
		  and below disk considered for SF and gas densities.

	   *rmax* (default='20 kpc'): radius of disk considered

	   *compare* (default=True):  whether to include Kennicutt (1998) and
			Bigiel+ (2008) for comparison

	   *radial* (default=True):  should bins be annuli or a rectangular grid?

	   *bins* (default=10):  How many radial bins should there be?

	   *legend*: boolean
		 whether to draw a legend or not
	'''

	if not radial:
		raise NotImplementedError("Sorry, only radial Schmidt law currently supported")

	if center:
		angmom.faceon(sim)

	if isinstance(pretime, str):
		pretime = units.Unit(pretime)

	# select stuff
	diskgas = sim.gas[filt.Disc(rmax, diskheight)]
	diskstars = sim.star[filt.Disc(rmax, diskheight)]

	youngstars = np.where(diskstars['tform'].in_units("Myr") >
						  sim.properties['time'].in_units(
							  "Myr", **sim.conversion_context())
						  - pretime.in_units('Myr'))[0]

	# calculate surface densities
	if radial:
		ps = profile.Profile(diskstars[youngstars], nbins=bins)
		pg = profile.Profile(diskgas, nbins=bins)
	else:
		# make bins 2 kpc
		nbins = rmax * 2 / binsize
		pg, x, y = np.histogram2d(diskgas['x'], diskgas['y'], bins=nbins,
								  weights=diskgas['mass'],
								  range=[(-rmax, rmax), (-rmax, rmax)])
		ps, x, y = np.histogram2d(diskstars[youngstars]['x'],
								  diskstars[youngstars]['y'],
								  weights=diskstars['mass'],
								  bins=nbins, range=[(-rmax, rmax), (-rmax, rmax)])

	if clear:
		plt.clf()

	plt.loglog(pg['density'].in_units('Msol pc^-2'),
			   ps['density'].in_units('Msol kpc^-2') / pretime / 1e6, "+",
			   **kwargs)

	if compare:
		xsigma = np.logspace(np.log10(pg['density'].in_units('Msol pc^-2')).min(),
							 np.log10(
								 pg['density'].in_units('Msol pc^-2')).max(),
							 100)
		ysigma = 2.5e-4 * xsigma ** 1.5        # Kennicutt (1998)
		xbigiel = np.logspace(1, 2, 10)
		ybigiel = 10. ** (-2.1) * xbigiel ** 1.0   # Bigiel et al (2007)
		plt.loglog(xsigma, ysigma, label='Kennicutt (1998)')
		plt.loglog(
			xbigiel, ybigiel, linestyle="dashed", label='Bigiel et al (2007)')

	plt.xlabel('$\Sigma_{gas}$ [M$_\odot$ pc$^{-2}$]')
	plt.ylabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
	if legend:
		plt.legend(loc=2)
	if (filename):
		logger.info("Saving %s", filename)
		plt.savefig(filename)


def oneschmidtlawpoint(sim, center=True, pretime='50 Myr',
					   diskheight='3 kpc', rmax='20 kpc', **kwargs):
	'''One Schmidt Law Point

	Determines values for star formation surface density and gas surface
	density for the entire galaxy based on the half mass cold gas radius.

	**Usage:**
	import pynbody.plot as pp
	pp.oneschmidtlawpoint(h[1])

	   *pretime* (default='50 Myr'): age of stars to consider for SFR

	   *diskheight* (default='3 kpc'): height of gas and stars above
		  and below disk considered for SF and gas densities.

	   *rmax* (default='20 kpc'): radius of disk considered
	'''

	if center:
		angmom.faceon(sim)

	cg = h[1].gas[filt.LowPass('temp', 1e5)]
	cgd = cg[filt.Disc('30 kpc', '3 kpc')]
	cgs = np.sort(cgd['rxy'].in_units('kpc'))
	rhgas = cgs[len(cgs) / 2]
	instars = h[1].star[filt.Disc(str(rhgas) + ' kpc', '3 kpc')]
	minstars = np.sum(
		instars[filt.LowPass('age', '100 Myr')]['mass'].in_units('Msol'))
	ingasm = np.sum(
		cg[filt.Disc(str(rhgas) + ' kpc', '3 kpc')]['mass'].in_units('Msol'))
	rpc = rhgas * 1000.0
	rkpc = rhgas
	xsigma = ingasm / (np.pi * rpc * rpc)
	ysigma = minstars / (np.pi * rkpc * rkpc * 1e8)
	return xsigma, ysigma


def satlf(sim, band='v', filename=None, MWcompare=True, Trentham=True,
		  clear=True, legend=True,
		  label='Simulation', **kwargs):
	'''

	satellite luminosity function

	**Options:**

	*band* ('v'): which Johnson band to use. available filters: U, B,
	V, R, I, J, H, K

	*filename* (None): name of file to which to save output

	*MWcompare* (True): whether to plot comparison lines to MW

	*Trentham* (True): whether to plot comparison lines to Trentham +
					 Tully (2009) combined with Koposov et al (2007)

	By default, satlf will use the formation mass of the star.  In
	tipsy, this will be taken from the starlog file.

	**Usage:**

	>>> import pynbody.plot as pp
	>>> h = s.halos()
	>>> pp.satlf(h[1],linestyle='dashed',color='k')


	'''
	from ..analysis import luminosity as lum
	import os

	halomags = []
	# try :
	for haloid in sim.properties['children']:
		if (sim._halo_catalogue.contains(haloid)):
			halo = sim._halo_catalogue[haloid]
			try:
				halo.properties[band + '_mag'] = lum.halo_mag(halo, band=band)
				if np.isfinite(halo.properties[band + '_mag']):
					halomags.append(halo.properties[band + '_mag'])
			except IndexError:
				pass  # no stars in satellite
	# except KeyError:
		#raise KeyError, str(sim)+' properties have no children key as a halo type would'

	if clear:
		plt.clf()
	plt.semilogy(sorted(halomags), np.arange(len(halomags)) + 1, label=label,
				 **kwargs)
	plt.xlabel('M$_{' + band + '}$')
	plt.ylabel('Cumulative LF')
	if MWcompare:
		# compare with observations of MW
		tolfile = os.path.join(os.path.dirname(__file__), "tollerud2008mw")
		if os.path.exists(tolfile):
			tolmags = [float(q) for q in file(tolfile).readlines()]
		else:
			raise IOError(tolfile + " not found")
		plt.semilogy(sorted(tolmags), np.arange(len(tolmags)),
					 label='Milky Way')

	if Trentham:
		halomags = np.array(halomags)
		halomags = halomags[np.asarray(np.where(np.isfinite(halomags)))]
		xmag = np.linspace(halomags.min(), halomags.max(), 100)
		# Trentham + Tully (2009) equation 6
		# number of dwarfs between -11>M_R>-17 is well correlated with mass
		logNd = 0.91 * np.log10(sim.properties['mass']) - 10.2
		# set Nd from each equal to combine Trentham + Tully with Koposov
		coeff = 10.0 ** logNd / (10 ** -0.6 - 10 ** -1.2)

		# print 'Koposov coefficient:'+str(coeff)
		# Analytic expression for MW from Koposov
		#import pdb; pdb.set_trace()
		yn = coeff * 10 ** ((xmag + 5.0) / 10.0)  # Koposov et al (2007)
		plt.semilogy(xmag, yn, linestyle="dashed",
					 label='Trentham & Tully (2009)')

	if legend:
		plt.legend(loc=2)
	if (filename):
		logger.info("Saving %s", filename)
		plt.savefig(filename)


def sbprofile(sim, band='v', diskheight='3 kpc', rmax='20 kpc', binning='equaln',
			  center=True, clear=True, filename=None, axes=False, fit_exp=False,
			  print_ylabel=True, fit_sersic=False, **kwargs):
	'''

	surface brightness profile

	**Usage:**

	>>> import pynbody.plot as pp
	>>> h = s.halos()
	>>> pp.sbprofile(h[1],exp_fit=3,linestyle='dashed',color='k')

	**Options:**

	*band* ('v'): which Johnson band to use. available filters: U, B,
					 V, R, I, J, H, K

	*fit_exp*(False): Fits straight exponential line outside radius specified.

	*fit_sersic*(False): Fits Sersic profile outside radius specified.

	*diskheight('3 kpc')*
	*rmax('20 kpc')*:  Size of disk to be profiled

	*binning('equaln')*:  How show bin sizes be determined? based on
		  pynbody.analysis.profile

	*center(True)*:  Automatically align face on and center?

	*axes(False)*: In which axes (subplot) should it be plotted?

	*filename* (None): name of file to which to save output

	**needs a description of all keywords**

	By default, sbprof will use the formation mass of the star.
	In tipsy, this will be taken from the starlog file.

	'''

	if center:
		logger.info("Centering...")
		angmom.faceon(sim)

	logger.info("Selecting disk stars")
	diskstars = sim.star[filt.Disc(rmax, diskheight)]
	logger.info("Creating profile")
	ps = profile.Profile(diskstars, type=binning)
	logger.info("Plotting")
	r = ps['rbins'].in_units('kpc')

	if axes:
		plt = axes
	else:
		import matplotlib.pyplot as plt
	if clear:
		plt.clf()

	plt.plot(r, ps['sb,' + band], linewidth=2, **kwargs)
	if axes:
		plt.set_ylim(max(ps['sb,' + band]), min(ps['sb,' + band]))
	else:
		plt.ylim(max(ps['sb,' + band]), min(ps['sb,' + band]))
	if fit_exp:
		exp_inds = np.where(r.in_units('kpc') > fit_exp)
		expfit = np.polyfit(np.array(r[exp_inds]),
							np.array(ps['sb,' + band][exp_inds]), 1)

		# 1.0857 is how many magnitudes a 1/e decrease is
		print("h: ", 1.0857 / expfit[0], "  u_0:", expfit[1])

		fit = np.poly1d(expfit)
		if 'label' in kwargs:
			del kwargs['label']
		if 'linestyle' in kwargs:
			del kwargs['linestyle']
		plt.plot(r, fit(r), linestyle='dashed', **kwargs)
	if fit_sersic:
		sersic_inds = np.where(r.in_units('kpc') < fit_sersic)
		sersicfit = np.polyfit(np.log10(np.array(r[sersic_inds])),
							   np.array(ps['sb,' + band][sersic_inds]), 1)
		fit = np.poly1d(sersicfit)
		print("n: ", sersicfit[0], "  other: ", sersicfit[1])
		if 'label' in kwargs:
			del kwargs['label']
		if 'linestyle' in kwargs:
			del kwargs['linestyle']
		plt.plot(r, fit(r), linestyle='dashed', **kwargs)
		#import pdb; pdb.set_trace()
	if axes:
		if print_ylabel:
			plt.set_ylabel(band + '-band Surface brightness [mag as$^{-2}$]')
	else:
		plt.xlabel('R [kpc]')
		plt.ylabel(band + '-band Surface brightness [mag as$^{-2}$]')
	if filename:
		logger.info("Saving %s", filename)
		plt.savefig(filename)


def f(x, alpha, delta, g):
	return -np.log10(10.0 ** (x * alpha) + 1.0) + delta * (np.log10(1 + np.exp(x))) ** g / (1 + np.exp(10 ** -x))


def behroozi(xmasses, z, alpha=-1.412, Kravtsov=False):
	'''Based on Behroozi+ (2013) return what stellar mass corresponds to the
	halo mass passed in.

	**Usage**

	   >>> from pynbody.plot.stars import moster
	   >>> xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
	   >>> ystarmasses, errors = moster(xmasses,halo_catalog._halos[1].properties['z'])
	   >>> plt.fill_between(xmasses,np.array(ystarmasses)/np.array(errors),
						 y2=np.array(ystarmasses)*np.array(errors),
						 facecolor='#BBBBBB',color='#BBBBBB')
	'''
	loghm = np.log10(xmasses)
	# from Behroozi et al (2013)
	if Kravtsov: EPS = -1.642
	else: EPS = -1.777
	EPSpe = 0.133
	EPSme = 0.146

	EPSanu = -0.006
	EPSanupe = 0.113
	EPSanume = 0.361

	EPSznu = 0
	EPSznupe = 0.003
	EPSznume = 0.104

	EPSa = 0.119
	EPSape = 0.061
	EPSame = -0.012

	M1 = 11.514
	M1pe = 0.053
	M1me = 0.009

	M1a = -1.793
	M1ape = 0.315
	M1ame = 0.330

	M1z = -0.251
	M1zpe = 0.012
	M1zme = 0.125

	if Kravtsov: alpha=-1.779
	AL = alpha
	ALpe = 0.02
	ALme = 0.105

	ALa = 0.731
	ALape = 0.344
	ALame = 0.296

	if Kravtsov: DEL=4.394
	else: DEL = 3.508
	DELpe = 0.087
	DELme = 0.369

	DELa = 2.608
	DELape = 2.446
	DELame = 1.261

	DELz = -0.043
	DELzpe = 0.958
	DELzme = 0.071

	if Kravtsov: G=0.547
	else: G = 0.316
	Gpe = 0.076
	Gme = 0.012

	Ga = 1.319
	Gape = 0.584
	Game = 0.505

	Gz = 0.279
	Gzpe = 0.256
	Gzme = 0.081

	a = 1.0 / (z + 1.0)
	nu = np.exp(-4 * a ** 2)
	logm1 = M1 + nu * (M1a * (a - 1.0) + M1z * z)
	logeps = EPS + nu * (EPSanu * (a - 1.0) + EPSznu * z) - EPSa * (a - 1.0)
	analpha = AL + nu * ALa * (a - 1.0)
	delta = DEL + nu * DELa * (a - 1.0)
	g = G + nu * (Ga * (a - 1.0) + z * Gz)

	x = loghm - logm1
	f0 = -np.log10(2.0) + delta * np.log10(2.0) ** g / (1.0 + np.exp(1))
	smp = logm1 + logeps + f(x, analpha, delta, g) - f0

	if isinstance(smp, np.ndarray):
		scatter = np.zeros(len(smp))
	scatter = 0.218 - 0.023 * (a - 1.0)

	return 10 ** smp, 10 ** scatter


def moster(xmasses, z):
	'''Based on Moster+ (2013) return what stellar mass corresponds to the
	halo mass passed in.

	**Usage**

	   >>> from pynbody.plot.stars import moster
	   >>> xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
	   >>> ystarmasses, errors = moster(xmasses,halo_catalog._halos[1].properties['z'])
	   >>> plt.fill_between(xmasses,np.array(ystarmasses)/np.array(errors),
						 y2=np.array(ystarmasses)*np.array(errors),
						 facecolor='#BBBBBB',color='#BBBBBB')
	'''
	hmp = np.log10(xmasses)
	# from Moster et al (2013)
	M10a = 11.590470
	M11a = 1.194913
	R10a = 0.035113
	R11a = -0.024729
	B10a = 1.376177
	B11a = -0.825820
	G10a = 0.608170
	G11a = 0.329275

	M10e = 0.236067
	M11e = 0.353477
	R10e = 0.00577173
	R11e = 0.00693815
	B10e = 0.153
	B11e = 0.225
	G10e = 0.059
	G11e = 0.173

	a = 1.0 / (z + 1.0)
	m1 = M10a + M11a * (1.0 - a)
	r = R10a + R11a * (1.0 - a)
	b = B10a + B11a * (1.0 - a)
	g = G10a + G11a * (1.0 - a)
	smp = hmp + np.log10(2.0 * r) - np.log10((10.0 ** (hmp - m1)) ** (-b) + (10.0 ** (hmp - m1)) **
											 (g))
	eta = np.exp(np.log(10.) * (hmp - m1))
	alpha = eta ** (-b) + eta ** g
	dmdm10 = (g * eta ** g + b * eta ** (-b)) / alpha
	dmdm11 = (g * eta ** g + b * eta ** (-b)) / alpha * (1.0 - a)
	dmdr10 = np.log10(np.exp(1.0)) / r
	dmdr11 = np.log10(np.exp(1.0)) / r * (1.0 - a)
	dmdb10 = np.log10(np.exp(1.0)) / alpha * np.log(eta) * eta ** (-b)
	dmdb11 = np.log10(np.exp(1.0)) / alpha * \
		np.log(eta) * eta ** (-b) * (1.0 - a)
	dmdg10 = -np.log10(np.exp(1.0)) / alpha * np.log(eta) * eta ** g
	dmdg11 = -np.log10(np.exp(1.0)) / alpha * \
		np.log(eta) * eta ** g * (1.0 - a)
	sigma = np.sqrt(dmdm10 * dmdm10 * M10e * M10e + dmdm11 * dmdm11 * M11e * M11e + dmdr10 * dmdr10 * R10e * R10e + dmdr11 * dmdr11 * R11e * R11e + dmdb10 * dmdb10 * B10e * B10e + dmdb11 * dmdb11 * B11e
					* B11e + dmdg10 * dmdg10 * G10e * G10e + dmdg11 * dmdg11 * G11e * G11e)
	return 10 ** smp, 10 ** sigma

def behroozi(xmasses, z):
	'''Based on Behroozi+ (2013) return what stellar mass corresponds to the
	halo mass passed in.

	**Usage**

	   >>> from pynbody.plot.stars import moster
	   >>> xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
	   >>> ystarmasses, errors = moster(xmasses,halo_catalog._halos[1].properties['z'])
	   >>> plt.fill_between(xmasses,np.array(ystarmasses)/np.array(errors),
						 y2=np.array(ystarmasses)*np.array(errors),
						 facecolor='#BBBBBB',color='#BBBBBB')
	'''
	loghm = np.log10(xmasses)
	# from Behroozi et al (2013)
	EPS = -1.777
	EPSpe = 0.133
	EPSme = 0.146

	EPSanu = -0.006
	EPSanupe = 0.113
	EPSanume = 0.361

	EPSznu = 0
	EPSznupe = 0.003
	EPSznume = 0.104

	EPSa = 0.119
	EPSape = 0.061
	EPSame = -0.012

	M1 = 11.514
	M1pe = 0.053
	M1me = 0.009

	M1a = -1.793
	M1ape = 0.315
	M1ame = 0.330

	M1z = -0.251
	M1zpe = 0.012
	M1zme = 0.125

	AL = -1.412
	ALpe = 0.02
	ALme = 0.105

	ALa = 0.731
	ALape = 0.344
	ALame = 0.296

	DEL = 3.508
	DELpe = 0.087
	DELme = 0.369

	DELa = 2.608
	DELape = 2.446
	DELame = 1.261

	DELz = -0.043
	DELzpe = 0.958
	DELzme = 0.071

	G = 0.316
	Gpe = 0.076
	Gme = 0.012

	Ga = 1.319
	Gape = 0.584
	Game = 0.505

	Gz = 0.279
	Gzpe = 0.256
	Gzme = 0.081

	a = 1.0 / (z + 1.0)
	nu = np.exp(-4 * a ** 2)
	logm1 = M1 + nu * (M1a * (a - 1.0) + M1z * z)
	logeps = EPS + nu * (EPSanu * (a - 1.0) + EPSznu * z) - EPSa * (a - 1.0)
	alpha = AL + nu * ALa * (a - 1.0)
	delta = DEL + nu * DELa * (a - 1.0)
	g = G + nu * (Ga * (a - 1.0) + z * Gz)

	x = loghm - logm1
	f0 = -np.log10(2.0) + delta * np.log10(2.0) ** g / (1.0 + np.exp(1))
	smp = logm1 + logeps + f(x, alpha, delta, g) - f0

	if isinstance(smp, np.ndarray):
		scatter = np.zeros(len(smp))
	scatter = 0.218 - 0.023 * (a - 1.0)

	return 10 ** smp, 10 ** scatter

def subfindguo(halo_catalog, clear=False, compare=True, baryfrac=False,
		filename=False, **kwargs):
	'''Stellar Mass vs. Halo Mass

	Takes a halo catalogue and plots the member stellar masses as a
	function of halo mass.

	Usage:

	>>> import pynbody.plot as pp
	>>> h = s.halos()
	>>> pp.guo(h,marker='+',markerfacecolor='k')

	**Options:**

	*compare* (True): Should comparison line be plotted?
		 If compare = 'guo', Guo+ (2010) plotted instead of Behroozi+ (2013)

	*baryfrac* (False):  Should line be drawn for cosmic baryon fraction?

	*filename* (None): name of file to which to save output
	'''

	# if 'marker' not in kwargs :
	#    kwargs['marker']='o'

	starmasshalos = []
	totmasshalos = []
	f_b = halo_catalog[0].properties['omegaB0']/halo_catalog[0].properties['omegaM0'] 
	for halo in halo_catalog:
		for subhalo in halo.sub:
			subhalo.properties['MassType'].convert_units('Msol')
			halostarmass = subhalo.properties['MassType'][4]
			if halostarmass:
				starmasshalos.append(halostarmass)
				totmasshalos.append(np.sum(subhalo.properties['MassType']))

	if clear:
		plt.clf()

	plt.loglog(totmasshalos, starmasshalos, 'o', **kwargs)
	plt.xlabel('Total Halo Mass')
	plt.ylabel('Halo Stellar Mass')

	if compare:
		xmasses = np.logspace(
			np.log10(min(totmasshalos)), 1 + np.log10(max(totmasshalos)), 20)
		if compare == 'guo':
			# from Sawala et al (2011) + Guo et al (2009)
			ystarmasses = xmasses*0.129*((xmasses/2.5e11)**-0.926 + (xmasses/2.5e11)**0.261)**-2.44
		else :
			ystarmasses, errors = behroozi(xmasses,halo_catalog._halos[1].properties['z'])
		plt.fill_between(xmasses,np.array(ystarmasses)/np.array(errors),
						 y2=np.array(ystarmasses)*np.array(errors),
						 facecolor='#BBBBBB',color='#BBBBBB')
		plt.loglog(xmasses,ystarmasses,label='Behroozi et al (2013)')

	if baryfrac :
		xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
		ystarmasses = xmasses*f_b
		plt.loglog(xmasses,ystarmasses,linestyle='dotted',label='f_b = '+'%.2f' % f_b)
		ystarmasses = xmasses*0.1*f_b
		plt.loglog(xmasses,ystarmasses,linestyle='dashed',label='0.1 f_b = '+'%.2f' % (0.1*f_b))

	plt.axis([0.8*min(totmasshalos),1.2*max(totmasshalos),
			  0.8*min(starmasshalos),1.2*max(starmasshalos)])

	if (filename):
		logger.info("Saving %s", filename)
		plt.savefig(filename)

def guo(halo_catalog, clear=False, compare=True, baryfrac=False,
		filename=False, **kwargs):
	'''Stellar Mass vs. Halo Mass

	Takes a halo catalogue and plots the member stellar masses as a
	function of halo mass.

	Usage:

	>>> import pynbody.plot as pp
	>>> h = s.halos()
	>>> pp.guo(h,marker='+',markerfacecolor='k')

	**Options:**

	*compare* (True): Should comparison line be plotted?
		 If compare = 'guo', Guo+ (2010) plotted instead of Behroozi+ (2013)

	*baryfrac* (False):  Should line be drawn for cosmic baryon fraction?

	*filename* (None): name of file to which to save output
	'''

	# if 'marker' not in kwargs :
	#    kwargs['marker']='o'

	starmasshalos = []
	totmasshalos = []

	halo_catalog._halos[1]['mass'].convert_units('Msol')

	for i in np.arange(len(halo_catalog._halos)) + 1:
		halo = halo_catalog[i]
		halostarmass = np.sum(halo.star['mass'])
		if halostarmass:
			starmasshalos.append(halostarmass)
			totmasshalos.append(np.sum(halo['mass']))

	if clear:
		plt.clf()

	plt.loglog(totmasshalos, starmasshalos, 'o', **kwargs)
	plt.xlabel('Total Halo Mass')
	plt.ylabel('Halo Stellar Mass')

	if compare:
		xmasses = np.logspace(
			np.log10(min(totmasshalos)), 1 + np.log10(max(totmasshalos)), 20)
		if compare == 'guo':
			# from Sawala et al (2011) + Guo et al (2009)
			ystarmasses = xmasses*0.129*((xmasses/2.5e11)**-0.926 + (xmasses/2.5e11)**0.261)**-2.44
		else :
			ystarmasses, errors = behroozi(xmasses,halo_catalog._halos[1].properties['z'])
		plt.fill_between(xmasses,np.array(ystarmasses)/np.array(errors),
						 y2=np.array(ystarmasses)*np.array(errors),
						 facecolor='#BBBBBB',color='#BBBBBB')
		plt.loglog(xmasses,ystarmasses,label='Behroozi et al (2013)')

	if baryfrac :
		xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
		ystarmasses = xmasses*0.04/0.24
		plt.loglog(xmasses,ystarmasses,linestyle='dotted',label='f_b = 0.16')

	plt.axis([0.8*min(totmasshalos),1.2*max(totmasshalos),
			  0.8*min(starmasshalos),1.2*max(starmasshalos)])

	if (filename):
		logger.info("Saving %s", filename)
		plt.savefig(filename)
