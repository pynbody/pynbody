"""
Routines for plots related to stellar particles

.. versionchanged:: 2.0

  *satlf*, *guo*, *subfindguo*, *moster* and *behroozi* have been removed; these routines were
  not being actively maintained and presented out-of-date data.

"""

import logging
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pynbody.analysis.luminosity

from .. import array, filt, transformation, units
from ..analysis import angmom, cosmology, profile
from ..sph import kernels, render_spherical_image, renderers
from . import sph as plot_sph

logger = logging.getLogger('pynbody.plot.stars')


def _bytscl(arr, mini=0, maxi=10000):
	X = (arr - mini) / (maxi - mini)
	X[X > 1] = 1
	X[X < 0] = 0
	return X


def _nw_scale_rgb(r, g, b, scales=[4, 3.2, 3.4]):
	return r * scales[0], g * scales[1], b * scales[2]


def _nw_arcsinh_fit(r, g, b, nonlinearity=3):
	radius = r + g + b
	val = np.arcsinh(radius * nonlinearity) / nonlinearity / radius
	return r * val, g * val, b * val


def _combine(r, g, b, magnitude_range, brightest_mag=None, masked=False):
	# flip sign so that brightest pixels have biggest value
	r = -r
	g = -g
	b = -b

	if brightest_mag is None:
		brightest_mag = []

		# find something close to the maximum that is not quite the maximum
		for x in r, g, b:
			if masked:
				x_tmp = x.flatten()[x.flatten()<0]
				ordered = np.sort(x_tmp.data)
			else:
				ordered = np.sort(x.flatten())
			brightest_mag.append(ordered[-len(ordered) // 5000])

		brightest_mag = max(brightest_mag)
	else:
		brightest_mag = -brightest_mag

	rgbim = np.stack([_bytscl(channel, brightest_mag - magnitude_range, brightest_mag)
					   for channel in (r, g, b)], axis=-1)
	return rgbim, -brightest_mag

def _convert_to_mag_arcsec2(image, angular=False):
	if not angular:
		assert image.units=="pc^-2"
	else:
		assert image.units=='pc^-2 sr^-1'

	pc2_to_sqarcsec = 2.3504430539466191e-09
	return -2.5*np.log10(image*pc2_to_sqarcsec)

def contour_surface_brightness(sim, band='v', width=50, resolution=None, axes=None, label=True,
								contour_kwargs=None, smooth_floor=0.0):
	"""Plot surface brightness contours in the given band.

	For information about how surface brightnesses are calculated, see the documentation for
	:mod:`pynbody.analysis.luminosity`.

	Parameters
	----------

	sim : SimSnap
		The simulation snapshot to plot.

	band : str
		The band to plot. Default is 'v'.

	width : float | str, optional
		The width of the image to produce

	resolution : int, optional
		The resolution of the image to produce. The default is determined by the configuration
		option ``image-default-resolution``.

	axes : matplotlib axes object, optional
		If not None, the axes object to plot to

	label : bool, optional
		If True, the contours are labelled with the surface brightness in mag arcsec^-2

	contour_kwargs : dict or None, optional
		Keyword arguments to pass to the matplotlib contour plot function. For example, this
		can be used to determine the contour levels.

	smooth_floor : float or str, optional
		The minimum size of the smoothing kernel, either as a float or a unit string.
		Setting this to a non-zero value makes smoother, clearer contours but loses fine detail.
		Default is 0.0.

	"""

	return plot_sph.contour(sim, qty=band + '_lum_den', units="pc^-2", width=width,
							resolution=resolution, axes=axes, label=label, smooth_floor=smooth_floor,
							_transform = _convert_to_mag_arcsec2,
							log=False, contour_kwargs=contour_kwargs)



def render(sim,
		   r_band='I', g_band='V', b_band='U',
		   width=50,
		   r_scale=0.5, g_scale=1.0, b_scale=1.0,
		   with_dust=False,
		   dynamic_range=2.0,
		   mag_range=None,
		   resolution=None,
		   noplot=False, return_image=False,
		   return_range=False):
	'''
	Make a 3-color image of stars.

	For information about how surface brightnesses are calculated, see the documentation for
	:mod:`pynbody.analysis.luminosity`.

	If ``with_dust`` is True, a simple dust screening is applied; see below for important notes
	on the dust screening.

	.. versionchanged:: 2.0

	  For consistency with other plotting routines, the *axes* and *clear* arguments have been
	  removed. Set up axes as needed using the standard matplotlib commands.

	  *ret_im* has been renamed to *return_image* for consistency with other plotting routines,
	  and *ret_range* has been renamed to *return_range*.

	  The dust screening model has been improved (though see important notes below on limitations),
	  and SSP tables are also considerably improved (see :mod:`pynbody.analysis.luminosity`).

	Parameters
	----------

	sim : SimSnap
		The simulation snapshot to plot.

	r_band, g_band, b_band : str
		The filter bands to use for R, G and B image channels. Default is 'I', 'V', 'U'. These bands
		are as defined in :mod:`pynbody.analysis.luminosity`, or overriden using the
		:func:`pynbody.analysis.luminosity.use_custom_ssp_table` function.

	width : float | str, optional
		The width of the image to produce, either as a float or a unit string.

	r_scale, g_scale, b_scale : float, optional
		The scaling of the red, green and blue channels before they are combined.

	dynamic_range : float, optional
		The number of dex in luminosity over which the image brightness ranges, if
		``mag_range`` is not provided.

	with_dust : bool, optional
		If True, the image is rendered with a simple dust screening model. See important notes below.

	mag_range : tuple, optional
		The brightest and faintest surface brightnesses in the final image, in
		mag arcsec^-2.

	resolution : int, optional
		The resolution of the image to produce. The default is determined by the configuration
		option ``image-default-resolution``.

	noplot : bool, optional
		If True, the image is not plotted; most useful alongside ``return_image``, if you want to save the
		image to a file.

	return_image : bool, optional
		If True, the image is returned as an array (N x N x 3) for the RGB channels. Default is False.

	return_range : bool, optional
		If True, a tuple with the range of the image in mag arcsec^-2 is returned. Default is False.


	Returns
	-------

	If ``return_image`` is True, an array (N x N x 3) representing the RGB image is returned.

	If ``return_range`` is True, a tuple with the range of the image in mag arcsec^-2 is returned.

	Notes
	-----

	The dust screening model is exceptionally simple and can only be used for indicative purposes.
	For more accurate results, radiative transfer is essential and is provided by other packages
	such as `skirt <https://skirt.ugent.be/root/_home.html>`_.

	The model assumes that the dust is proportional to the metal density. It estimates a V-band
	extinction A_V using empirical data from Draine & Lee (1984, ApJ, 285, 89) and Savage and Mathis
	(1979, ARA&A, 17, 73). This is then converted to extinction in the given bands using the
	Calzetti law (2000, ApJ, 533, 682) with an R_V of 3.1.

	The model furthermore assumes that half the dust is in front of the stars and half behind, because
	there is no radiative transfer to account for the actual distribution of dust in the 3d space.

	'''

	renderer = renderers.make_render_pipeline(sim.s, quantity=r_band + '_lum_den', width=width,
											  out_units="pc^-2", resolution=resolution)

	r = renderer.render() * r_scale
	renderer.set_quantity(g_band + '_lum_den')
	g = renderer.render() * g_scale
	renderer.set_quantity(b_band + '_lum_den')
	b = renderer.render() * b_scale


	# convert all channels to mag arcsec^-2

	r=_convert_to_mag_arcsec2(r)
	g=_convert_to_mag_arcsec2(g)
	b=_convert_to_mag_arcsec2(b)

	width = renderer.geometry.width

	if with_dust is True:
		# render image with a simple dust absorption correction based on Calzetti's law using the gas content.

		a_v = _dust_Av_image(sim, width, resolution)

		ext_b, ext_g, ext_r = _a_v_to_band_extinctions(a_v, b_band, g_band, r_band)

		r = r+ext_r
		g = g+ext_g
		b = b+ext_b

	if mag_range is None:
		rgbim, mag_max = _combine(r, g, b, dynamic_range * 2.5)
		mag_min = mag_max + 2.5*dynamic_range
	else:
		mag_max, mag_min = mag_range
		rgbim, mag_max = _combine(r, g, b, mag_min - mag_max, mag_max)

	if not noplot:
		axes = plt.gca()
		axes.imshow(
			rgbim[::-1, :], extent=(-width / 2, width / 2, -width / 2, width / 2))
		axes.set_xlabel('x [' + str(sim.s['x'].units) + ']')
		axes.set_ylabel('y [' + str(sim.s['y'].units) + ']')

	if return_image:
		return rgbim

	if return_range:
		return mag_max, mag_min


def _aa_to_invum(wavelengths):
	return 1e4 / wavelengths

def _calzetti00_invum(x, r_v):
	"""Calzetti extinction law in inverse microns, for specified r_v"""
	if x > 1.5873015873015872:
		k = 2.659 * (((0.011 * x - 0.198) * x + 1.509) * x - 2.156)
	else:
		k = 2.659 * (1.040*x - 1.857)

	return 1.0 + k / r_v

def _calzetti00(wavelengths, r_v):
	"""Calzetti extinction law, for specified r_v"""
	return _calzetti00_invum(_aa_to_invum(wavelengths), r_v)

def _a_v_to_band_extinctions(a_v, b_band, g_band, r_band, r_v=3.1):
	ssp_table = pynbody.analysis.luminosity.get_current_ssp_table()
	wavelengths = [ssp_table.get_central_wavelength(band) for band in (b_band, g_band, r_band)]

	extinction_per_av = [_calzetti00(wavelength, r_v) for wavelength in wavelengths]

	results = [a_v * ext for ext in extinction_per_av]

	return results


def _dust_Av_image(sim, width, resolution, healpix=False):
	"""Produce a map of the extinction Av for the given simulation, using the gas content.

	Note that the dust model is very simple and naive! See comments inline.
	"""

	# Assume that only the gas with z>0 absorbs light (i.e. 'all light' is produced in the midplane)
	gas = sim.g

	# calculate the density of metals, and assume that dust is proportional to the metal density
	rho_metals = gas['rho'] * gas['metals']
	rho_metals.units = gas['rho'].units

	if healpix:
		# healpix output is in units of X sr^-1 where the input quantity is in X kpc^-3. If X is the number of
		# absorbing grains, we get the number of absorbing grains per steradian; what we actually wanted was the
		# projected number density of absorbing grains along a pencil beam (X cm^-2) in that direction.
		# Since Ntot = int r^2 n(r) dr x omega, we need to divide by r^2 to get the projected density in cm^-2.
		rho_metals /= gas['r']**2
		renderer = renderers.make_render_pipeline(gas, quantity=rho_metals, out_units="m_p sr^-1 cm^-2",
												  nside=resolution, target='healpix')
	else:
		renderer = renderers.make_render_pipeline(gas, quantity=rho_metals, width=width, out_units="m_p cm^-2",
												  resolution=resolution)
	column_den = renderer.render()

	# From Draine & Lee (1984, ApJ, 285, 89) in the V band (lambda^-1 ~= 2 micron^-1), the optical
	# depth is 0.5 for an H column density of 10^21 cm^2. That scaling in turn is based on data in
	# the review of Savage and Mathis (1979, ARA&A, 17, 73). Amazingly, this is the most up-to-date
	# reference I could find on the subject. There is no mention of the metallicity dependence but
	# one would assume dust columns should scale at least with metal columns (perhaps even more
	# steeply with local metallicity). Given the wild approximations in all this, I assume a
	# metallicity of 0.02 for gas in the Milky Way

	tau_to_mag_extinction = 2.5 / np.log(10.)

	a_v = tau_to_mag_extinction * 0.5 * column_den / 1e21 / 0.02

	# Finally, we assume that half the dust is in front of the stars and half behind. This is
	# yet another big assumption!

	a_v/=2

	return a_v


def render_mollweide(sim,
					 r_band='I', g_band='V', b_band='U',
					 r_scale=0.5, g_scale=1.0, b_scale=1.0,
					 mag_range=None, dynamic_range=2.0, nside=None,
					 with_dust = False, noplot=False, return_image=False, return_range=False, xsize=1600):
	'''
	Make a 3-color all-sky image of stars in a mollweide projection, i.e. a projection of all angles around the origin.

	.. versionchanged:: 2.0

	  For consistency with other plotting routines, the *axes* and *clear* arguments have been removed.
	  Set up axes as needed using the standard matplotlib commands.

	  *ret_im* has been renamed to *return_image* for consistency with other plotting routines,
	  and *ret_range* has been renamed to *return_range*.

	  The *xside* option has been added.

	  The *with_dust* option has been added to mirror the functionality in :func:`~pynbody.plot.stars.render`.
	  However, see the parameter documentation below for important caveats on dust screening in this context.

	Parameters
	----------

	sim : SimSnap
		The simulation snapshot to plot.

	r_band, g_band, b_band : str
		The filter bands to use for R, G and B image channels. Default is 'I', 'V', 'U'. These bands
		are as defined in :mod:`pynbody.analysis.luminosity`, or overriden using the
		:func:`pynbody.analysis.luminosity.use_custom_ssp_table` function.

	r_scale, g_scale, b_scale : float, optional
		The scaling of the red, green and blue channels before they are combined.

	mag_range : tuple, optional
		The brightest and faintest surface brightnesses in the final image, in mag arcsec^-2.

	dynamic_range : float, optional
		The number of dex in luminosity over which the image brightness ranges, if
		``mag_range`` is not provided.

	nside : int, optional
		The healpix nside resolution to use (must be power of 2). Default is determined by pynbody config file.

	xsize : int, default 1600
		The *xsize* parameter for healpy.mollview, which determines the resolution of the projection (i.e. does not
		affect the resolution of the actual image, but the presentation.) Default is 1600.

	with_dust : bool, default False
		If True, the image is rendered with a simple dust screening model. See important notes on dust screening
		in :func:`~pynbody.plot.stars.render`. These are even more important in the case of rendering
		a full-sky image from within a galaxy, to the point where the results may carry little physical meaning.

	noplot : bool, optional
		If True, the image is not plotted; most useful alongside ``return_image``, if you want to save the
		image to a file.

	return_image : bool, optional
		If True, the image is returned as an array (N x N x 3) for the RGB channels. Default is False.

	return_range : bool, optional
		If True, a tuple with the range of the image in mag arcsec^-2 is returned. Default is False.

	Returns
	-------

	If ``return_image`` is True, an array (N x N x 3) representing the RGB image is returned.

	If ``return_range`` is True, a tuple with the range of the image in mag arcsec^-2 is returned.

	'''


	def _get_channel(band, scale):
		renderer = renderers.make_render_pipeline(sim.s, quantity=sim.s[band + '_lum_den']/sim.s['r']**2,
												  nside=nside, target='healpix', out_units="pc^-2 sr^-1")
		return _convert_to_mag_arcsec2(renderer.render() * scale, angular=True)

	def _project_channel(result_healpix):
		from healpy import projaxes
		ax = projaxes.HpxMollweideAxes(plt.gcf(), (0.02, 0.05, 0.96, 0.9))
		result = ax.projmap(result_healpix, nest=False, xsize=xsize)
		return result

	r = _get_channel(r_band, r_scale)
	g = _get_channel(g_band, g_scale)
	b = _get_channel(b_band, b_scale)

	if with_dust is True:
		# render image with a simple dust absorption correction based on Calzetti's law using the gas content.

		a_v = _dust_Av_image(sim, None, nside, healpix=True)

		ext_b, ext_g, ext_r = _a_v_to_band_extinctions(a_v, b_band, g_band, r_band)

		r = r+ext_r
		g = g+ext_g
		b = b+ext_b

	if mag_range is None:
		rgbim, mag_max = _combine(r, g, b, dynamic_range * 2.5)
		mag_min = mag_max + 2.5*dynamic_range
	else:
		mag_max, mag_min = mag_range
		rgbim, mag_max = _combine(r, g, b, mag_min - mag_max, mag_max)

	rgbim_projected = np.stack([_project_channel(x) for x in rgbim.T], axis=-1)
	rgbim_projected[rgbim_projected < 0] = 1.0

	if not noplot:
		axes = plt.gca()
		axes.imshow(rgbim_projected[::-1, :])
		axes.axis('off')

	if return_image and return_range:
		return rgbim, (mag_max, mag_min)
	elif return_image:
		return rgbim
	elif return_range:
		return mag_max, mag_min


def sfh(sim, massform=True, trange=None, bins=100, **kwargs):
	"""Make a star formation history plot.

	By default, sfh will use the formation mass of the star.  In tipsy, this will be
	taken from the starlog file.  Set ``massform=False`` if you want the final (observed)
	star formation history

	.. versionchanged:: 2.0
	  The *subplot*, *filename*, *legend* and *clear* arguments have been removed.
	  Set up axes as needed using the standard matplotlib commands.

	  The return arrays have been swapped (i.e. time bins are now the first return value),
	  to be consistent with other histogram routines.

	Parameters
	----------

	sim : SimSnap
		The simulation snapshot to plot.

	massform : bool, default True
		Decides whether to use original star mass (massform) or final star mass. Default is
		True. If True and the massform array cannot be found, the final star mass is used instead
		(and a warning issued).

	trange : list, array, or tuple, optional
		Specifies the time range over which to plot the SFH in Gyr.
		Default is the full range of the simulation.

	bins : int, default 100
		Number of bins to use for the SFH. Default is 100.

	**kwargs :
		Additional keyword arguments are passed to the matplotlib hist function.

	Returns
	-------

	tbins : array.SimArray
		Array of time bin edges in Gyr

	sfh : array.SimArray
		Array of star formation rates in Msol/yr

	"""

	import matplotlib.pyplot as pyplot

	simstars = sim.star

	tforms_gyr = simstars['tform'].in_units("Gyr")

	if trange:
		if len(trange) != 2:
			raise ValueError("trange must be a list or tuple of length 2")
	else:
		trange = [tforms_gyr.min(), tforms_gyr.max()]
	binnorm = 1e-9 * bins / (trange[1] - trange[0])


	if massform:
		try:
			weight = simstars['massform'].in_units('Msol') * binnorm
		except (KeyError, units.UnitsException):
			warnings.warn(
				"Could not load massform array -- falling back to current stellar masses", RuntimeWarning)
			weight = simstars['mass'].in_units('Msol') * binnorm
	else:
		weight = simstars['mass'].in_units('Msol') * binnorm

	sfhist, thebins, patches = plt.hist(tforms_gyr, weights=weight, bins=bins, range=trange,
										histtype='step', **kwargs)

	plt.xlabel('Time [Gyr]', fontsize='large')
	plt.ylabel(r'SFR [M$_\odot$ yr$^{-1}$]', fontsize='large')

	from .util import add_redshift_axis
	add_redshift_axis(sim)

	return array.SimArray(thebins, "Gyr"), array.SimArray(sfhist, "Msol yr**-1")


def schmidtlaw(sim, center=True, pretime='50 Myr', diskheight='3 kpc', rmax='20 kpc',
			   compare=False, bins=10, **kwargs):
	'''Plot star formation surface density vs gas surface density in radial annuli

	.. versionchanged:: 2.0

	  If *center* is True, the transformation of the simulation is now reverted after the plot is made.

	  The *filename*, *legend* and *clear* arguments have been removed. Use the matplotlib functions
	  directly to save the figure or modify the axes.

	  The *radial* argument has been removed, since ``radial=False`` was not implemented.

	  The default *compare* argument is now False.


	Parameters
	----------

	sim : SimSnap
		The simulation snapshot to plot.

	center : bool
		Center and align the input simulation as required.

	pretime : str, optional
		Age of stars to consider for SFR. Default is '50 Myr'.

	diskheight : str, optional
		Height of gas and stars above and below disk considered for SF and gas densities. Default is '3 kpc'.

	rmax : str, optional
		Radius of disk considered. Default is '20 kpc'.

	compare : bool, optional
		Whether to plot Kennicutt (1998) and Bigiel+ (2008) relations for comparison.
		Default is False.

	**kwargs :
		Additional keyword arguments are passed to the matplotlib plot function.

	'''

	if center:
		trans = angmom.faceon(sim)
	else:
		trans = transformation.NullTransformation(sim)

	if isinstance(pretime, str):
		pretime = units.Unit(pretime)

	with trans:

		# select stuff
		diskgas = sim.gas[filt.Disc(rmax, diskheight)]
		diskstars = sim.star[filt.Disc(rmax, diskheight)]

		youngstars = np.where(diskstars['tform'].in_units("Myr") >
							  sim.properties['time'].in_units(
								  "Myr", **sim.conversion_context())
							  - pretime.in_units('Myr'))[0]

		# calculate surface densities
		ps = profile.Profile(diskstars[youngstars], nbins=bins, rmin=0, rmax=rmax)
		pg = profile.Profile(diskgas, nbins=bins, rmin=0, rmax=rmax)

		plt.loglog(pg['density'].in_units('Msol pc^-2'),
				   (ps['density']/pretime).in_units('Msol kpc**-2 yr**-1'), "+",
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

		plt.xlabel(r'$\Sigma_{gas}$ [M$_\odot$ pc$^{-2}$]')
		plt.ylabel(r'$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

		return pg['density'].in_units('Msol pc^-2'), (ps['density']/pretime).in_units('Msol kpc**-2 yr**-1')



def sbprofile(sim, band='V', diskheight='3 kpc', rmax='20 kpc', binning='equaln',
			  center=True, fit_exp=None, fit_sersic=None, **kwargs):
	'''Make a surface brightness profile

	.. versionchanged:: 2.0

	  The *filename*, *axes* and *clear* arguments have been removed.
	  Use the matplotlib functions directly to save the figure or modify the axes.

	  If *center* is True, the transformation of the simulation is now reverted after the plot is made.


	Parameters
	----------

	sim : SimSnap
		The simulation snapshot to plot.

	band : str, optional
		Which band to use; see :mod:`pynbody.analysis.luminosity` for available bands
		and more information about how surface brightnesses are calculated. Default is 'v'.

	diskheight : str, optional
		Height of disk to be profiled. Default is '3 kpc'.

	rmax : str, optional
		Size of disk to be profiled. Default is '20 kpc'.

	binning : str, optional
		How should bin sizes be determined? Default is 'equaln'. See :mod:`pynbody.analysis.profile`
		for more information.

	center : bool, optional
		Automatically align face on and center the simulation. Default is True.

	fit_exp : float, optional
		If set, fit and plot an exponential profile to the data for radii greater than
		this value.

	**kwargs :
		Additional keyword arguments are passed to the matplotlib plot function.

	Returns
	-------

	r : array.SimArray
		Array of radii in kpc

	sb : array.SimArray
		Array of surface brightnesses in mag arcsec^-2

	1/e : float
		The scale length of the exponential fit, in kpc, if *fit_exp* is set.

	exp0 : float
		The central surface brightness of the exponential fit, in mag arcsec^-2,
		if *fit_exp* is set.

	'''

	if center:
		logger.info("Centering...")
		trans = angmom.faceon(sim)
	else:
		trans = transformation.NullTransformation(sim)

	with trans:
		logger.info("Selecting disk stars")
		diskstars = sim.star[filt.Disc(rmax, diskheight)]
		logger.info("Creating profile")
		ps = profile.Profile(diskstars, type=binning)
		logger.info("Plotting")
		r = ps['rbins'].in_units('kpc')

		import matplotlib.pyplot as plt

		plt.plot(r, ps['sb,' + band], linewidth=2, **kwargs)
		plt.ylim(max(ps['sb,' + band]), min(ps['sb,' + band]))

		returns = [r, ps['sb,' + band]]

		if fit_exp:
			exp_inds = np.where(r.in_units('kpc') > fit_exp)
			expfit = np.polyfit(np.array(r[exp_inds]),
								np.array(ps['sb,' + band][exp_inds]), 1)

			# 1.0857 is how many magnitudes a 1/e decrease is
			returns += [1.0857 / expfit[0], expfit[1]]

			fit = np.poly1d(expfit)
			if 'label' in kwargs:
				del kwargs['label']
			if 'linestyle' in kwargs:
				del kwargs['linestyle']
			plt.plot(r, fit(r), linestyle='dashed', **kwargs)

		plt.xlabel('R [kpc]')
		plt.ylabel(band + '-band Surface brightness [mag as$^{-2}$]')
		return tuple(returns)
