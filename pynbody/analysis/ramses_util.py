"""

ramses_util
===========

Handy utilities for using RAMSES outputs in pynbody. For a complete
demo on how to use RAMSES outputs with pynbody, have a look at the
`ipython notebook demo
<http://nbviewer.ipython.org/github/pynbody/pynbody/blob/master/examples/notebooks/pynbody_demo-ramses.ipynb>`_

File Conversion
---------------

>>> pynbody.analysis.ramses_util.convert_to_tipsy_fullbox('output_00101') # will convert the whole output

Now you can run AHF or pkdgrav using the file named
`output_00101_fullbox.tipsy` as an input or

>>> s_tipsy = pynbody.load('output_00101_fullbox.tipsy')

You can also just output a part of the simulation :

>>> s = pynbody.analysis.ramses_util.load_center('output_00101', align=False) # centered on halo 0
>>> pynbody.analysis.ramses_util.convert_to_tipsy_simple('output_00101', file = pynbody.filt.Sphere('200 kpc')

Now we've got a file called `output_00101.tipsy` which holds only the
200 kpc sphere centered on halo 0.

Generating tform
----------------

A problem with RAMSES outputs in pynbody is that the `tform` array is
in funny units that aren't easily usable. To generate a new `tform`
array (in Gyr) you can use the :func:`get_tform` defined here. It's
very easy:

>>> s = pynbody.load('output_00101')
>>> pynbody.analysis.ramses_util.get_tform(s)

By default, this will simply convert in-place the formation times.
If you want the changes to be persistent, you can use

>>> pynbody.analysis.ramses_util.get_tform(s, use_part2birth=True)

This now generates for each `partXXXX.outYYYYY` file a corresponding
`birthXXXXX.outYYYYY` file containing the formation time in physical units.
This uses the routine `part2birth` located in the
RAMSES utils (see the `bitbucket repository
<https://bitbucket.org/rteyssie/ramses>`_).

:func:`get_tform` also deletes the previous `tform` array (not from disk, just
from the currently loaded snapshot). The next time you call :func:`get_tform`,
the data will be loaded from the disk and `part2birth` won't need to
be run again.
"""

import os
import subprocess
import warnings
from pathlib import Path

import numpy as np

import pynbody
from pynbody.snapshot.ramses import RamsesSnap

from .. import config_parser
from ..analysis._cosmology_time import friedman
from ..units import Unit

ramses_utils = config_parser.get('ramses', 'ramses_utils')

part2birth_path = os.path.join(ramses_utils, "f90", "part2birth")


def convert_to_tipsy_simple(output, halo=0, filt=None):
    """
    Convert RAMSES output to tipsy format readable by
    e.g. pkdgrav. This is a quick and dirty conversion, meant to be
    used for quick visualization or other simple post
    processing. Importantly, none of the cosmologically-relevant
    information is carried forward. For a more complete conversion for
    e.g. running through pkdgrav or Amiga Halo Finder, see
    :func:`convert_to_tipsy_fullbox`.

    The snapshot is put into units where G=1, time unit = 1 Gyr and
    mass unit = 2.222286e5 Msol.

    **Input**:

    *output* : path to RAMSES output directory

    **Optional Keywords**:

    *filt* : a filter to apply to the box before writing out the tipsy file

    *halo* : which hop halo to center on -- default = 0

    """

    h = output.halos()[halo]
    s = pynbody.analysis.halo.center(h)

    for key in ['pos', 'vel', 'mass', 'iord', 'metal']:
        try:
            s[key]
        except:
            pass

    s['eps'] = s.g['smooth'].min()

    for key in ['rho', 'temp', 'p']:
        s.g[key]

    # try to load tform -- if it fails assign -1
    del(s.s['tform'])
    try:
        get_tform(s)
    except:
        s.s['tform'] = -1.0
        s.s['tform'].units = 'Gyr'

    massunit = 2.222286e5  # in Msol
    dunit = 1.0  # in kpc
    denunit = massunit / dunit ** 3
    velunit = 8.0285 * np.sqrt(6.67384e-8 * denunit) * dunit
    timeunit = dunit / velunit * 0.97781311

    s['pos'].convert_units('kpc')
    s['vel'].convert_units('%e km s^-1' % velunit)
    s['mass'].convert_units('%e Msol' % massunit)
    s['eps'].convert_units('kpc')
    s.g['rho'].convert_units('%e Msol kpc^-3' % denunit)

    s.s['tform'].convert_units('Gyr')
    del(s.g['smooth'])
    s.s['metals'] = s.s['metal']
    s.g['metals'] = s.g['metal']
    del(s['metal'])
    s.g['temp']
    s.properties['a'] = pynbody.analysis.cosmology.age(s)
    if filt is not None:
        s[filt].write(pynbody.tipsy.TipsySnap, '%s.tipsy' % output[-12:])
    else:
        s.write(pynbody.tipsy.TipsySnap, '%s.tipsy' % output[-12:])


def get_tipsy_units(sim):
    """
    Returns snapshot `sim` units in the pkdgrav/gasoline unit
    system.  This is probably not a function to be called by users,
    but it is used instead by other routines for file conversion.

    **Input**:

    *sim*: RAMSES simulation snapshot

    **Return values**:

    *lenunit, massunit, timeunit* : tuple specifying the units in kpc, Msol, and Gyr

    """

    # figure out the units starting with mass
    lenunit = sim['x'].units.in_units("a kpc", **sim.conversion_context())
    massunit = pynbody.analysis.cosmology.rho_crit(sim, z=0, unit='Msol kpc^-3') * lenunit ** 3
    G = pynbody.units.G.in_units("kpc**3 Msol**-1 Gyr**-2")
    timeunit = np.sqrt(1 / G * lenunit ** 3 / massunit)
    velunit = lenunit / timeunit
    potentialunit = (velunit ** 2)

    return Unit('%.5e a kpc' % lenunit), Unit('%.5e Msol' % massunit),\
           Unit('%.5e Gyr' % timeunit), Unit('%.5e a kpc Gyr**-1' % velunit),\
           Unit('%.5e kpc**2 Gyr**-2 a**-1' % potentialunit)


def convert_to_tipsy_fullbox(s, write_param=True):
    """
    Convert RAMSES file `output` to tipsy format readable by pkdgrav
    and Amiga Halo Finder. Does all unit conversions etc. into the
    pkdgrav unit system. Creates a file called `output_fullbox.tipsy`.

    **Input**:

    *output*: name of RAMSES output

    **Optional Keywords**:

    *write_param*: whether or not to write the parameter file (default = True)

    """

    if type(s) is not RamsesSnap:
        raise ValueError("This routine can only be used for Ramses snapshots but you are calling with " + str(type(s)))

    warnings.warn("This routine currently makes the assumption that the ramses snapshot is cosmological\n"
                  "when converting units. Beware if converting isolated runs.")

    lenunit, massunit, timeunit, velunit, potentialunit = get_tipsy_units(s)
    tipsyfile = "%s_fullbox.tipsy" % (s._filename)

    s['mass'].convert_units(massunit)
    s['pos'].convert_units(lenunit)
    s['vel'].convert_units(velunit)

    # try to load the potential array -- if it's not there, make it zeroes
    try:
        s['phi'].convert_units(potentialunit)
    except KeyError:
        s['phi'] = 0.0

    if "gas" in s.families():
        s['eps'] = s.g['smooth'].min()
        s['eps'].units = s['pos'].units
        s.g['temp']
        s.g['metals'] = s.g['metal']
        del(s.g['metal'])
    else:
        # If we don't have gas, i.e. a DMO sim,
        # load with force_gas to get the AMR smoothing
        # This can only be temporary to ensure that the converted tipsy snapshot is still DMO
        s_with_gas_forced = pynbody.load(s._filename, force_gas=True)
        s['eps'] = s_with_gas_forced.g['smooth'].min()
        s['eps'].units = s['pos'].units

    if "star" in s.families():
        s.st['tform'].convert_units(timeunit)

    del(s['smooth'])

    s.write(filename='%s' %
            tipsyfile, fmt=pynbody.tipsy.TipsySnap, binary_aux_arrays=True)

    if write_param:
        write_tipsy_param(s, tipsyfile)


def write_tipsy_param(sim, tipsyfile):
    """Write a pkdgrav-readable parameter file for RAMSES snapshot
    `sim` with the prefix `filename`
    """

    # determine units
    lenunit, massunit, timeunit, velunit, _ = get_tipsy_units(sim)

    # write the param file
    f = open('%s.param' % tipsyfile, 'w')
    f.write('dKpcUnit = %f\n' % lenunit)
    f.write('dMsolUnit = %e\n' % massunit)
    f.write('dOmega0 = %f\n' % sim.properties['omegaM0'])
    f.write('dLambda = %f\n' % sim.properties['omegaL0'])
    h = Unit('%f km s^-1 Mpc^-1' % (sim.properties['h'] * 100))
    f.write('dHubble0 = %f\n' % h.in_units(velunit / lenunit))
    f.write('bComove = 1\n')
    f.close()


def write_ahf_input(sim, tipsyfile):
    """Write an input file that can be used by the `Amiga Halo Finder
    <http://popia.ft.uam.es/AHF/Download.html>`_ with the
    corresponding `tipsyfile` which is the `sim` in tipsy format.
    """

    # determine units
    lenunit, massunit, timeunit, velunit, _ = get_tipsy_units(sim)
    h = Unit('%f km s^-1 Mpc^-1' % (sim.properties['h'] * 100))

    f = open('%s.AHF.input' % tipsyfile, 'w')
    f.write('[AHF]\n')
    f.write('ic_filename = %s\n' % tipsyfile)
    f.write('ic_filetype = 90\n')
    f.write('outfile_prefix = %s\n' % tipsyfile)
    f.write('LgridDomain = 256\n')
    f.write('LgridMax = 2097152\n')
    f.write('NperDomCell = 5\n')
    f.write('NperRefCell = 5\n')
    f.write('VescTune = 1.0\n')
    f.write('NminPerHalo = 50\n')
    f.write('RhoVir = 0\n')
    f.write('Dvir = 200\n')
    f.write('MaxGatherRad = 1.0\n')
    f.write('[TIPSY]\n')
    f.write('TIPSY_BOXSIZE = %e\n' % (sim.properties['boxsize'].in_units(
        'Mpc') * sim.properties['h'] / sim.properties['a']))
    f.write('TIPSY_MUNIT   = %e\n' % (massunit * sim.properties['h']))
    f.write('TIPSY_OMEGA0  = %f\n' % sim.properties['omegaM0'])
    f.write('TIPSY_LAMBDA0 = %f\n' % sim.properties['omegaL0'])
    f.write('TIPSY_VUNIT   = %e\n' %
            velunit.ratio('km s^-1 a', **sim.conversion_context()))
    f.write('TIPSY_EUNIT   = %e\n' % (
        (pynbody.units.k / pynbody.units.m_p).in_units('km^2 s^-2 K^-1') * 5. / 3.))
    f.close()


def get_tform_using_part2birth(sim, part2birth_path):
    from scipy.io import FortranFile

    if hasattr(sim, 'base'):
        top = sim.base
    else:
        top = sim

    cpu_range = top._cpus

    top.s['tform'] = -1.0
    done = 0

    for cpu_id in cpu_range:

        # Birth files are located inside the output directory.
        birthfile_name = "birth_%s.out%05d" %(top._timestep_id, cpu_id)
        birthfile_path = os.path.join(top.filename, birthfile_name)
        try:
            birth_file = FortranFile(birthfile_path)
        except OSError:
            try:
                # birth_xxx doesn't exist, create it with ramses part2birth util
                with open(os.devnull, 'w') as fnull:
                    cwd = Path(top.filename).parent
                    subprocess.call([part2birth_path, '-inp', 'output_%s' % top._timestep_id],
                                    stdout=fnull, stderr=fnull, cwd=cwd)
                birth_file = FortranFile(birthfile_path)
            except OSError:
                msg = (
                    "Failed to read 'tform' from birth files at %s and to generate "
                    "them with utility at %s.\n Formation times in Ramses code units "
                    "can be accessed through the 'tform_raw' array."
                )
                warnings.warn(
                    msg % (birthfile_path, part2birth_path)
                )
                raise

        ages = birth_file.read_reals(np.float64)
        new = np.where(ages > 0)[0]
        top.s['tform'][done:done + len(new)] = ages[new]
        done += len(new)

        birth_file.close()
    top.s['tform'].units = 'Gyr'

    return sim.s['tform']


def get_tform(sim, use_part2birth=None, part2birth_path=part2birth_path):
    """

    Convert conformal times to physical times for stars and **replaces** the original
     `tform` array.

    Parameters
    ----------
    sim : RAMSES snapshot or subsnapshot
    use_part2birth : boolean, optional
        If True, use the `part2birth` tool (see notes below) to convert the formation
        times to physical times. If False, use a Python-based convertor.
        See notes for the default value.
    part2birth_path : str, optional
        Path to the `part2birth` util. Only used if `use_part2birth` is also True.
        See notes for the default value.


    Notes
    -----
    The behaviour of the function can be customized in the configuration file.

    The value `use_part2birth_by_default` controls whether the conversion should
    be made using `part2birth` or in Python. It can be set as follows

        [ramses]
        use_part2birth_by_default = True  # will use part2birth to convert times
        use_part2birth_by_default = False  # will use internal Python routine

    The default path to `part2birth` is obtained by joining the RAMSES utils
    path (as read from configuration) and `f90/part2birth`.

    For example, with the following configuration,

        [ramses]
        ramses_utils = /home/user/ramses/utils

    the default path would be `/home/user/ramses/utils/f90/part2birth`.

    """
    if use_part2birth is None:
        use_part2birth = config_parser.getboolean('ramses', 'use_part2birth_by_default')

    if use_part2birth:
        return get_tform_using_part2birth(sim, part2birth_path=part2birth_path)

    if hasattr(sim, 'base'):
        top = sim.base
    else:
        top = sim

    conformal_ages = top.star["tform_raw"].view(np.ndarray)

    cm_per_km = 1e5
    cm_per_Mpc = 3.085677580962325e+24
    s_per_Gyr = 3.1556926e+16

    tau_frw, t_frw, dtau, n_frw, time_tot = friedman(
        top.properties["omegaM0"],
        top.properties["omegaL0"],
        1.0 - top.properties["omegaM0"] - top.properties["omegaL0"],
    )
    h100 = top.properties["h"]
    nOver2 = n_frw / 2
    unit_t = top._info["unit_t"]
    t_scale = 1.0 / (h100 * 100 * cm_per_km / cm_per_Mpc) / unit_t

    # calculate index into lookup table (n_frw elements in
    # lookup table)
    dage = 1 + (10 * conformal_ages / dtau)
    dage = np.minimum(dage, nOver2 + (dage - nOver2) / 10.0)
    iage = np.array(dage, dtype=np.int32)

    # linearly interpolate physical times from t_frw and tau_frw lookup
    # tables.
    t = t_frw[iage] * (conformal_ages - tau_frw[iage - 1]) / (tau_frw[iage] - tau_frw[iage - 1])
    t = t + (
        t_frw[iage - 1] * (conformal_ages - tau_frw[iage]) / (tau_frw[iage - 1] - tau_frw[iage])
    )

    top.s["tform"] = pynbody.analysis.cosmology.age(sim, z=0) + t * t_scale * unit_t / s_per_Gyr
    top.s['tform'].units = 'Gyr'
    return sim.s["tform"]
