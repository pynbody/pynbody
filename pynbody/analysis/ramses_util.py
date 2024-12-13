"""RAMSES-specific analysis utilities

This module provides utilities for working with RAMSES simulations, in two main areas as below.

File Conversion
---------------

For running AHF or pkdgrav, you can convert ramses files to tipsy format using

>>> pynbody.analysis.ramses_util.convert_to_tipsy_fullbox('output_00101') # will convert the whole output

Now you can run AHF or pkdgrav using the file named
`output_00101_fullbox.tipsy` as an input or

>>> s_tipsy = pynbody.load('output_00101_fullbox.tipsy')

You can also just output a part of the simulation :

>>> s = pynbody.analysis.ramses_util.load_center('output_00101', align=False) # centered on halo 0
>>> pynbody.analysis.ramses_util.convert_to_tipsy_simple('output_00101', filt = pynbody.filt.Sphere('200 kpc'))

Now we've got a file called `output_00101.tipsy` which holds only the
200 kpc sphere centered on halo 0.

Generating tform
----------------

Raw RAMSES outputs write the ``tform`` array in unusual units. In pynbody v2, when the user requests
the ``tform`` array, it is converted to physical units using this module's routine :func:`get_tform`.
This happens transparently and for most users it will not be necessary to call this function directly.

"""

import os
import subprocess
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

import pynbody
from pynbody.snapshot.ramses import RamsesSnap

from .. import config_parser
from ..analysis.cosmology import age, tau
from ..units import Unit

ramses_utils = config_parser.get('ramses', 'ramses_utils')

part2birth_path = os.path.join(ramses_utils, "f90", "part2birth")


def convert_to_tipsy_simple(output, halo=0, filt=None):
    """
    Convert RAMSES output to tipsy format readable by e.g. pkdgrav or AHF.

    .. warning::

      This is a quick and dirty conversion, meant to be used for simple post-processing. Importantly, none of the
      cosmologically-relevant information is carried forward. For a more complete conversion for e.g. running through
      pkdgrav or Amiga Halo Finder, see :func:`convert_to_tipsy_fullbox`.

    The snapshot is put into units where G=1, time unit = 1 Gyr and
    mass unit = 2.222286e5 Msol.

    Parameters
    ----------

    output : str
        Path to RAMSES output directory

    halo : int, optional
        Which halo to center on (default = 0)

    filt : pynbody.filt, optional
        A filter to apply to the box before writing out the tipsy file


    """

    h = output.halos()[halo]
    s = pynbody.analysis.halo.center(h)

    for key in ['pos', 'vel', 'mass', 'iord', 'metal']:
        try:
            s[key]
        except KeyError:
            pass

    s['eps'] = s.g['smooth'].min()

    for key in ['rho', 'temp', 'p']:
        s.g[key]

    # try to load tform -- if it fails assign -1
    del(s.s['tform'])
    try:
        get_tform(s)
    except Exception:
        s.s['tform'] = -1.0
        s.s['tform'].units = 'Gyr'

    massunit = 2.222286e5  # in Msol
    dunit = 1.0  # in kpc
    denunit = massunit / dunit ** 3
    velunit = 8.0285 * np.sqrt(6.67384e-8 * denunit) * dunit

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
        s[filt].write(pynbody.snapshot.tipsy.TipsySnap, '%s.tipsy' % output[-12:])
    else:
        s.write(pynbody.snapshot.tipsy.TipsySnap, '%s.tipsy' % output[-12:])


def get_tipsy_units(sim):
    """Return the units in the pkdgrav/gasoline unit system for a RAMSES snapshot `sim`.

    Parameters
    ----------

    sim : RamsesSnap
        RAMSES simulation snapshot

    Returns
    -------

    lenunit, massunit, timeunit, velunit, potentialunit : tuple
        Units in kpc, Msol, Gyr, km/s, and kpc^2 Gyr^-2 a^-1
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
    """Convert RAMSES file `output` to tipsy format readable by pkdgrav and AHF.

    This routine automatically performs all required conversions into the pkdgrav unit system.

    Creates a file called `output_fullbox.tipsy`.

    Parameters
    ----------

    s : RamsesSnap | str
        RAMSES snapshot

    write_param : bool, optional
        Whether or not to write a parameter file for pkdgrav (default = True)

    """

    if isinstance(s, str):
        s = pynbody.load(s)

    if type(s) is not RamsesSnap:
        raise TypeError("This routine can only be used for Ramses snapshots but you are calling with " + str(type(s)))

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
            tipsyfile, fmt=pynbody.snapshot.tipsy.TipsySnap, binary_aux_arrays=True)

    if write_param:
        write_tipsy_param(s, tipsyfile)


def write_tipsy_param(sim, tipsyfile):
    """Write a pkdgrav-readable parameter file for the given RAMSES snapshot

    Parameters
    ----------

    sim : RamsesSnap
        RAMSES snapshot

    tipsyfile : str
        Path to the tipsy file that has been converted from the RAMSES snapshot `sim`

    Notes
    -----

    pkdgrav may be downloaded `here <https://bitbucket.org/dpotter/pkdgrav3/src/master/>`_.
    """

    # determine units
    lenunit, massunit, timeunit, velunit, _ = get_tipsy_units(sim)

    # write the param file
    with open('%s.param' % tipsyfile, 'w') as f:
        f.write('dKpcUnit = %f\n' % lenunit)
        f.write('dMsolUnit = %e\n' % massunit)
        f.write('dOmega0 = %f\n' % sim.properties['omegaM0'])
        f.write('dLambda = %f\n' % sim.properties['omegaL0'])
        h = Unit('%f km s^-1 Mpc^-1' % (sim.properties['h'] * 100))
        f.write('dHubble0 = %f\n' % h.in_units(velunit / lenunit))
        f.write('bComove = 1\n')

def write_ahf_input(sim, tipsyfile):
    """Write an input file for AHF for the RAMSES snapshot `sim` and the converted tipsy file `tipsyfile`

    Parameters
    ----------

    sim : RamsesSnap
        RAMSES snapshot

    tipsyfile : str
        Path to the tipsy file that has been converted from the RAMSES snapshot `sim`

    Notes
    -----

    Amiga Halo Finder may be downloaded `here <http://popia.ft.uam.es/AHF/Download.html>`_.

    """

    # determine units
    _lenunit, massunit, _timeunit, velunit, _ = get_tipsy_units(sim)

    with open('%s.AHF.input' % tipsyfile, 'w') as f:
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


def _get_tform_using_part2birth(sim, part2birth_path):
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
    assert done == len(top.s), f"Not all particles have a formation time. Found {done}/{len(top.s)}"
    top.s['tform'].units = 'Gyr'

    return sim.s['tform']

def get_tform(sim, *, times_are_proper: bool, use_part2birth: Optional[bool]=None, part2birth_path: str=part2birth_path, ):
    """

    Convert RAMSES times to physical times for stars and **replaces** the original
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
    times_are_proper : boolean, optional
        If True, `tform` is assumed to be in proper time.
        If False, it is assumed to be in conformal time.

    Notes
    -----
    The behaviour of the function can be customized in the configuration file.

    The value ``use_part2birth_by_default`` controls whether the conversion should
    be made using ``part2birth`` or in Python. It can be set as follows

    .. code-block:: ini

        [ramses]
        use_part2birth_by_default = True  # will use part2birth to convert times
        use_part2birth_by_default = False  # will use internal Python routine

    The default path to ``part2birth`` is obtained by joining the RAMSES utils
    path (as read from configuration) and ``f90/part2birth``.

    For example, with the following configuration,

    .. code-block:: ini

        [ramses]
        ramses_utils = /home/user/ramses/utils

    the default path would be ``/home/user/ramses/utils/f90/part2birth``.

    """
    if use_part2birth is None:
        use_part2birth = config_parser.getboolean('ramses', 'use_part2birth_by_default')

    if use_part2birth:
        return _get_tform_using_part2birth(sim, part2birth_path=part2birth_path)

    if hasattr(sim, 'base'):
        top = sim.base
    else:
        top = sim

    birth_raw = top.star["tform_raw"].view(np.ndarray)

    H0 = (top.properties["h"] * 100 * Unit("km s^-1 Mpc^-1")).in_units("Gyr^-1")

    if times_are_proper:
        # Times are computed in units of H0
        # with a value of 0 corresponding to z=0
        times = birth_raw

        time_tot = age(top, z=0) * H0
        birth_date = (time_tot + times) / H0
    else:
        h0 = top.properties["h"]
        aexp_bins = np.geomspace(1e-3, 1, 10_000)
        z_bins = 1 / aexp_bins - 1
        tau_bins = tau(top, z=z_bins, unit="0.01 s Mpc km^-1") * h0
        age_bins = age(top, z=z_bins)
        birth_date = np.interp(birth_raw, tau_bins, age_bins)

    top.s["tform"] = birth_date
    top.s['tform'].units = "Gyr"
    return sim.s["tform"]
