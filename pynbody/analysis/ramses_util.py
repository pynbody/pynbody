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

This now generated a directory called `birth` in the parent directory
of your output. It then calls the routine `part2birth` located in the
RAMSES utils (see the `bitbucket repository
<https://bitbucket.org/rteyssie/ramses>`_. :func:`get_tform` also
deletes the previous `tform` array (not from disk, just from the
currently loaded snapshot). The next time you call :func:`get_tform`,
the data will be loaded from the disk and `part2birth` won't need to
be run again.

Feb 2016 - RS
Note that in this version of ramses_util I have moved the 'birth' files
into the output directory to which they pertain. The old code wasn't
placing them in a subdir of 'birth' in the top output and since I had
to fix it I thought it better not to have a single directory with
potentially 1000's of files for all RAMSES outputs. The get_tform
routine now creates the files under the approrpiate "output_" dir
and reads them back from there.


"""

import pynbody
import subprocess
import numpy as np
import os
from .. units import Unit

from .. import config_parser

ramses_utils = config_parser.get('ramses', 'ramses_utils')

part2birth_path = ramses_utils + 'f90/part2birth'


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

    cmtokpc = 3.2407793e-22
    lenunit = sim._info['unit_l'] / sim.properties['a'] * cmtokpc
    massunit = pynbody.analysis.cosmology.rho_crit(
        sim, z=0, unit='Msol kpc^-3') * lenunit ** 3
    G_u = 4.4998712e-6  # G in kpc^3 / Msol / Gyr^2
    timeunit = np.sqrt(1 / G_u * lenunit ** 3 / massunit)

    return Unit('%e kpc' % lenunit), Unit('%e Msol' % massunit), Unit('%e Gyr' % timeunit)


def convert_to_tipsy_fullbox(output, write_param=True):
    """
    Convert RAMSES file `output` to tipsy format readable by pkdgrav
    and Amiga Halo Finder. Does all unit conversions etc. into the
    pkdgrav unit system. Creates a file called `output_fullbox.tipsy`.

    **Input**: 

    *output*: name of RAMSES output

    **Optional Keywords**:

    *write_param*: whether or not to write the parameter file (default = True)

    """

    s = pynbody.load(output)

    lenunit, massunit, timeunit = get_tipsy_units(s)

#    l_unit = Unit('%f kpc'%lenunit)
#    t_unit = Unit('%f Gyr'%timeunit)
    velunit = lenunit / timeunit

    tipsyfile = "%s_fullbox.tipsy" % (output)

    s['mass'].convert_units(massunit)
    s.g['temp']

    # get the appropriate tform
    get_tform(s)
    s.g['metals'] = s.g['metal']
    s['pos'].convert_units(lenunit)
    s['vel'].convert_units(velunit)
    s['eps'] = s.g['smooth'].min()
    s['eps'].units = s['pos'].units

    # try to load the potential array -- if it's not there, make it zeroes
    try:
        s['phi']
    except KeyError:
        s['phi'] = 0.0

    del(s.g['metal'])
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
    lenunit, massunit, timeunit = get_tipsy_units(sim)
    h = Unit('%f km s^-1 Mpc^-1' % (sim.properties['h'] * 100))
    l_unit = Unit('%f kpc' % lenunit)
    t_unit = Unit('%f Gyr' % timeunit)
    v_unit = l_unit / t_unit

    # write the param file
    f = open('%s.param' % tipsyfile, 'w')
    f.write('dKpcUnit = %f\n' % lenunit)
    f.write('dMsolUnit = %e\n' % massunit)
    f.write('dOmega0 = %f\n' % sim.properties['omegaM0'])
    f.write('dLambda = %f\n' % sim.properties['omegaL0'])
    h = Unit('%f km s^-1 Mpc^-1' % (sim.properties['h'] * 100))
    f.write('dHubble0 = %f\n' % h.in_units(v_unit / l_unit))
    f.write('bComove = 1\n')
    f.close()


def write_ahf_input(sim, tipsyfile):
    """Write an input file that can be used by the `Amiga Halo Finder
    <http://popia.ft.uam.es/AHF/Download.html>`_ with the
    corresponding `tipsyfile` which is the `sim` in tipsy format.
    """

    # determine units
    lenunit, massunit, timeunit = get_tipsy_units(sim)
    h = Unit('%f km s^-1 Mpc^-1' % (sim.properties['h'] * 100))
    l_unit = Unit('%f kpc' % lenunit)
    t_unit = Unit('%f Gyr' % timeunit)
    v_unit = l_unit / t_unit

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

 #   velunit = Unit('%f cm'%s._info['unit_l'])/Unit('%f s'%s._info['unit_t'])

    f.write('TIPSY_VUNIT   = %e\n' %
            v_unit.ratio('km s^-1 a', **sim.conversion_context()))

    # the thermal energy in K -> km^2/s^2

    f.write('TIPSY_EUNIT   = %e\n' % (
        (pynbody.units.k / pynbody.units.m_p).in_units('km^2 s^-2 K^-1') * 5. / 3.))
    f.close()


def get_tform(sim, part2birth_path=part2birth_path):
    """Use `part2birth` to calculate the formation time of stars in
    Gyr and **replaces** the original `tform` array.

    **Input**: 

    *sim*: RAMSES snapshot

    **Optional Keywords:** 

    *part2birth_path:* by default, this is
     $HOME/ramses/utils/f90/part2birth, as specified in
     `default_config.ini` in your pynbody install directory. You can
     override this like so -- make a file called ".pynbodyrc" in your
     home directory, and include

    [ramses]

    ramses_utils = /path/to/your/ramses/utils/directory

    """

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
        except (IOError, OSError):      # Both are necessary as python 2 throws OSError, while python 3 throws IOError
            try:
                # birth_xxx doesn't exist, create it with ramses part2birth util
                with open(os.devnull, 'w') as fnull:
                    subprocess.call([part2birth_path, '-inp', 'output_%s' % top._timestep_id],
                                    stdout=fnull, stderr=fnull)
                birth_file = FortranFile(birthfile_path)
            except (IOError, OSError):
                import warnings
                warnings.warn("Failed to read 'tform' from birth files at %s and to generate them with utility at %s.\n"
                              "Formation times in Ramses code units can be accessed through the 'tform_raw' array."
                              % (birthfile_path, part2birth_path))
                raise IOError

        ages = birth_file.read_reals(np.float64)
        new = np.where(ages > 0)[0]
        top.s['tform'][done:done + len(new)] = ages[new]
        done += len(new)

        birth_file.close()
    top.s['tform'].units = 'Gyr'

    return sim.s['tform']
