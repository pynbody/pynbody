.. profile tutorial


Halo loading in Pynbody
=======================


Basic Load the Halos from AHF catalog
-------------------------------------
Simple example for loading AHF halo catalog.  Currently, AHF is the only halo catalog format that is supported directly, though we hope to add support for different halo catalog formats in the future.  Simulation snapshot file and AHF halo files should all be in working directory:


>>> import pynbody
>>> s = pynbody.load('tipsyfilename') 
>>> h = s.halos() # this scans for available halo catalogues and loads
>>> print h[1] # you should find this is now a SimSnap representing halo ID 1 in your catalogue
>>> h[1].properties ##  prints all 'properties' of halo 1
>>> h[1].properties['mass']  # halo 1 mass
>>> h[2].properties['Xc']  # halo 2 x-center
>>> h[1].keys()  # what is stored for each particle in halo 1
>>> h[1]['x'] # x-positions of each particle in halo 1 (h[1]['x'][0 - npart-1])
>>> s['grp'][0] # halo id of particle 0 
>>> h[1]['grp'] # halo ids particles in halo 1 (most will be '1')
>>> h[3].star['pos'][0] ## component data. components are gas, star, dark
>>> max(h[1]['grp']) # id of smallest Npart subhalo of halo 1, or '1' if no subhalos
>>> pynbody.analysis.halo.potential_minimum(h[2]) # position of particle with minimum (tipsy) potential in halo 2
>>> h[1].properties['children'] # subhalo IDs of halo 1 (if any)

Note that h[n] is a subview of f, meaning it shares the underlying
data. Therefore any changes to data in h[n] is automatically reflected
in f. Similarly, h[n] is not centred, since its pos array is shared
with f. Investigate pynbody.analysis.halo for centring tools.  How it
works

The halo.py module does most of the work. It can be extended with more
base classes of HaloCatalogue to support new formats. The halos()
method of SimSnap? goes through the _halo_classes list at the bottom
of halo.py using the static method _can_load of the HaloCatalogue
classes to see which if any can find a catalogue corresponding to the
SnapShot. The first one which returns True gets instantiated.



Partial loading the Halos versus full load
-------------------------------------------

By default, partial loading (i.e. `lazy loading') is used when loading simulation files.  This means that the command h=s.halos() does not actually load halo information until needed.  Sometimes the simulation snapshot or the halo _particles file is very large, and is not desirable or possible to load everything into memory.  Hence, partial loading can be very useful.  The _particles file is needed to find out which particles belong to a particular halo.  

In the following example only particles from a single halo are loaded:

>>> import pynbody
>>> s = pynbody.load('tipsyfilename') # nothing gets loaded yet.
>>> h = s.halos(dummy=True) # loads properties, but skips the _particles file load.
>>> h[2] # `dummy' halo - its properties are loaded, but not particle data.
>>> h2=h.load_copy(2) # loads particle ids from _particles file for halo 2 only
>>> h2.star['pos'][2] # Mow that data is requested, the simulation snapshot read.



Partial loading can be switched off as follows:

DESCRIBE HOW................


Halos only usage (if no simulation snapshot present) - Does this work?)
-----------------------------------------------------------------------
>>> s=pynbody.new()
>>> s.properties  # prints the default properties just loaded (can be changed by e.g. s.properties['h']=.73)
>>> s._filename="simulation_snapshot_name_without_the_ahf_suffixes"
>>> h=s.halos() ## load the halos from the AHF files
>>> h[1].properties ##  prints all 'properties' of halo 1
>>> h[1].properties['mass']  # halo 1 mass

This can be useful if one wants to analyze the halo catalog without needing any information about the simulation particles.






Write halo catalog (i.e. convert AHF outfiles to tipsy format)
--------------------------------------------------------------

Tipsy is a particle viewer.  A tipsy format file can be useful for quick viewing in tipsy to check
whether the AHF halo finder did anything sensible. Write the (ahf)
halo catalog to disk. Former idl users might notice that this produces outfiles
similar to `Alyson's idl script'. 

The 3 written file types are: 
1- .gtp (tipsy file with halos as star particles). 
2- .grp (ascii halo id of every snapshot particle, 0 if none). 
3- .stat ascii condensed version of AHF halos file).

This halo file set is compatible with the halo finder SKID.

tipsy and skid can be found at http://www-hpcc.astro.washington.edu/tools/.  


Extra steps required if AHF was run as MPI parallel:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note:: If AHF was run using MPI (multiple processors) -- Skip this if AHF was run in serial or OpenMP.  There will be a set of AHF output files (_halos, _particles...) for each MPI process.  A couple of short steps are required to make it concatenate the filesets together.  At this time, MPI AHF does not write any _substructure files and does not keep global halo IDs across domains, so it is not always desireable to use MPI AHF. 

from the command prompt (example for a z=0 AHF output):

> cat simfilename.00*z0.000.AHF_halos | cat > simfilename.z0.000.AHF_halos

The  _particles files requires an extra step to sort out headers.  Get the total number of halos (= number lines in _halos file - 1 for header).  From the command prompt:

> wc simfilename.z0.000.AHF_halos | gawk '{print $1-1}' > simfilename.z0.000.AHF_particles

> gawk 'FNR > 1' simfilename.ahf.0*.z0.000.AHF_particles >> simfilename.z0.000.AHF_particles

> cat simfilename.00*z0.000.AHF_profiles | cat > simfilename.z0.000.AHF_profiles
    #  the _profiles concatenation is optional.  Note MPI AHF produces no _substructure files.


Now convert to tipsy If serial/OpenMP AHF (or after above file concatenations made if MPI):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: If AHF was run using MPI, first concatnate the AHF processor outfiles into a single outfile set, as described above.

In directory with AHF outputs, where there should be a _halos file, a _particles file, and a _substructure (if not run with MPI):

>>> import pynbody
>>> s=pynbody.load(simfile)
>>> h=s.halos(make_grp=True)  # _particles file is read now, not `partial loaded'
>>> h.writestat(s,h,simfile.stat)
>>> h.writetipsy(s,h,simfile.gtp) 
>>> s['grp'].write() # writes 'simfile.grp'

.. note:: If a .grp file is not needed, it can be skipped by using s.halos() instead of s.halos(make_grp=True) (and then also skipping the s['grp].write().  The reason to skip the _grp file creation is that for simulations with large particle numbers, the _particles file read through, which is triggered by s['grp'].write() (with partial loading enabled), can be quite slow when the _particles file is large.


pynbody computes the `hubble' constant by default -- specifying avoids rounding errors in converting halo positions, which might be important for some applications, as in:
>>> h.writetipsy(s,h,"test.gtp",hubble=0.7)  
>>> h.writestat(s,h,"test.stat",hubble=0.7) 

In the above example, we also override the default outfile name, which just adds an extension to the simulation file name.


.. note:: The default AHF and pynbody expectation (as of 2012.04.17) is that TIPSY_PARTICLE_ORDERING (gas, dark, star) is NOT set in the AHF compilation, so pynbody expects non-tipsy default AHF particle ID ordering of dark, star, gas. 

.. note:: AHF orders halos by particle number, which is not quite the same as ordering by halo mass if there are multiple particle masses. 

.. note:: One a .grp file is written, there will now be both a _particles file and a .grp file in the directory.  By default, pynbody will try to load the .grp file.  _particles file loading can be forced by 

>>> h=pynbody.halo.AHFCatalogue(s)


Generating catalogues on the fly with pynbody
---------------------------------------------

There is also a mechanism for running halo-finders on-the-fly with
pynbody from simulation snapshot. This is currently implemented by the
AHFCatalogue class which reads amiga halo catalogues.

The AHFCatalogue._can_load() looks for an AHF _particles file. If that
is not found and all the other _can_loads fail (----------WHAT OTHER FILES ATTEMPT TO LOAD?), pynbody searches your
executable PATH environment variable for AHFstep. If it finds that, it
creates the necessary input files and runs Amiga Halo Finder for
you. AHFstep doesn't take that long (1 minute for 10 million
particles?). Once AHFstep finishes, the HaloCatalog loads the particle
file into Halo objects for each halo that are IndexedSubSnaps. Each
Halo has slightly extended properties that include all the values from
the AHF_halos file. The AHFCatalogue also loads the substructure file
into the ['children'] property.

Installing AHFstep for pynbody compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: These are quite old instructions and may be out of date.

From the command prompt:

>wget http://popia.ft.uam.es/AHF/files/ahf-v1.0.tgz

>tar zxf ahf-v1.0.tgz

>cd ahf-v1.0

Edit Makefile.config. For Tipsy, uncomment the DEFINEFLAGS line under MW1.1024g1bwk, g1536, GALFOBS. Possibly switch CC line to use icc (then -fopenmp becomes -openmp) and up the OPTIMIZE line to -O3

From command prompt:

>make AHF

>mkdir ~/bin/

>cp bin/AHF* ~/bin/

>export PATH="$PATH:${HOME}/bin"

It is a good idea to put this last line above into your .bashrc file on Linux or .profile in Mac OS X. 


