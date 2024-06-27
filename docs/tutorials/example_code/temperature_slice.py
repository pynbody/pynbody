import matplotlib.pylab as plt

import pynbody
import pynbody.plot.sph as sph

# load the snapshot and set to physical units
s = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')
s.physical_units()

# load the halos
halos = s.halos()

# center on the largest halo and align the disk
pynbody.analysis.angmom.sideon(halos[1])

#create a simple slice showing the gas temperature
sph.image(halos[1].g, qty="temp", width=50, cmap="YlOrRd", denoise=True, approximate_fast=False)
