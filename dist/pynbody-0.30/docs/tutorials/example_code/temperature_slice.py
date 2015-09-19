import pynbody
import pynbody.plot.sph as sph
import matplotlib.pylab as plt

# load the snapshot and set to physical units
s = pynbody.load('testdata/g15784.lr.01024.gz')
s.physical_units()

# load the halos
h = s.halos()

# center on the largest halo and align the disk
pynbody.analysis.angmom.sideon(h[1])

#create a simple slice showing the gas temperature
sph.image(h[1].g,qty="temp",width=50,cmap="YlOrRd", denoise=True,approximate_fast=False)

