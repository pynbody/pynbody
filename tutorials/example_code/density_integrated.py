import pynbody
import pynbody.plot.sph as sph
import matplotlib.pylab as plt

# load the snapshot and set to physical units
s = pynbody.load('testdata/g15784.lr.01024.gz')
s.physical_units()

# load the halos
h = s.halos()

# center on the largest halo and align the disk
pynbody.analysis.angmom.faceon(h[1])

#create an image of gas density integrated down the line of site (z axis)
sph.image(h[1].g,qty="rho",units="g cm^-2",width=100,cmap="Greys")

