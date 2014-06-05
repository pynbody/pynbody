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

# create the subplots
f, axs = plt.subplots(1,2,figsize=(14,6))

#create a simple slice showing the gas temperature, with velocity vectors overlaid
sph.velocity_image(h[1].g, vector_color="cyan", qty="temp",width=50,cmap="YlOrRd", 
                   denoise=True,approximate_fast=False, subplot=axs[0], show_cbar = False)

#you can also make a stream visualization instead of a quiver plot
pynbody.analysis.angmom.faceon(h[1])
s['pos'].convert_units('Mpc')
sph.velocity_image(s.g, width='3 Mpc', cmap = "Greys_r", mode='stream', units='Msol kpc^-2', 
                   density = 2.0, vector_resolution=100, vmin=1e-1,subplot=axs[1], 
                   show_cbar=False, vector_color='black')
