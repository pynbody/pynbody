import pynbody
import numpy as np

np.random.seed(1)

def make_fake_bar(npart=100000, max=1, min=-1, barlength=.8, barwidth=0.05, phi=0, fraction=0.2):
    
    x = np.random.sample(int(npart*fraction))*(max-min) + min
    y = np.random.sample(int(npart*fraction))*(max-min) + min
    
    xbar = np.random.sample(npart, )*(barlength/2+barlength/2) - barlength/2
    ybar = np.random.sample(npart)*(barwidth/2+barwidth/2) - barwidth/2

    x = np.concatenate([x,xbar])
    y = np.concatenate([y,ybar])
    
    good = np.where((x**2 + y**2 < 1))[0]
    
    s = pynbody.snapshot.new(len(good))
    s['x'] = x[good]
    s['y'] = y[good]
    s['pos'].units = 'kpc'
    s['mass'] = 1.0
    s['mass'].units = 'Msol'
    s['vel'] = 1.0
    s['vel'].units = 'km s^-1'
    s.rotate_z(phi)
    return s

def test_fourier_profile():
    bar = make_fake_bar(phi=45)

    p = pynbody.analysis.profile.Profile(bar, nbins=50)

    assert(np.all(p['fourier']['amp'][2,4:20] > 0.1))
    assert(np.allclose(np.abs(p['fourier']['phi'][2,4:20]/2), np.pi/4.0, rtol=0.05))


def test_potential_profile_fp64():
    f = pynbody.new(100)
    coords = np.random.normal(size=(100,3))
    del f['pos']
    del f['mass']
    f['pos'] = np.array(coords,dtype=np.float64)
    f['eps'] = np.ones(100,dtype=np.float64)
    f['mass'] = np.ones(100,dtype=np.float64)
    p = pynbody.analysis.profile.Profile(f, nbins=50)
    p['pot']


def test_potential_profile_fp32():
    f = pynbody.new(100)
    coords = np.random.normal(size=(100,3))
    del f['pos']
    del f['mass']
    f['pos'] = np.array(coords,dtype=np.float32)
    f['eps'] = np.ones(100,dtype=np.float32)
    f['mass'] = np.ones(100,dtype=np.float32)
    p = pynbody.analysis.profile.Profile(f, nbins=50)
    p['pot']
