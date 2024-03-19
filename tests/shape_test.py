import numpy as np
import numpy.testing as npt
import pynbody

def test_2D_shape():
  from pynbody.analysis.halo import shape
  f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024.gz")
  h = f.halos()
  pynbody.analysis.halo.center(h[0], mode='hyb')

  halo_shape = pynbody.analysis.halo.shape(h[0].d, nbins=3, rmin=0, rmax=0.004, ndim=2)
  assert len(halo_shape[0]) == len(halo_shape[1])
  npt.assert_allclose(halo_shape[0], [0.00023014, 0.00094025, 0.00234817], atol=1e-5)
  npt.assert_allclose(halo_shape[1], [[0.00023933, 0.0002213 ],
                                      [0.0009826 , 0.00089972],
                                      [0.00255131, 0.0021612 ]], atol=1e-5)
  assert np.all(halo_shape[2] == [51991, 51957, 52368])
  npt.assert_allclose(halo_shape[3], [[[ 0.79139587,  0.61130399],
                                       [ 0.61130399, -0.79139587]],
                                      [[ 0.81009301,  0.58630139],
                                       [ 0.58630139, -0.81009301]],
                                      [[ 0.86194519,  0.50700147],
                                       [ 0.50700147, -0.86194519]]], atol=1e05)

def test_3D_shape():
  from pynbody.analysis.halo import shape
  f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024.gz")
  h = f.halos()
  pynbody.analysis.halo.center(h[0], mode='hyb')

  halo_shape = pynbody.analysis.halo.shape(h[0].d, nbins=3, rmin=0, rmax=0.004, ndim=3)
  assert len(halo_shape[0]) == len(halo_shape[1])
  npt.assert_allclose(halo_shape[0], [0.00031606, 0.00120896, 0.00275828], atol=1e-5)
  npt.assert_allclose(halo_shape[1], [[0.00034493, 0.00031589, 0.00028975],
                                      [0.00136191, 0.0012902 , 0.00100561],
                                      [0.00306224, 0.00283095, 0.00242072]], atol=1e-5)
  assert np.all(halo_shape[2] == [50178, 50295, 50155])
  npt.assert_allclose(halo_shape[3], [[[ 0.5718012 , -0.79536163, -0.20110509],
                                       [ 0.7572302 ,  0.41735419,  0.50241209],
                                       [-0.31566724, -0.43956269,  0.84091547]],
                                      [[ 0.14345368, -0.94976754,  0.27814146],
                                       [-0.84688619, -0.26322816, -0.46205489],
                                       [ 0.5120594 , -0.16927068, -0.84210606]],
                                      [[-0.79724828, -0.52498303,  0.29797315],
                                       [-0.55661753,  0.44830226, -0.69942978],
                                       [ 0.23360672, -0.72347627, -0.64962296]]], atol=1e-5)



