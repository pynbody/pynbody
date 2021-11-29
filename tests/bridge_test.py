import pynbody
import numpy as np


def test_order_bridge():
    f1 = pynbody.new(dm=1000)
    f2 = pynbody.new(dm=3000)
    f1['iord'] = np.arange(5, 2005, 2, dtype=int)
    f2['iord'] = np.arange(3000, dtype=int)

    b = pynbody.bridge.OrderBridge(f1, f2)

    h1 = f1[:50:2]
    assert b(h1).ancestor is f2
    assert (b(h1)['iord'] == h1['iord']).all


def test_bridge_factory():
    f1 = pynbody.load("testdata/g15784.lr.01024")
    f2 = pynbody.load("testdata/g15784.lr.01024")
    b = pynbody.bridge.bridge_factory(f1, f2)

    assert isinstance(b, pynbody.bridge.OrderBridge)


def test_nonmonotonic_transfer_matrix():
    # test the case where a non-monotonic bridge between two simulations is queries for particle transfer
    f1 = pynbody.new(dm=10)
    f2 = pynbody.new(dm=10)
    f1['iord'] = np.arange(0,10)
    f2['iord'] = np.array([0,2,4,6,8,1,3,5,7,9],dtype=np.int32)


    # 100% of mass in group 1 transfers to group 1
    # 80% of mass in group 0 transfers to group 0
    # 20% of mass in group 0 transfers to group 1

    f1['grp'] = np.array([0,0,0,0,0,1,1,1,1,1],dtype=np.int32)
    f2['grp'] = np.array([0,0,0,0,1,1,1,1,1,1],dtype=np.int32)[f2['iord']]

    b = pynbody.bridge.OrderBridge(f1,f2,monotonic=False)

    h1 = pynbody.halo.GrpCatalogue(f1)
    h2 = pynbody.halo.GrpCatalogue(f2)

    xfer = b.catalog_transfer_matrix(0,1,h1,h2)

    assert (xfer==[[4,1],[0,5]]).all()


def test_nonmonotonic_incomplete_bridge():
    # Test the non-monotonic bridge where not all the particles are shared
    f1 = pynbody.new(dm=10)
    f2 = pynbody.new(dm=9)
    f1['iord'] = np.arange(0,10)
    f2['iord'] = np.array([0,2,4,6,8,1,5,7,9])
    b = pynbody.bridge.OrderBridge(f1,f2,monotonic=False)

    assert (b(f1[:5])['iord']==np.array([0,1,2,4])).all()


def test_monotonic_incomplete_bridge():
    # Test the monotonic bridge where not all the particles are shared
    f1 = pynbody.new(dm=10)
    f2 = pynbody.new(dm=9)
    f1['iord'] = np.arange(0, 10)
    f2['iord'] = np.array([0, 1, 2, 4, 5 ,6, 7, 8, 9])
    b = pynbody.bridge.OrderBridge(f1, f2, monotonic=False)

    assert (b(f1[:5])['iord'] == np.array([0, 1, 2, 4])).all()


def test_family_morphing():
    f1 = pynbody.new(dm=5, gas=5)
    f2 = pynbody.new(dm=10)

    # set dm and gas iords separately as it's possible the new command initialises them out of order
    f1.dm['iord'] = np.arange(0,5)
    f1.gas['iord'] = np.arange(5,10)
    f2['iord'] = np.array([0,2,4,6,8,1,3,5,7,9])

    b = pynbody.bridge.OrderBridge(f1,f2,monotonic=False,allow_family_change=True)

    assert (b(f2).dm['iord']==np.array([0,2,4,1,3])).all()
    assert (b(f2).gas['iord'] == np.array([6, 8, 5, 7, 9])).all()