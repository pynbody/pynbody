"""Failed lazy reads must neither become data nor suppress their I/O error."""
import errno
import warnings

import h5py
import numpy as np
import numpy.testing as npt
import pytest

import pynbody
from pynbody.snapshot.simsnap import SimSnap


@pytest.fixture
def checksummed_snapshot(tmp_path):
    filename = tmp_path / 'snap.hdf5'
    n = 32
    positions = np.arange(3 * n, dtype=float).reshape(n, 3) + 1
    with h5py.File(filename, 'w') as f:
        header = f.create_group('Header')
        counts = np.array([0, n, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs.update(NumPart_ThisFile=counts, NumPart_Total=counts,
                            NumPart_Total_HighWord=np.zeros(6, dtype=np.uint32),
                            MassTable=np.zeros(6),
                            NumFilesPerSnapshot=1, Time=1., Redshift=0., BoxSize=1000.,
                            Omega0=0.3, OmegaLambda=0.7, HubbleParam=0.7)
        f.create_group('Units').attrs.update(UnitLength_in_cm=1., UnitMass_in_g=1.,
                                            UnitVelocity_in_cm_per_s=1.)
        particles = f.create_group('PartType1')
        coordinates = particles.create_dataset('Coordinates', data=positions, chunks=(8, 3), fletcher32=True)
        coordinates.attrs.update({'VarDescription': 'U_L', 'CGSConversionFactor': 1.,
                                  'aexp-scale-exponent': 0., 'h-scale-exponent': 0.})
        masses = particles.create_dataset('Masses', data=np.ones(n))
        masses.attrs.update({'VarDescription': 'U_M', 'CGSConversionFactor': 1.,
                             'aexp-scale-exponent': 0., 'h-scale-exponent': 0.})
        particles.create_dataset('ParticleIDs', data=np.arange(n, dtype=np.uint64))
    return filename, positions


def _flip_chunk_byte(filename):
    with h5py.File(filename, 'r') as f:
        offset = f['PartType1/Coordinates'].id.get_chunk_info(0).byte_offset
    with open(filename, 'r+b') as f:
        f.seek(offset + 8)
        byte = f.read(1)
        f.seek(offset + 8)
        f.write(bytes([byte[0] ^ 0xff]))


@pytest.mark.parametrize('access', ['snapshot', 'family', 'slice'])
def test_checksum_error_is_not_cached(checksummed_snapshot, access):
    filename, _ = checksummed_snapshot
    _flip_chunk_byte(filename)
    with h5py.File(filename, 'r') as f, pytest.raises(OSError):
        f['PartType1/Coordinates'][:]
    snapshot = pynbody.load(filename)
    target = {'snapshot': snapshot, 'family': snapshot.dm, 'slice': snapshot.dm[::2]}[access]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pynbody.util.hdf_bulk_read.BulkReadFallbackWarning)
        for _ in range(2):
            with pytest.raises(OSError):
                target['x']
            assert not {'pos', 'x', 'y', 'z'}.intersection(snapshot.keys())
            assert not {'pos', 'x', 'y', 'z'}.intersection(snapshot.family_keys())


@pytest.mark.parametrize('error', [OSError(errno.EMFILE, 'Too many open files'), ValueError('read failed'), KeyboardInterrupt('cancelled')])
@pytest.mark.parametrize('access', ['snapshot', 'family', 'promotion'])
def test_failed_load_cleans_coupled_arrays_and_allows_retry(error, access):
    class FailingSnap(SimSnap):
        fail = True

        def _load_array(self, name, fam=None):
            if access == 'promotion' and fam is None:
                missing = getattr(pynbody.snapshot, 'ArrayNotFoundError', OSError)
                raise missing('Load at family level')
            for field in ('pos', 'vel'):
                if fam is None:
                    self._create_array(field, 3)
                else:
                    self._create_family_array(field, fam, 3)
            if self.fail and (access != 'promotion' or fam is pynbody.family.dm):
                raise error
            for field in ('pos', 'vel'):
                if fam is None:
                    self[field][:] = 2
                else:
                    self[fam][field][:] = 2

    snapshot = pynbody.new(dm=3, gas=2, class_=FailingSnap)
    for name in list(snapshot.keys()):
        del snapshot[name]
    snapshot['untouched'] = np.arange(len(snapshot))
    target = snapshot.dm if access == 'family' else snapshot
    for _ in range(2):
        with pytest.raises(type(error)) as caught:
            target['pos']
        assert caught.value is error
        assert list(snapshot.keys()) == ['untouched']
        assert not snapshot.family_keys()
        npt.assert_array_equal(snapshot['untouched'], np.arange(len(snapshot)))
    snapshot.fail = False
    npt.assert_array_equal(target['pos'], np.full((len(target), 3), 2.))
    npt.assert_array_equal(target['vel'], np.full((len(target), 3), 2.))


def test_missing_array_still_derives_and_unknown_array_raises_keyerror(checksummed_snapshot):
    filename, positions = checksummed_snapshot
    snapshot = pynbody.load(filename)
    npt.assert_allclose(snapshot['r'], np.linalg.norm(positions, axis=1))
    with pytest.raises(KeyError):
        snapshot['does_not_exist']


def test_failed_postprocessing_cleans_loaded_arrays(checksummed_snapshot, monkeypatch):
    filename, positions = checksummed_snapshot
    snapshot = pynbody.load(filename)
    convert = snapshot._autoconvert_array_unit

    def fail_conversion(array):
        raise ValueError('conversion failed')

    monkeypatch.setattr(snapshot, '_autoconvert_array_unit', fail_conversion)
    with pytest.raises(ValueError, match='conversion failed'):
        snapshot['pos']
    assert not {'pos', 'x', 'y', 'z'}.intersection(snapshot.keys())
    monkeypatch.setattr(snapshot, '_autoconvert_array_unit', convert)
    npt.assert_array_equal(snapshot['pos'], positions)


@pytest.mark.parametrize('external, error, fallback', [
    (True, FileNotFoundError('no birth utility'), True),
    (False, FileNotFoundError('missing data'), False),
    (True, PermissionError('unreadable data'), False),
])
def test_ramses_only_retains_optional_birth_file_fallback(monkeypatch, external, error, fallback):
    from pynbody.analysis import ramses_util
    from pynbody.snapshot.ramses import RamsesSnap

    snapshot = pynbody.new(star=2)
    snapshot.star['tform'] = np.ones(2)
    snapshot._file_units_system = [pynbody.units.cm, pynbody.units.s]
    snapshot.is_cosmological = True
    snapshot.times_are_proper = True
    monkeypatch.setitem(pynbody.config_parser['ramses'], 'use_part2birth_by_default', str(external))

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(ramses_util, 'get_tform', unavailable)
    if fallback:
        RamsesSnap._convert_tform(snapshot)
    else:
        with pytest.raises(type(error)) as caught:
            RamsesSnap._convert_tform(snapshot)
        assert caught.value is error
