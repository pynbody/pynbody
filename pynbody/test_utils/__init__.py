"""Utilities for downloading and unpacking test data packages"""

import os
import pathlib
import shutil
import subprocess
import tarfile
import urllib.request

test_data_packages = {
    'swift': {'path': 'SWIFT',
              'archive_name': 'swift.tar.gz'},
    'adaptahop_longint': {'path': 'adaptahop_longint',
                       'archive_name': 'adaptahop_longint.tar.gz'},
    'arepo': {'path': 'arepo',
              'archive_name': 'arepo.tar.gz'},
    'gadget': {'path': 'gadget2',
               'archive_name': 'gadget.tar.gz'},
    'hbt': {'path': 'gadget4_subfind_HBT',
            'archive_name': 'gadget4_subfind_HBT.tar.gz'},
    'gasoline_ahf': {'path': 'gasoline_ahf',
                     'archive_name': 'gasoline.tar.gz'},
    'gizmo': {'path': 'gizmo',
                'archive_name': 'gizmo.tar.gz'},
    'grafic': {'path': 'grafic_test',
                'archive_name': 'grafic.tar.gz'},
    'lpicola': {'path': 'lpicola',
                'archive_name': 'lpicola.tar.gz'},
    'nchilada': {'path': 'nchilada_test',
                 'archive_name': 'nchilada.tar.gz'},
    'ramses': {'path': 'ramses',
               'archive_name': 'ramses.tar.gz'},
    'rockstar': {'path': 'rockstar',
                 'archive_name': 'rockstar.tar.gz'},
    'subfind': {'path': 'subfind',
                'archive_name': 'subfind.tar.gz'},
    'tng_subfind': {'path': 'arepo/tng',
                    'archive_name': 'tng_subfind.tar.gz'},
}

test_data_url = "https://zenodo.org/record/12687409/files/{archive_name}?download=1"

def precache_test_data():
    """Download and unpack all test data packages."""
    for package in test_data_packages.values():
        _download_and_unpack_test_data_if_not_present(package)

def test_data_hash():
    """Return a hash representing the data packages to be downloaded"""
    # print a hex digest of the hash of the test data package urls
    import hashlib
    m = hashlib.sha256()
    for package_name in test_data_packages:
        m.update(test_data_packages[package_name]['archive_name'].encode())
    m.update(test_data_url.encode())
    return m.hexdigest()

def download_and_unpack_test_data(archive_name):
    """Download and unpack test data with the given archive name and unpack path.

    Equivalent to running:

     wget https://zenodo.org/record/.../files/{archive_name}?download=1
     tar -xzf {archive_name}
    """

    url = test_data_url.format(archive_name=archive_name)
    unpack_path = f"testdata/"

    if not os.path.exists("testdata"):
        os.mkdir("testdata")

    # Wanted to do the following, but it fails with a certificate error on macos
    #
    #with urllib.request.urlopen(url) as data_file:
    #    with tarfile.open(fileobj=data_file) as tar:
    #        tar.extractall(unpack_path)

    subprocess.run(["wget", "-O", archive_name, url], check=True)

    # Unpack the tar file
    with tarfile.open(archive_name) as tar:
        tar.extractall(unpack_path, filter='data')

    # Remove the downloaded tar file
    os.remove(archive_name)


def ensure_test_data_available(*package_names):
    """Ensure that the specified test data packages are available in the testdata directory."""
    for package_name in package_names:
        if package_name not in test_data_packages:
            raise ValueError(f"Test data package {package_name} not found in test_data_packages")
        package = test_data_packages[package_name]
        _download_and_unpack_test_data_if_not_present(package)


def _download_and_unpack_test_data_if_not_present(package):
    if not pathlib.Path(f"testdata/{package['path']}").exists():
        download_and_unpack_test_data(package['archive_name'])
