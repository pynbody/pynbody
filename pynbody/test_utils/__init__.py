"""Utilities for downloading and unpacking test data packages

WARNING: This module must not depend on a working pynbody installation.
It is used during CI to download test data before pynbody is built/installed.
Only use standard library modules and certifi (which is available in CI).
"""

import os
import pathlib
import shutil
import ssl
import tarfile
import urllib.request

import certifi

test_data_packages = {
    'swift': {'verify_path': 'SWIFT',
              'archive_name': 'swift.tar.gz'},
    'swift_isolated': {'verify_path': 'SWIFT/isolated_0008.hdf5',
                       'extract_path': 'SWIFT',
                       'archive_name': 'swift_isolated.tar.gz'},
    'adaptahop_longint': {'verify_path': 'adaptahop_longint',
                       'archive_name': 'adaptahop_longint.tar.gz'},
    'arepo': {'verify_path': 'arepo',
              'archive_name': 'arepo.tar.gz'},
    'gadget': {'verify_path': 'gadget2',
               'archive_name': 'gadget.tar.gz'},
    'hbt': {'verify_path': 'gadget4_subfind_HBT',
            'archive_name': 'gadget4_subfind_HBT.tar.gz'},
    'gasoline_ahf': {'verify_path': 'gasoline_ahf',
                     'archive_name': 'gasoline.tar.gz'},
    'gizmo': {'verify_path': 'gizmo',
                'archive_name': 'gizmo.tar.gz'},
    'grafic': {'verify_path': 'grafic_test',
                'archive_name': 'grafic.tar.gz'},
    'lpicola': {'verify_path': 'lpicola',
                'archive_name': 'lpicola.tar.gz'},
    'nchilada': {'verify_path': 'nchilada_test',
                 'archive_name': 'nchilada.tar.gz'},
    'ramses': {'verify_path': 'ramses',
               'archive_name': 'ramses.tar.gz'},
    'rockstar': {'verify_path': 'rockstar',
                 'archive_name': 'rockstar.tar.gz'},
    'subfind': {'verify_path': 'subfind',
                'archive_name': 'subfind.tar.gz'},
    'tng_subfind': {'verify_path': 'arepo/tng',
                    'archive_name': 'tng_subfind.tar.gz'},
    'pkdgrav3': {'verify_path': 'pkdgrav3',
                 'archive_name': 'pkdgrav3.tar.gz'},
}

test_data_url = "https://zenodo.org/record/15769415/files/{archive_name}?download=1"

def precache_test_data(verbose=False):
    """Download and unpack all test data packages."""
    for package_name, package in test_data_packages.items():
        _download_and_unpack_test_data_if_not_present(package, package_name, verbose)

def test_data_hash():
    """Return a hash representing the data packages to be downloaded"""
    # print a hex digest of the hash of the test data package urls
    import hashlib
    m = hashlib.sha256()
    for package_name in test_data_packages:
        m.update(test_data_packages[package_name]['archive_name'].encode())
    m.update(test_data_url.encode())
    return m.hexdigest()


def download_and_unpack_test_data(archive_name, unpack_path="", verbose=False):
    """Download and unpack test data with the given archive name and unpack path.

    Equivalent to running:

        wget https://zenodo.org/record/.../files/{archive_name}?download=1
        tar -xzf {archive_name}
    """

    url = test_data_url.format(archive_name=archive_name)
    unpack_path = f"testdata/{unpack_path}"

    if not os.path.exists(unpack_path):
        os.makedirs(unpack_path, exist_ok=True)

    # use certifi to get the CA bundle for SSL verification
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # Download to a temporary file first
    temp_file = f"{archive_name}.tmp"
    try:
        if verbose:
            print(f"Downloading {archive_name} from {url}")
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(temp_file, 'wb') as f:
                shutil.copyfileobj(response, f)
        
        if verbose:
            print(f"Extracting {archive_name} to {unpack_path}")
        # Extract from the downloaded file
        with tarfile.open(temp_file) as tar:
            tar.extractall(unpack_path, filter='data')
        if verbose:
            print(f"Successfully unpacked {archive_name}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def ensure_test_data_available(*package_names):
    """Ensure that the specified test data packages are available in the testdata directory."""
    for package_name in package_names:
        if package_name not in test_data_packages:
            raise ValueError(f"Test data package {package_name} not found in test_data_packages")
        package = test_data_packages[package_name]
        _download_and_unpack_test_data_if_not_present(package, package_name, False)


def _download_and_unpack_test_data_if_not_present(package, package_name, verbose=False):
    if not pathlib.Path(f"testdata/{package['verify_path']}").exists():
        if verbose:
            print(f"Test data package '{package_name}' not found, downloading...")
        download_and_unpack_test_data(package['archive_name'], package.get('extract_path', ''), verbose)
    elif verbose:
        print(f"Test data package '{package_name}' already exists, skipping")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--fetch":
        precache_test_data(verbose=True)
    else:
        print(test_data_hash())
