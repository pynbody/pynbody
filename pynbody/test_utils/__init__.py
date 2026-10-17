"""Utilities for downloading and unpacking test data packages

WARNING: This module must not depend on a working pynbody installation.
It is used during CI to download test data before pynbody is built/installed.
Only use standard library modules.
"""

import argparse
import os
import pathlib
import shutil
import tarfile
import urllib.request

test_data_packages = {
    'swift': {'verify_path': 'SWIFT',
              'archive_name': 'swift.tar.gz'},
    'swift_isolated': {'verify_path': 'SWIFT/isolated_0008.hdf5',
                       'extract_path': 'SWIFT',
                       'archive_name': 'swift_isolated.tar.gz'},
    'swift_planetary': {'verify_path': 'SWIFT/planetary.hdf5',
                        'extract_path': 'SWIFT',
                        'archive_name': 'swift_planetary.tar.gz'},
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
    'tutorial_gadget4': {'verify_path': 'tutorial_gadget4/subhalo_desc_035.hdf5',
                         'archive_name': 'tutorial_gadget4.tar.gz',
                         'extended': True},
    'tutorial_gadget': {'verify_path': 'tutorial_gadget/snapshot_020',
                        'archive_name': 'tutorial_gadget.tar.gz',
                        'extended': True},
    'tiny_FIRE' : {'verify_path': 'tiny_FIRE', 
                    'archive_name': 'tiny_FIRE.tar.gz'},
    'soap': {'verify_path': 'SOAP',
             'archive_name': 'soap.tar.gz'}
}

# Base URL of the pynbody data store. Test data archives are under testdata/ and ion fraction
# tables under ionfrac/. Can be overridden with the PYNBODY_DATA_URL environment variable, e.g. to
# point at a mirror.
data_url = os.environ.get('PYNBODY_DATA_URL', 'https://pynbody.pontzen.uk').rstrip('/')

def precache_test_data(verbose=False, extended=False):
    """Download and unpack all test data packages.

    By default, only the packages required for resting are downloaded. To include extended packages for documentation/
    tutorials, set extended=True.
    """
    for package_name, package in test_data_packages.items():
        # only download packages that are not marked as 'extended' packages - unless extended=True
        if extended or not package.get('extended', False):
            _download_and_unpack_test_data_if_not_present(package, package_name, verbose)

def test_data_hash():
    """Return a hash representing the data packages to be downloaded"""
    # print a hex digest of the hash of the test data package urls
    import hashlib
    m = hashlib.sha256()
    for package_name in test_data_packages:
        m.update(test_data_packages[package_name]['archive_name'].encode())
    m.update(data_url.encode())
    return m.hexdigest()

def download_file(path, destination):
    """Download the file at the given path relative to data_url, writing it to destination.

    The download goes to a temporary file first, so that an interrupted download never leaves a partial file
    at the destination."""
    temp_file = f"{destination}.tmp"
    try:
        request = urllib.request.Request(f"{data_url}/{path}", headers={'User-Agent': 'pynbody'})
        with urllib.request.urlopen(request) as response, open(temp_file, 'wb') as f:
            shutil.copyfileobj(response, f)
        os.replace(temp_file, destination)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def download_and_unpack_test_data(archive_name, unpack_path="", verbose=False):
    """Download and unpack test data with the given archive name and unpack path.

    Equivalent to running:

        wget {data_url}/testdata/{archive_name}
        tar -xzf {archive_name}
    """

    unpack_path = f"testdata/{unpack_path}"

    if not os.path.exists(unpack_path):
        os.makedirs(unpack_path, exist_ok=True)

    # Download to a temporary file first
    temp_file = f"{archive_name}.tmp"
    try:
        if verbose:
            print(f"Downloading {archive_name}")
        download_file(f"testdata/{archive_name}", temp_file)

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


_required_ionfrac_tables = ("hm12", "fg20") # Tables needed by the tests

def download_ionfrac_table(name, destination):
    """Download an ion fraction table from the pynbody data repository

    Note that this is also called from pynbody.analysis.ionfrac to fetch
    the tables when they are needed and not already present.
    """
    download_file(f"ionfrac/{name}.npz", destination)

def precache_ionfrac_tables(verbose=False):
    """Download all ionfrac tables needed for tests"""
    os.makedirs("ionfrac_tables", exist_ok=True)
    for name in _required_ionfrac_tables:
        if verbose:
            print(f"Downloading table: {name}")
        download_ionfrac_table(name, pathlib.Path("ionfrac_tables") / pathlib.Path(name+".npz"))

def ionfrac_tables_hash():
    """Return a hash of the ionfrac table names"""
    import hashlib
    m = hashlib.sha256()
    for name in _required_ionfrac_tables:
        m.update(name.encode())
    m.update(data_url.encode())
    return m.hexdigest()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download data files needed for tests')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fetch-testdata", action="store_true", help="Download pynbody test data files")
    group.add_argument("--hash-testdata", action="store_true", help="Output the hash of the test data files")
    group.add_argument("--fetch-ionfrac", action="store_true", help="Download ionfrac table files needed for tests")
    group.add_argument("--hash-ionfrac", action="store_true", help="Output the hash of the ionfrac table names")
    args = parser.parse_args()

    if args.fetch_testdata:
        precache_test_data(verbose=True)
    elif args.hash_testdata:
        print(test_data_hash())
    elif args.fetch_ionfrac:
        precache_ionfrac_tables(verbose=True)
    elif args.hash_ionfrac:
        print(ionfrac_tables_hash())
    else:
        raise RuntimeError("No operation specified!")
