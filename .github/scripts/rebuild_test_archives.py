"""Rebuild the pynbody test data tarballs from an unpacked tests/testdata directory

This exists to recover the archives originally hosted on OSF from the copy of the
unpacked data held in the GitHub Actions cache. Every file under the testdata directory
is assigned to the package whose claimed path is the longest prefix of it, where a
package claims its verify_path (if it has an extract_path, or if verify_path is a
directory) or the top-level directory of its verify_path (otherwise). Files claimed by
no package are reported and written to an extra archive so that nothing is lost.

Usage: python rebuild_test_archives.py <testdata_dir> <output_dir>
"""

import hashlib
import importlib.util
import json
import pathlib
import sys
import tarfile


def load_test_data_packages(repo_root):
    spec = importlib.util.spec_from_file_location(
        "test_utils", repo_root / "pynbody" / "test_utils" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.test_data_packages


def get_claims(packages, testdata):
    """Return {claimed path relative to testdata: (package name, extract path)}"""
    claims = {}
    for name, package in packages.items():
        verify = pathlib.PurePosixPath(package['verify_path'])
        extract = pathlib.PurePosixPath(package.get('extract_path', ''))
        if 'extract_path' in package or (testdata / verify).is_dir():
            claimed = verify
        else:
            claimed = pathlib.PurePosixPath(verify.parts[0])
        if not (testdata / claimed).exists():
            print(f"WARNING: {name}: {claimed} not present in testdata; archive will not be built")
            continue
        if claimed in claims:
            raise ValueError(f"{name} and {claims[claimed][0]} both claim {claimed}")
        claims[claimed] = (name, extract)
    return claims


def owner_of(path, claims):
    for candidate in [path, *path.parents]:
        if candidate in claims:
            return candidate
    return None


def sha256(filename):
    m = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            m.update(block)
    return m.hexdigest()


def main(testdata, output):
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    packages = load_test_data_packages(repo_root)
    claims = get_claims(packages, testdata)

    members = {claimed: [] for claimed in claims}
    unassigned = []
    for path in sorted(testdata.rglob('*')):
        relpath = pathlib.PurePosixPath(path.relative_to(testdata).as_posix())
        claimed = owner_of(relpath, claims)
        if claimed is not None:
            members[claimed].append(relpath)
        elif not path.is_dir() or not any(path.iterdir()):
            unassigned.append(relpath)

    output.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for claimed, (name, extract) in claims.items():
        archive_name = packages[name]['archive_name']
        archive_path = output / archive_name
        with tarfile.open(archive_path, 'w:gz') as tar:
            for relpath in members[claimed]:
                tar.add(testdata / relpath, arcname=str(relpath.relative_to(extract)), recursive=False)
        manifest[archive_name] = {'package': name, 'extract_path': str(extract),
                                  'num_members': len(members[claimed]),
                                  'size': archive_path.stat().st_size, 'sha256': sha256(archive_path)}
        print(f"{archive_name}: {len(members[claimed])} members, {archive_path.stat().st_size / 1e6:.1f} MB")

    if unassigned:
        print("WARNING: the following paths are not claimed by any package and have been written to "
              "unassigned.tar.gz for manual inspection:")
        with tarfile.open(output / 'unassigned.tar.gz', 'w:gz') as tar:
            for relpath in unassigned:
                print(f"  {relpath}")
                tar.add(testdata / relpath, arcname=str(relpath), recursive=False)

    with open(output / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
