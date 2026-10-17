import os
import subprocess
import sys

import pytest


def test_import_plot_module():
    import pynbody
    pl_dir = dir(pynbody.plot)
    im = pynbody.plot.sph.image
    assert callable(im)


def test_concurrent_lazy_import_of_plot_module():
    # Run in a fresh interpreter, since pynbody.plot is probably already imported here
    code = """
import threading
import pynbody

n_threads = 16
barrier = threading.Barrier(n_threads)
errors = []

def touch():
    barrier.wait()
    try:
        pynbody.plot.sph.image
    except Exception as e:
        errors.append(repr(e))

threads = [threading.Thread(target=touch) for _ in range(n_threads)]
for t in threads:
    t.start()
for t in threads:
    t.join()

assert not errors, errors
assert isinstance(pynbody.plot, type(pynbody))
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


@pytest.mark.skipif(not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
                    reason="only meaningful on free-threaded Python with the GIL disabled")
def test_import_keeps_gil_disabled():
    # Run in a fresh interpreter without PYTHON_GIL (which CI sets to 0, forcing the GIL off whatever modules
    # declare), since once an extension module re-enables the GIL it stays enabled
    code = """
import importlib
import pkgutil
import sys

import pytest

import pynbody

# (pure python, and imported by pynbody.array.shared for whichever platform this is)
skip = {"pynbody.array.shared.win_detail", "pynbody.array.shared.posix_detail"}

assert not sys._is_gil_enabled(), "importing pynbody enabled the GIL"
for module in pkgutil.walk_packages(pynbody.__path__, "pynbody."):
    if module.name not in skip:
        importlib.import_module(module.name)
        assert not sys._is_gil_enabled(), f"importing {module.name} enabled the GIL"
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
