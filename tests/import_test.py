import subprocess
import sys


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
