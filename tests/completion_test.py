import IPython
from IPython.core.completer import provisionalcompleter
from packaging import version

if version.parse(IPython.__version__) >= version.parse("7.0.0"):
    from IPython.core.interactiveshell import InteractiveShell
else:
    # For older versions, import accordingly
    from IPython.frontend.terminal.interactiveshell import InteractiveShell

import numpy as np

import pynbody


def test_ipython_key_completions():
    f = pynbody.new(dm=1000, star=500, gas=500, order="gas,dm,star")
    f["pos"] = pynbody.array.SimArray(np.random.normal(scale=1.0, size=f["pos"].shape), units="kpc")
    f["vel"] = pynbody.array.SimArray(np.random.normal(scale=1.0, size=f["vel"].shape), units="km s**-1")
    f["mass"] = pynbody.array.SimArray(np.random.uniform(1.0, 10.0, size=f["mass"].shape), units="Msol")

    ip = InteractiveShell.instance()
    # Inject your object into the IPython user namespace
    ip.user_ns["f"] = f

    # Simulate the text where completion is requested
    text = "f["
    cursor_pos = len(text)

    # Get completions from IPython's completer
    with provisionalcompleter():
        completions = list(ip.Completer.completions(text, cursor_pos))

    # Extract completion texts
    completion_list = [str(c.text)[1:-1] for c in completions]  # if c.
    expected_keys = f.all_keys()

    assert set(completion_list) == set(expected_keys)


if __name__ == "__main__":
    print("testing ipython completions...")
    test_ipython_key_completions()
    print("passed!")
