---
name: Bug or inaccuracy report
about: Create a report to help us improve pynbody
title: ''
labels: ''
assignees: ''

---

Using the template below will help us reproduce your problem, which is an essential first step in helping or improving the code. Feel free to customise the template if you are used to raising bug reports, but be sure that you give us enough information to start investigating.

**Describe the bug**
A clear and concise description of what the bug or inaccuracy is.

**To Reproduce**
Provide a simple, minimal python script that we can use to reproduce the bug or inaccuracy.

Preferably reproduce the bug using pynbody's own test data (as used by the tutorials, and downloadable from
[zenodo](https://zenodo.org/doi/10.5281/zenodo.12552027). If this impossible, provide us a pointer to another file
that we can use to reproduce the problem. This file should be as small as possible to illustrate the problem, as it
may be incorporated into future versions of the test data.

**Expected behaviour or result**
Please provide a clear and concise description of what you expected, and how it differs from the result.

**Setup (please complete the following information):**
 - OS
 - Python version
 - Numpy version
 - Pynbody version

You can access this information from python in the following one-liner:

```python
import sys, platform, pynbody, numpy; print("%s\r\n%s\r\npynbody %s\r\nnumpy %s\r\n"%(sys.version, platform.platform(), pynbody.__version__, numpy.__version__))
```

**Additional context**
Add any other context about the problem here.
