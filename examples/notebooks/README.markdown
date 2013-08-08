#Pynbody IPython notebooks

The [IPython notebook](http://ipython.org/notebook.html) is a tool for
interactive data exploration. We have prepared a few notebooks to
complement our tutorials -- you can think of them as interactive
totorials. These notebooks can be viewed as static content via the
[IPython Notebook Viewer](http://nbviewer.ipython.org) using the links
below, or you can use them as starting points for your own analysis.

For more information, see the [IPython notebook
documentation](http://ipython.org/ipython-doc/stable/interactive/htmlnotebook.html). 


##Pynbody notebooks available through the Notebook viewer

[Pynbody demo](http://nbviewer.ipython.org/urls/raw.github.com/pynbody/pynbody/master/examples/notebooks/pynbody_demo.ipynb)

[Pynbody Ramses demo](http://nbviewer.ipython.org/urls/raw.github.com/pynbody/pynbody/master/examples/notebooks/pynbody_demo-ramses.ipynb)

##Using the notebooks interactively

If you have a recent version of IPython (> 0.12) you should be able to
use these notebooks interactively. Copy them to a different directory
and start the notebook server from that directory:

```
$ mkdir pynbody_notebooks
$ cd pynbody_notebooks
$ ipython notebook --pylab inline
```

This will start up the ipython session and open up a browser window
listing the notebooks available from the current directory. 

##Connecting to an analysis machine/cluster

If you need to perform your analysis on specialized hardware (and who
doesn't), the easiest way to proceed is to start up the notebook
session on the remote machine and connecting to the notebook via an
SSH tunnel.

On remote: 

```
remote> ipython notebook --pylab inline
```

locally:

```
local> ssh -L 8888:127.0.0.1:8888 user@remote
```

This will make all requests on port 8888 of your local machine get
automatically routed to port 8888 of the remote machine. Now all you
need to do is start up your browser and in the address bar type
`127.0.0.1:8888` -- it should bring up the notebook screen, served
from the remote machine.
