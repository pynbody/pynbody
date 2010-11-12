import numpy as np

def open_(filename, *args) :
    """Open a file, determining from the filename whether to use
    gzip decompression"""
    
    if filename[-3:]==".gz" :
	import gzip
	return gzip.open(filename, *args)
    else :
	return open(filename, *args)
