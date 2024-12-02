import numpy as np

import pynbody


def make_guassian_blob(n_part, seed=1337):
    np.random.seed(seed)
    f = pynbody.new(n_part)
    f['pos'] = np.random.normal(size=(n_part, 3))
    f['vel'] = np.random.normal(size=(n_part, 3))
    f['mass'] = np.ones(n_part) / n_part
    return f

def make_uniform_blob(n_part, seed=1337):
    np.random.seed(seed)
    f = pynbody.new(n_part)
    f['pos'] = np.random.uniform(-1.0, 1.0, size=(n_part, 3))
    badflag = f['r'] > 1.0
    while np.any(badflag):
        f['pos'][badflag] = np.random.uniform(-1.0, 1.0, size=(badflag.sum(), 3))
        badflag = f['r'] > 1.0
    f['vel'] = np.random.normal(size=(n_part, 3))
    f['mass'] = np.ones(n_part) / n_part
    return f
