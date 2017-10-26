"""Provides auxilliary and compatibility methods"""
import numpy as _np
import numpy.linalg as _linalg

numpy_version = tuple(map(int, _np.__version__.split(".")[:2]))

# --- NUMPY COMPATIBILITY MEHTHODS ---

# Workaround for ndarray.dot for numpy < 1.5
if numpy_version >= (1, 5):
    empty = _np.empty
else:
    class CompatibilityArray(_np.ndarray):
        def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                    strides=None, order=None, info=None):
            # Create ndarray instance of our type. Triggers __array_finalize__
            return _np.ndarray.__new__(cls, shape, dtype, buffer, offset,
                                      strides, order)

        def dot(self, b):
            return _np.dot(self, b)

    def empty(shape, dtype=float, order='C'):
        return CompatibilityArray(shape, dtype, order=order)

# Workaround for minlength argument of _np.np_bincount for numpy < 1.6
if numpy_version >= (1, 6):
    bincount = _np.bincount
else:
    def bincount(x, weights, minlength):
        if not x: return _np.zeros(minlength, _np.asarray(x).dtype)
        x = _np.bincount(x, _np.asarray(weights, _np.double))
        xfull = _np.zeros(minlength, x.dtype)
        xfull[:x.size] = x
        return xfull

if numpy_version >= (1, 8):
    linalg_inv = _linalg.inv
else:
    def linalg_inv(a):
        try:
            return _linalg.inv(a)
        except _linalg.LinAlgError:
            # Work around problem with 0-by-0 matrices in numpy < 1.8
            a = _np.asarray(a)
            if not a.size:
                return a.copy()
            # Work around problem with > 2D matrices in numpy < 1.8
            if a.ndim > 2:
                return _np.reshape([_linalg.inv(p) for p
                                    in a.reshape(-1, a.shape[-2:])], a.shape)
            # Probably just numeric problem, just raise then
            raise

def linalg_det(a):
    try:
        return _linalg.det(a)
    except _linalg.LinAlgError:
        # Work around problem with 0-by-0 matrices in numpy (up to current!)
        a = _np.asarray(a)
        if not a.size:
            return _np.ones(a.shape[:-2])
        # Work around problem with > 2D matrices in numpy < 1.8
        if a.ndim > 2:
            return _np.reshape([_linalg.det(p) for p
                                in a.reshape(-1, a.shape[-2:])], a.shape[:-2])
        # Probably just numeric problem, just raise then
        raise

