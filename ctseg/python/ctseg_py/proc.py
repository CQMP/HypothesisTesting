"""Pre- and postprocessing functions"""
from __future__ import division

import numpy as np

from ctseg_py import program_str

# --- data file handling ---

def print_quantity(outf, data, last_axis=None, header=""):
    """Prints an array as a set of key/value pairs"""
    try:
        outf.write("# %s: %s\n" % (program_str, header))
    except AttributeError:
        outf = file(outf, "w")
        outf.write("# %s: %s\n" % (program_str, header))

    have_plotaxis = last_axis is not None
    if not have_plotaxis:
        last_axis = np.arange(data.shape[-1])
    if len(last_axis) != data.shape[-1]:
        raise ValueError("axis %d does not match shape %s",
                         last_axis.shape, data.shape)

    if have_plotaxis:
        fmt_str = ("%3d",) * (data.ndim-1) + ("%10.5f",)
    else:
        fmt_str = ("%3d",) * data.ndim
    if data.dtype.fields:
        data["sqerr"] = np.sqrt(data["sqerr"])
        if issubclass(data["mean"].dtype.type, np.complexfloating):
            data = np.rec.fromarrays(
                    [data["mean"].real, data["mean"].imag, data["sqerr"]],
                    names=["real", "imag", "err"])
    else:
        if issubclass(data.dtype.type, np.complexfloating):
            data = np.rec.fromarrays([data.real, data.imag],
                                     names=["real", "imag"])
        else:
            data = np.rec.fromarrays([data], names=["value"])

    fmt_str = fmt_str + ("%14.7g",) * len(data.dtype.fields)
    fmt_str = " ".join(fmt_str) + "\n"
    for idx, vals in np.ndenumerate(data):
        if have_plotaxis and idx[-1] == 0 and sum(idx[:-1]):
            outf.write("\n\n")
        if any(tuple(vals)):
            outf.write(fmt_str % (idx[:-1] + (last_axis[idx[-1]],) + tuple(vals)))

# --- pre- and post-processing ---

def promote_diag(arr):
    """Promotes the 0-th dimension of an array to a 0/1-diagonal"""
    diag_dim = arr.shape[0]
    newarr = np.zeros((diag_dim,) + arr.shape, arr.dtype)
    diag_rng = np.arange(diag_dim)
    newarr[diag_rng, diag_rng, ...] = arr
    return newarr

def get_gdisc4iw(giw, beta, n4iwb, n4iwf):
    """Construct  disconnected part of a two-particle Green's function"""
    if giw.shape[-1] < 2 * (n4iwf + n4iwb - 1):
        raise ValueError("Not enough frequencies is giw to construct g4iw.")

    nflavours = giw.shape[0]
    g4iw = np.zeros((nflavours,)*4 + (2 * n4iwb - 1, 2 * n4iwf, 2 * n4iwf),
                    giw.dtype)
    if not n4iwf or not n4iwb:
        return g4iw

    # Compute window restrictions for the two-particle part
    iv4start = (giw.shape[-1] - g4iw.shape[-1])/2
    iv4range = slice(iv4start, -iv4start)
    iv4idx = np.arange(2 * n4iwf)

    iw4eq0 = n4iwb - 1
    iw4vals = np.arange(-n4iwb + 1, n4iwb)
    assert iw4vals.size == g4iw.shape[-3]

    giw_T = giw.transpose(1, 0, 2)

    # Add straight term, given by: G_AB(iv) G_CD(iv') delta(w,0)
    g4iw[:,:,:,:,iw4eq0,:,:] += (giw[:,:,None,None,iv4range,None]
                                 * giw[None,None,:,:,None,iv4range])

    # Add cross term, given by: -G_AD(iv+iw) G_CB(iv) delta(v,v')
    for iiw4, iw4 in enumerate(iw4vals):
        iv4_plus_iw4 = slice(iv4start + iw4, -iv4start + iw4)
        g4iw[:,:,:,:,iiw4,iv4idx,iv4idx] -= (giw[:,None,None,:,iv4_plus_iw4]
                                             * giw_T[None,:,:,None,iv4range])

    g4iw *= beta
    return g4iw
