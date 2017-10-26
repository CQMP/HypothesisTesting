from __future__ import division
import sys
import os
import numpy as np

try:
    from IPython.core.display import clear_output
except ImportError:
    def clear_output(wait=False): pass

def termsize():
    return tuple(map(int, os.popen('stty size', 'r').read().split()))

def create_buffer(rows=None, cols=None, out=sys.stdout):
    if rows is None or cols is None:
        if out.isatty(): tsize = termsize()
        else: tsize = 24, 80
        if rows is None: rows = tsize[0]
        if cols is None: cols = tsize[1]
    buf = np.empty((rows, cols), np.character)
    buf[...] = " "
    buf[:, -1] = "\n"
    return buf

def progress(seq, total=None, outfile=sys.stdout):
    if total is None:
        total = len(seq)
    seq = iter(seq)

    # Set up the progress bar
    barticks = 70
    bar = np.empty(barticks, np.character)
    bar[:] = ' '
    def put_bar(ielem):
        bartick = ielem * barticks // total
        percentage = ielem * 100 / total
        clear_output()
        bar[:bartick] = '*'
        outfile.write('\r%3d%% [%s]' % (percentage, ''.join(bar)))
        outfile.flush()

    # Compute number of ticks
    tick_len = np.maximum(total//100, 1)

    # Cycle through the elements
    try:
        for ielem in xrange(0, total, tick_len):
            put_bar(ielem)
            for _ in xrange(tick_len):
                yield next(seq)
    except StopIteration:
        pass
    else:
        # Ensure that we return the rest if the total was wrong
        put_bar(total)
        outfile.write('\n')
        for elem in seq:
            yield elem

    clear_output()

def draw(buf):
    print "".join(buf.flat)

def _pairs(x0, y0, x1, y1):
    if x1 < x0:
        return _pairs(x1, y1, x0, y0)
    slope = (y1 - y0)/(x1 - x0)
    yoffset = y0 - x0 * slope
    x = np.arange(np.round(x0), np.round(x1)+1, dtype=int)
    y = yoffset + x * slope
    p = np.empty(x.shape, np.character)
    return x, y, p

def get_line(r0, c0, r1, c1):
    if np.abs(c1 - c0) > np.abs(r1 - r0):
        # going chiefly left/right (column-wise)
        c, rreal, p = _pairs(c0, r0, c1, r1)
        r = np.array(rreal + 1/6., np.int)
        p[...] = '-'
        p[np.abs(rreal - r) > 1/3.] = '_'
    else:
        # going chiefly up/down (row-wise)
        r, c, p = _pairs(r0, c0, r1, c1)
        c = np.array(c + 1/2., np.int)
        diff = c[1:] - c[:-1]
        p[...] = '|'
        p[:-1][diff > 0] = '\\'
        p[1:][diff > 0] = '\\'
        p[:-1][diff < 0] = '/'
        p[1:][diff < 0] = '/'
    return np.rec.fromarrays((r, c, p), names=("r", "c", "p"))

def plot_xy(buf, X, Y, xmin=None, xmax=None, ymin=None, ymax=None):
    if xmin is None: xmin = X.min()
    if xmax is None: xmax = X.max()
    if ymin is None: ymin = Y.min()
    if ymax is None: ymax = Y.max()

    R = (buf.shape[0] - 1)/(ymax - ymin) * (ymax - Y)
    C = (buf.shape[1] - 1)/(xmax - xmin) * (X - xmin)
    Out = []
    oldr, oldc = R[0], C[0]
    for currr, currc in np.transpose((R[1:], C[1:])):
        Out.append(get_line(oldr, oldc, currr, currc))
        oldr, oldc = currr, currc

    Out = np.hstack(Out)
    buf[Out["r"], Out["c"]] = Out["p"]

# def spy(mat, buffer, title='matrix'):
#     """Visualises a large sparse matrix by plotting it in text mode"""
#     from scipy import sparse
#     row_factor = int(np.ceil(mat.shape[0]/(max_box[0] + 1)))
#     col_factor = int(np.ceil(mat.shape[1]/max_box[1]))
#     row_box = int(np.ceil(mat.shape[0]/row_factor))
#     col_box = int(np.ceil(mat.shape[1]/col_factor))
#
#     rows, cols, values = sparse.find(mat)
#     nelements = values.size
#     #values[...] = 1
#     values = np.abs(values)
#     groups = rows//row_factor*col_box + cols//col_factor
#     values, groups = _sum_by_group(values, groups)
#
#     weights = np.zeros(row_box*col_box)
#     weights[groups] = values
#     weights = weights.reshape(row_box, col_box)
#
#     disp = OutputBuffer(out, max_box)
#     disp.chars[-1] = '_'
#     disp.chars[weights != 0] = '.'
#
#     weights /= np.amax(weights)
#     signs = ((.001,  '-'), (.01,  'o'), (.1, 'O'),
#              (.25, '*'), (.50, '$'), (.75, '@'), (.875, '#'))
#     for threshold, sign in signs:
#         disp.chars[weights > threshold] = sign
#
#     title = (" %s: %s %s, %d elements (%.3f%%) " %
#              (title, ' x '.join(map(str,mat.shape)), str(mat.dtype),
#               nelements, 100*nelements/np.prod(mat.shape)))
#     print >> out, title.center(disp.chars.shape[1], '_')
#     for drow in disp:
#         print >> out, ''.join(drow)
