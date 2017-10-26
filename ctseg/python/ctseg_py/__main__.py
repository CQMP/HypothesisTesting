#!/usr/bin/env python
"""CT-HYB(SEG) diagrammatic strong-coupling impurity solver

Part of the w2dynamics package.

Author: Markus Wallerberger
"""
from __future__ import division
import sys
import time
import os
import shutil

import numpy as np  # fail-fast if numpy is not available
import scipy.interpolate

import cthyb

from ctseg_py._npcompat import linalg_inv
from ctseg_py.config import parse_options, problem_from_config
from ctseg_py.proc import promote_diag, print_quantity
from ctseg_py import fourier, program_str

from ctseg_py.stat import pseudovalues

def format_sign(mean_sign):
    diff1 = np.abs(np.abs(mean_sign) - 1)
    if not diff1 or diff1 >= 5e-5:
        return "%9.4f  %s" % (mean_sign, ("","- WARNING!")[int(diff1 > .25)])
    else:
        return "   1 - %.2g" % diff1

def make_id(random):
    my_hash = 0
    for _ in range(64//random.ENTROPY):
        my_hash *= 1 << random.ENTROPY
        my_hash += random.raw()
    return hex(my_hash)[2:].rstrip('L').zfill(16)

def set_up(opts, system_dir, imp):
    # Creating result directory
    try:
        os.mkdir(system_dir)
    except OSError, e:
        # Result directory exists
        if e.errno != os.errno.EEXIST:
            raise
        if opts.run_type == 'new':
            print "Note: removing previous run"
            shutil.rmtree(system_dir)              # TODO: check for MPI
            os.mkdir(system_dir)
        elif opts.run_type == 'replace':
            print "Note: output directory exists, replacing any duplicate bins"
        elif opts.run_type != 'add':
            print "WARNING: previous run exists"
    else:
        # Restult directory does not exist

        if opts.run_type == 'add':
            raise RuntimeError("No previous run exists, so cannot --add")

    # Writing first stuff there
    print_quantity("%s/hybrtau.dat" % system_dir, imp.hybrtau, imp.tau_hybr,
                   "Hybridisation function in tau")
    print_quantity("%s/hybriw.dat" % system_dir, imp.hybriw, imp.ffreq,
                   "Hybridisation function in iw")

def spline_intp(y, k):
    # We need to augment the data, because splrep cuts away border points
    # otherwise. Best guess: same exponential coefficient
    b = k - 1
    yaug = np.zeros(y.size + 2 * b)
    yaug[b:-b] = y
    yaug[:b] = y[0] * (y[0]/y[1])**np.arange(b,0,-1)
    yaug[-b:] = y[-1] * (y[-1]/y[-2])**np.arange(1,b+1)
    xaug = np.arange(-b, y.size+b)

    # Get the corresponding coefficients
    poly = scipy.interpolate.PPoly.from_spline(
                scipy.interpolate.splrep(xaug, yaug, task=-1, t=xaug[b:-b], k=k))
    coeffs = poly.c[::-1, 2*b:-2*b]

    assert coeffs.shape == (k+1, y.size - 1), "Coefficients wrong shape"
    assert np.allclose(coeffs[0], y[:-1]), "Interpoland does not follow input"
    return coeffs.T

def get_hybr(strategy, hybrtau, beta):
    if strategy == 'linear':
        return cthyb.PolyHybFuncD.linear(hybrtau, beta, -1)
    elif strategy == 'bspline3':
        nflavours, ntau = hybrtau.shape[-2:]
        poly = np.reshape([spline_intp(y, 3)
                           for y in hybrtau.reshape(nflavours**2, ntau)],
                          (nflavours, nflavours, ntau-1, 4))
        return cthyb.PolyHybFuncD(poly, beta, -1)
    else:
        raise ValueError("Unknown hybridisation representation")

def get_simulation(opts, imp, seed=None):
    # Create simulation object
    # TODO: A lot more can be done here ...
    if seed is None:
        seed = opts.seed

    if imp.hybr_is_diagonal:
        blocks = [cthyb.BlockD(get_hybr(opts.hybr_repr, hybrtau_block[None,None],
                                        imp.beta))
                  for hybrtau_block in imp.hybrtau.diagonal().T]
    else:
        blocks = [cthyb.BlockD(get_hybr(opts.hybr_repr, imp.hybrtau, imp.beta))]

    bath_tr = cthyb.BathD(blocks)
    local_tr = cthyb.SegmentTraceD(imp.beta, imp.site_ham.diagonal(), imp.u_dens)
    diagram = cthyb.DiagramD(local_tr, bath_tr)

    # QMC simulation
    rnd = cthyb.DefaultRandom(seed)
    sim = cthyb.DriverD(diagram, rnd)

    estimators = {
        "gtau": cthyb.GtauEstimatorD(sim.current.bath, opts.ntau),
        "gsigmatau": cthyb.GSigmatauEstimatorD(sim.current, opts.ntau),
        "single_occ": cthyb.OccupationEstimatorD(sim.current.local, 1),
        "double_occ": cthyb.OccupationEstimatorD(sim.current.local, 2),
        "sign": cthyb.SignEstimatorD(),
        "order": cthyb.OrderEstimatorD(sim.current, 100),
    }
    if opts.meas_chiiw:
        estimators["nniw"] = cthyb.ChiiwEstimatorD(sim.current.local, opts.niwb,
                                                   opts.nfft)
    if opts.meas_giw:
        estimators["giw"] = cthyb.GiwEstimatorD(sim.current.bath, opts.niwf,
                                                opts.nfft)
    if opts.meas_gsigmaiw:
        estimators["gsigmaiw"] = cthyb.GSigmaiwEstimatorD(sim.current,
                                                          opts.niwf, opts.nfft)
    if opts.meas_g4iw:
        estimators["g4iw"] = cthyb.G4iwEstimatorD(sim.current.bath,
                                opts.g4iw_niwf, opts.g4iw_niwb, opts.nfft)

    return sim, estimators

def compute_batch(sim, estimators, sweeps, sweep_size):
    # reset all data to make sure we are doing nothing stupid
    sim.rates.reset()
    recompute = cthyb.DiaRecomputeD(sim.current)

    accumulators = {}
    for est_name, est in estimators.items():
        accumulators[est_name] = np.zeros(est.accum_size, est.accum_dtype)

    sum_sign = 0
    # compute some meta-quantities
    cpu_time = time.clock()
    batch_id = make_id(sim.random)

    sim.record_rates = True
    for isweep in xrange(sweeps):
        sim.sweep(sweep_size)
        curr_sign = sim.current.sign
        sum_sign += curr_sign
        for est_name, est in estimators.items():
            est.estimate(accumulators[est_name], curr_sign)

        if isweep % 100 == 0:
            recompute.propose()
            if recompute.error > 1e-8:
                print "WARNING: Loss of precision: %g" % recompute.error
            recompute.accept()

    # check that we are doing the right thing once in a while
    sim.current.verify()

    batch_data = {"#generator": program_str,
                  "#batch-hash": batch_id,
                  "cpu_time": time.clock() - cpu_time,
                  "rates": np.asarray(sim.rates.sum, float)
                  }
    for est_name, est in estimators.items():
        val = est.postprocess(accumulators[est_name], sum_sign)
        batch_data[est_name] = val.reshape(est.result_shape)

    return batch_data

def produce(opts, system_dir, imp, seed=None):
    sim, estimators = get_simulation(opts, imp, seed)
    warmup_share = opts.warmup
    nsteps = opts.steps
    ncorr = opts.sweep_size
    nwarmups = int(nsteps * warmup_share)

    if opts.run_type == 'replace':
        batch_file_opts = os.O_WRONLY | os.O_CREAT
    else:
        batch_file_opts = os.O_WRONLY | os.O_CREAT | os.O_EXCL

    try:
        ibatch = 0
        batch_id = "NULL"
        print "Thermalising (%d warmup steps) ..." % nwarmups
        sim.sweep(nwarmups)

        sweeps_per_bin = int(np.ceil((nsteps - nwarmups)/(ncorr * opts.batches)))
        print "Performing simulation (%d batches a %d sweeps a %d steps)..." % \
                        (opts.batches, sweeps_per_bin, ncorr)
        for ibatch in xrange(opts.batches):
            batch_data = compute_batch(sim, estimators, sweeps_per_bin, ncorr)

            batch_data["#previous-batch-hash"] = batch_id
            batch_id = batch_data["#batch-hash"]
            print "Batch %4d of %4d (ID: %s)..." % (ibatch+1, opts.batches, batch_id)

            try:
                batch_file = os.open("%s/batch-%s.npz" % (system_dir, batch_id),
                                     batch_file_opts, 0664)
            except OSError, e:
                if e.errno == os.errno.EEXIST:
                    print "ERROR: batch already exists (use --replace to override)"
                raise

            np.savez(os.fdopen(batch_file, 'w'), **batch_data)
            del batch_data

        print "\nLAST QMC CONFIGURATION:"
        sim.current.dump()
        print

    except KeyboardInterrupt:
        print " WARNING: simulation aborted after %d batches" % ibatch
        sys.exit()

def batch_file_names(system_dir):
    return ("%s/%s" % (system_dir, fname)
            for fname in os.listdir(system_dir) if fname.startswith("batch-"))

def postprocess_batch_data(data, imp, opts):
    single_occ = data['single_occ']
    double_occ = data['double_occ']
    order = data['order']
    mean_order = (order * np.arange(order.shape[-1])).sum(-1)

    gtau = data['gtau']
    gsigmatau = data['gsigmatau']

    # Perform all operations that are LINEAR in the measured quantities
    # (composition, addition, Fourier transform), because these should not
    # introduce bias in the estimators

    gtau = imp.upfold(gtau)
    gsigmatau = imp.upfold(gsigmatau)

    ntau = gtau.shape[-1]
    tau = (np.arange(ntau) + .5) * imp.beta/ntau

    occupancies = np.diag(single_occ) + double_occ

    # Border values
    # -G_ab(beta-) = <c^+_b c_a> = <n_a> delta_ab
    # -G_ab(0+) = <c_a c^+_b> = (1 - <n_a>) delta_ab
    # -GSigma_ab(beta-) = sum_j U_{bj} <n_j c_b^+ c_a>
    g_beta = -single_occ
    g_zero = -1 + single_occ

    gsigma_beta = -(imp.u_dens * occupancies).sum(-1)
    gsigma_zero = -gsigma_beta - (imp.u_dens * single_occ).sum(-1)

    g_beta = promote_diag(g_beta)
    g_zero = promote_diag(g_zero)

    gsigma_beta = promote_diag(gsigma_beta)
    gsigma_zero = promote_diag(gsigma_zero)

    gtau_aug = np.empty(gtau.shape[:-1] + (gtau.shape[-1] + 2,), gtau.dtype)
    gtau_aug[..., 0] = g_zero
    gtau_aug[..., 1:-1] = gtau
    gtau_aug[..., -1] = g_beta

    gsigmatau_aug = np.empty_like(gtau_aug)
    gsigmatau_aug[..., 0] = gsigma_zero
    gsigmatau_aug[..., 1:-1] = gsigmatau
    gsigmatau_aug[..., -1] = gsigma_beta

    g_mom = fourier.moments_from_borders(g_zero, g_beta, imp.beta)
    gtau_model = fourier.pmodel_tau(g_mom, tau, imp.beta)
    giw_model =  fourier.pmodel_iw(g_mom, imp.ffreq)
    giw = fourier.iw_from_taubins(gtau - gtau_model,
                                  imp.beta, imp.ffreq.size) + giw_model

    gsigma_mom = fourier.moments_from_borders(gsigma_zero, gsigma_beta, imp.beta)
    gsigmatau_model = fourier.pmodel_tau(gsigma_mom, tau, imp.beta)
    gsigmaiw_model =  fourier.pmodel_iw(gsigma_mom, imp.ffreq)
    gsigmaiw = fourier.iw_from_taubins(gsigmatau - gsigmatau_model,
                                       imp.beta, imp.ffreq.size) + gsigmaiw_model

    epot = imp.u_dens.dot(single_occ) + imp.site_ham.diagonal() * single_occ
    ekin = mean_order / -imp.beta

    if opts.meas_giw:
        giw_meas = data["giw"]
        giw_meas = imp.upfold(giw_meas)
        data["giw"] = giw_meas

    if opts.meas_gsigmaiw:
        gsigmaiw_meas = data["gsigmaiw"]
        gsigmaiw_meas = imp.upfold(gsigmaiw_meas)
        data["gsigmaiw"] = gsigmaiw_meas

    if opts.meas_g4iw:
        g4iw_str, g4iw_cr = data["g4iw"]
        g4iw_str = imp.upfold(imp.upfold(g4iw_str, 1), 2)  # A B C D iw iv iv'
        g4iw_cr = imp.upfold(imp.upfold(g4iw_cr, 1), 2)  # A D C B iw iv iv'
        g4iw_cr = g4iw_cr.transpose(0, 3, 2, 1, 4, 5, 6)
        g4iw = g4iw_str
        g4iw -= g4iw_cr
        data["g4iw"] = g4iw

    data["occupancies"] = occupancies
    data["gtau"] = gtau_aug
    data["gsigmatau"] = gsigmatau_aug
    data["giw_fromtau"] = giw
    data["gsigmaiw_fromtau"] = gsigmaiw
    data["epot"] = epot
    data["ekin"] = ekin
    data["mean_order"] = mean_order


def collect(batch_files, postprocessor=None, full=False):
    print "Collecting ..."
    data = None

    if not batch_files:
        raise ValueError("No batch files found ...")
    nbatches = len(batch_files)

    print "Aggregating %d batch files ..." % nbatches
    for batchno, fname in enumerate(batch_files):
        batch_file = np.load(fname)
        batch_data = dict(batch_file)
        if postprocessor is not None:
            postprocessor(batch_data)

        # Initialise the aggregates. This is where memory may get low for full.
        if data is None:
            if full:
                data = dict((name, np.empty((nbatches,) + dset.shape, dset.dtype))
                            for name, dset in batch_data.items()
                            if not name.startswith('#'))
            else:
                data = dict((name, np.zeros(dset.shape,
                                    [("mean", dset.dtype), ("sqerr", float)]))
                            for name, dset in batch_data.items()
                            if not name.startswith('#')
                            )

        # Put stuff into the aggregates
        for name, dset in batch_data.items():
            if name.startswith('#'):
                continue

            aggr = data[name]
            if full:
                aggr[batchno] = dset
            else:
                aggr["mean"] += dset
                aggr["sqerr"] += np.abs(dset)**2

    # Post-process mean and squared error
    if not full:
        for name, aggr in data.items():
            aggr["mean"] /= nbatches
            aggr["sqerr"] /= nbatches
            aggr["sqerr"] -= np.abs(aggr["mean"])**2
            aggr["sqerr"] /= nbatches - 1

    return data

def get_error(*data, **kwds):
    func = kwds.get('func', None)
    pre_aggregated = bool(data[0].dtype.fields)

    if pre_aggregated:
        if func is None:
            data, = data
            return data
        else:
            mean = func(*(dset["mean"] for dset in data))
            sqerr = np.nan*np.zeros(mean.shape, float)
    else:
        # Jackknife procedure: transform data to their pseudovalues
        if func is None:
            data, = data
        else:
            data = pseudovalues(lambda *x: func(*(i.mean(0) for i in x)), *data)

        mean = data.mean(0)
        sqerr = data.var(0)/(data.shape[0] - 1.)

    return np.rec.fromarrays([mean, sqerr], names=['mean', 'sqerr'])

def para_symmetrize(aiw):
    nflv, niw = aiw.shape[-2:]
    ssymm, tsymm = np.ix_((1,0), (1,0))
    aiw = aiw.reshape(nflv//2, 2, nflv//2, 2, niw)
    aiw = .5 * (aiw + aiw[:, ssymm, :, tsymm, :].transpose(2, 0, 3, 1, 4))
    return aiw.reshape(nflv, nflv, niw)

def postprocess(data, opts, imp):
    print "Postprocessing ..."
    def calc_giw_inv(giw):
        if imp.hybr_is_diagonal:
            return promote_diag(1/giw.diagonal().T)
        else:
            return np.transpose([linalg_inv(g) for
                                 g in giw.transpose(2, 0, 1)], (1, 2, 0))

    def calc_sigmaiw_dyson(giw):
        if opts.force_para:
            giw = para_symmetrize(giw)

        return g0iw_inv - calc_giw_inv(giw)

    def calc_sigmaiw_impr(giw, gsigmaiw):
        if opts.force_para:
            giw = para_symmetrize(giw)
            gsigmaiw = para_symmetrize(gsigmaiw)

        return np.einsum('ijW,jkW->ikW', calc_giw_inv(giw), gsigmaiw)

    def calc_zfactor(siw):
        iwpos = slice(siw.shape[-1]//2, None)
        imsiw_diag = siw.diagonal(0, -3, -2).T[..., iwpos].imag
        slope = imsiw_diag[...,1] - imsiw_diag[...,0]
        slope[slope >= 0] = np.nan
        return 1./(1. - slope)

    def calc_chiiw(nniw, occupancies):
        fillings = occupancies.diagonal()
        chiiw = nniw.copy()
        chiiw[..., 0] -= imp.beta * fillings[:, None] * fillings[None, :]
        return chiiw

    # Writes a bunch of stuff with errors
    edata = dict((name, get_error(dset)) for name, dset in data.items())

    g0iw_inv = (1j * imp.ffreq[None,None,:] * np.eye(imp.nflavours)[:,:,None]
                - imp.site_ham[:,:,None] - imp.hybriw[:,:,::-1])

    try:
        giw = data['giw']
    except KeyError:
        giw = data['giw_fromtau']

    try:
        gsigmaiw = data['gsigmaiw']
    except KeyError:
        gsigmaiw = data['gsigmaiw_fromtau']

    sigmaiw_impr = get_error(giw, gsigmaiw, func=calc_sigmaiw_impr)
    sigmaiw_dyson = get_error(giw, func=calc_sigmaiw_dyson)
    zfactor = get_error(sigmaiw_impr, func=calc_zfactor)

    if 'nniw' in data:
        edata['chiiw'] = get_error(data['nniw'], data['occupancies'],
                                   func=calc_chiiw)

    edata['sigmaiw_dyson'] = sigmaiw_dyson
    edata['sigmaiw_impr'] = sigmaiw_impr
    edata['zfactor'] = zfactor
    edata['g0iw_inv'] = g0iw_inv
    return edata

def write_results(edata, imp, opts, system_dir):
    # First, write results in numpy-friendly format
    np.savez("%s/results.npz" % system_dir, **edata)

    # Now, write it in human-friendly format
    gtau = edata['gtau']
    gsigmatau = edata['gsigmatau']
    ekin = edata['ekin']
    epot = edata['epot']
    occupancies = edata['occupancies']
    single_occ = occupancies.diagonal()
    order = edata['order']
    rates = edata['rates']
    cpu_time = edata['cpu_time']
    sign = edata['sign']
    sigmaiw_dyson = edata['sigmaiw_dyson']
    sigmaiw_impr = edata['sigmaiw_impr']
    zfactor = edata['zfactor']
    g0iw_inv = edata['g0iw_inv']
    mean_order = edata['mean_order']

    try:
        giw = edata['giw']
    except KeyError:
        giw = edata['giw_fromtau']

    try:
        gsigmaiw = edata['gsigmaiw']
    except KeyError:
        gsigmaiw = edata['gsigmaiw_fromtau']

    gtau_beta_half = gtau.diagonal()[gtau.shape[-1]//2]["mean"]

#    sim_steps = opts.nsteps #nwarmups + sim.qmc_stat.count()
    rates = rates["mean"]
    moves_per_sec = rates.sum()/cpu_time['mean']
    total_accept_rate = rates[:,2].sum()/rates.sum()
    accept_rates = rates[:,2]/rates.sum(1)
    hard_rates = rates[:,0]/rates.sum(1)

    maxorder = order.shape[-1] - (order['mean'][..., ::-1] != 0).argmax(-1).max()
    order = order[..., :maxorder+1]
    order_axis = np.arange(maxorder+1)

    titles = "insert", "remove"

    tau = (np.concatenate(([0], np.arange(0, opts.ntau) + .5, [opts.ntau]))
           * imp.beta/opts.ntau)

    print "\nQMC METRICS:"
    print "%22s =%s" % ("mean sign", format_sign(sign['mean']))
    print "%22s =%9.0f moves/cpusec" % ("performance", moves_per_sec)
    print "%22s =%9.2f moves" % ("autocorr estimate",
                                 mean_order["mean"].sum()/total_accept_rate)

    print
    print "    move type     accepts   hard rej.   prop time   acc. time"
    print "   ----------- ----------- ----------- ----------- -----------"
    for i, title in enumerate(titles):
        print "    %-10s%9.2f %% %9.2f %% %11s %11s" % \
            (title, 100 * accept_rates[i], 100 * hard_rates[i], "N/A", "N/A")

    print "\nSYSTEM:"
    print "      flavour     filling   G(beta/2)    Z-factor       E-kin       E-pot"
    print "   ----------- ----------- ----------- ----------- ----------- -----------"
    for flv in range(imp.nflavours):
        flvstr = "%d-%-4s" % (flv//2+1, ("up","down")[flv%2])
        print " %12s %11.4f %11.4f %11.4f %11.3f %11.3f" % \
              (flvstr, single_occ["mean"][flv], gtau_beta_half[flv],
               zfactor["mean"][flv], ekin["mean"][flv], epot["mean"][flv])
    print "               -----------                         ----------- ----------"
    print " %12s %11.4f %11s %11s %11.3f %11.3f" % \
              ("", single_occ["mean"].sum(), "", "", ekin["mean"].sum(), epot["mean"].sum())

    # WRITE DEBUG OUTPUT
    print "\nOutput ..."
    print_quantity("%s/gtau.dat" % system_dir, gtau, tau,
                   "impurity Green's function in tau")
    print_quantity("%s/giw.dat" % system_dir, giw, imp.ffreq,
                   "impurity Green's function in iw")
    if opts.siw == "improved":
        print_quantity("%s/gsigmatau.dat" % system_dir, gsigmatau, tau,
                       "impurity improved estimator (G*Sigma) in tau")
        print_quantity("%s/gsigmaiw.dat" % system_dir, gsigmaiw, imp.ffreq,
                       "impurity improved estimator (G*Sigma) in iw")

    if opts.force_para:
        print_quantity("%s/gtau_symm.dat" % system_dir, para_symmetrize(gtau['mean']), tau,
                       "impurity Green's function in tau, spin-symmetrized")
        print_quantity("%s/giw_symm.dat" % system_dir, para_symmetrize(giw['mean']), imp.ffreq,
                       "impurity Green's function in iw, spin-symmetrized")
    if opts.siw == "improved" and opts.force_para:
        print_quantity("%s/gsigmatau_symm.dat" % system_dir, para_symmetrize(gsigmatau['mean']), tau,
                       "impurity improved estimator (G*Sigma) in tau, spin-symmetrized")
        print_quantity("%s/gsigmaiw_symm.dat" % system_dir, para_symmetrize(gsigmaiw['mean']), imp.ffreq,
                       "impurity improved estimator (G*Sigma) in iw, spin-symmetrized")

    print_quantity("%s/occupations.dat" % system_dir, occupancies, None,
                   "single and double occupations <n_i n_j>")
    print_quantity("%s/order.dat" % system_dir, order, order_axis,
                   "histogram of expansion order <k>")
    print_quantity("%s/sigmaiw_impr.dat" % system_dir, sigmaiw_impr, imp.ffreq,
                   "self-energy on the Matsubara axis from improved estimators")
    print_quantity("%s/sigmaiw_dyson.dat" % system_dir, sigmaiw_dyson, imp.ffreq,
                   "self-energy on the Matsubara axis from Dyson equation")
    print_quantity("%s/g0iw_inv.dat" % system_dir, g0iw_inv, imp.ffreq,
                   "inverse of non-interacting impurity Greens function in iw")

    if 'chiiw' in edata:
        print_quantity("%s/chiiw.dat" % system_dir,
                       edata['chiiw'], (2*np.pi/opts.beta)*np.arange(opts.niwb),
                       "Density susceptibility chi_nn in iw")


def main():
    param, opts, args = parse_options()
    imp = problem_from_config(opts)
    print "System ID:", imp.hash
    if opts.title is None: opts.title = 'cthyb'
    system_dir = opts.title + '-' + imp.hash[:8]
    print "Result directory:", system_dir

    if opts.action is None or opts.action == 'setup':
        set_up(opts, system_dir, imp)

    if opts.action is None or opts.action == 'run':
        produce(opts, system_dir, imp)

    if opts.action is None or opts.action == 'collect':
        batch_files = list(batch_file_names(system_dir))
        data = collect(batch_files,
                       lambda data: postprocess_batch_data(data, imp, opts),
                       opts.errors == 'proper')
        edata = postprocess(data, opts, imp)
        write_results(edata, imp, opts, system_dir)

if __name__ == '__main__':
    main()
