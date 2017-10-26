import sys
import optparse

import numpy as np

from ctseg_py import system
from ctseg_py import fourier

def parse_options(args=sys.argv, version_str=None):
    p = optparse.OptionParser(
            usage="%prog [--help] [OPTION ...] INPUT_FILE ...",
            version=version_str,
            description="CT-HYB(SEG) diagrammatic Anderson impurity solver",
            )

    g = optparse.OptionGroup(p, "Local impurity problem")
    g.add_option("--beta", type="float", metavar="VALUE",
                 help="inverse temperature")
    g.add_option("--nflavours", type="int", metavar="INT",
                 help="number of spin-orbitals")
    g.add_option("--mu", type="float", metavar="VALUE",
                 help="chemical potential")
    g.add_option("--eloc", type="string", metavar="FILE",
                 help="local one-particle Hamiltonian E (crystal field)")
    g.add_option("--u-value", type="float", metavar="VALUE",
                 help="intra-orbital Hubbard U")
    g.add_option("--v-value", type="float", metavar="VALUE",
                 help="inter-orbital Hubbard U'")
    g.add_option("--j-value", type="float", metavar="VALUE",
                 help="Hund's coupling J")
    g.add_option("--u-full", type="string", metavar="FILE",
                 help="full four-index U matrix (only orbitals)")
    g.add_option("--u-densities", type="string", metavar="FILE",
                 help="two-index density U matrix (orbital/spin)")
    g.add_option("--int-type", metavar="density|general",
                 choices=["density", "general"],
                 help="local interaction symmetries")
    p.add_option_group(g)

    g = optparse.OptionGroup(p, "Bath problem")
    g.add_option("--isolated", action="store_true",
                 help="Use an isolated system (no bath)"),
    g.add_option("--hybr-tau", type="string", metavar="FILE",
                 help="imaginary-time hybridisation function"),
    g.add_option("--hybr-iw", type="string", metavar="FILE",
                 help="hybridisation function on the Matsubara axis"),
    g.add_option("--hybr-mom", type="string", metavar="FILE",
                 help="moments of the Hybridisation function"),
    g.add_option("--bath-e", type="string", metavar="FILE",
                 help="discrete bath site energies")
    g.add_option("--bath-v", type="string", metavar="FILE",
                 help="discrete hybridisation strengths to impurity")
    g.add_option("--ntau-hybr", type="int", metavar="INT",
                 help="tau discretisation for the hybridisation function"),
    g.add_option("--hybr-repr", metavar="linear|bspline3",
                 choices=['linear', 'bspline3'], default='bspline3',
                 help="representation strategy for the hybridisation function")
    p.add_option_group(g)

    g = optparse.OptionGroup(p, "Simulation")
    g.add_option("--steps", type="int", metavar="INT",
                 help="number of Monte Carlo steps")
    g.add_option("--warmup", type="float", metavar="VALUE",
                 help="warm-up share", default=0.10)
    g.add_option("--batches", type="int", metavar="INT", default=10,
                 help="number of Monte Carlo batches or `bins'")
    g.add_option("--sweep-size", type="int", metavar="INT",
                 help="sweep size")
    g.add_option("--seg-maxlen", type="float", metavar="VALUE",
                 help="maximum length of segments")
    g.add_option("--seed", type="int", metavar="INT", default=4711,
                 help="QMC seed")
    g.add_option("--no-nfft", action="store_false", dest="nfft", default=True,
                 help="Do not use NFFT for the measurement of quantities")
    g.add_option("--quad-precision", action="store_true", default=False,
                 help="calculate with quad precision")
    p.add_option_group(g)

    g = optparse.OptionGroup(p, "Measurement")
    g.add_option("--niwf", type="int", metavar="INT",
                 help="number of positive fermionic Matsubara frequencies")
    g.add_option("--niwb", type="int", metavar="INT",
                 help="number of non-negative bosonic Matsubara frequencies")
    g.add_option("--ntau", type="int", metavar="INT",
                 help="tau discretisation")
    g.add_option("--g4iw-niwf", type="int", metavar="INT",
                 help="positive fermionic frequencies for two-partice GF")
    g.add_option("--g4iw-niwb", type="int", metavar="INT",
                 help="non-negative bosonic frequencies for two-particle GF")
    g.add_option("--meas-chiiw", action="store_true", default=False,
                 help="measure density susceptibility chi(iw)")
    g.add_option("--meas-g4iw", action="store_true", default=False,
                 help="measure two-particle Green's function G(iv, iv', iw)")
    g.add_option("--no-meas-giw", action="store_false", dest="meas_giw",
                 default=True, help="do not measure G(iw) directly")
    g.add_option("--no-meas-gsigmaiw", action="store_false", dest="meas_gsigmaiw",
                 default=True, help="do not measure GSigma(iw) directly")
    p.add_option_group(g)

    g = optparse.OptionGroup(p, "Post-processing and debugging")
    g.add_option("--siw", metavar="improved|dyson",
                 choices=["improved", "dyson"],
                 help="method to compute self-energy")
    g.add_option("--force-para", action="store_true", default=False,
                 help="(discouraged) force a paramagnetic solution")
    g.add_option("--title", type="string", metavar="PREFIX",
                 help="title of the run")
    g.add_option("--debug", action="store_true", default=False,
                 help="debugging mode")
    g.add_option("--timing", action="store_true", default=False,
                 help="record move-resolved performan ce data")
    g.add_option("--new", action="store_const", const="new", dest="run_type",
                 help="Create a new run (remove previous output)")
    g.add_option("--replace", action="store_const", const="replace", dest="run_type",
                 help="Override batches if they already exist")
    g.add_option("--add", action="store_const", const="add", dest="run_type",
                 help="Add data to previous run (ensure that it exists)")
    g.add_option("--setup", action="store_const", const="setup", dest="action",
                 help="Only set up the calculation, do not produce data")
    g.add_option("--run", action="store_const", const="run", dest="action",
                 help="Only run a previously set-up calculation")
    g.add_option("--errors", metavar="proper|simple|none",
                 choices=["proper", "simple", "none"], default="proper",
                 help="Quality of error analysis")
    g.add_option("--collect", action="store_const", const="collect", dest="action",
                 help="Collect results")
    g.add_option("--explain", action="store_true", default=False,
                 help=optparse.SUPPRESS_HELP)
    p.add_option_group(g)

    o, remaining = p.parse_args(args)
    if o.explain: import ctseg_py; ctseg_py.tl()

    # remaining
    if remaining[1:]:
        ifargs = []
        try:
            for inf in remaining[1:]:
                line = None
                for line in file(inf, "r"):
                    t = line[:line.find('#')].split(None, 2)
                    if not t: continue
                    ifargs.extend(("--" + t[0].replace("_", "-"), t[1]))
        except Exception, e:
            if not line: line = e
            p.error("error parsing input file `%s'\n\t%s" % (inf, line))
        args = ifargs + args
        o, _ = p.parse_args(args)

    def xor(**groups):
        def error():
            s = "Exactly one of the follwing option groups expected:\n"
            for key, grp in groups.iteritems():
                s += "  - %s: %s\n" % (key, ", ".join(n for n in grp))
            p.error(s)
        tkey = None
        for key, group in groups.iteritems():
            if sum(getattr(o, name) is not None for name in group):
                if tkey is not None: error()
                tkey = key
        if tkey is None: error()
        return tkey

    def ident(group, default=None):
        present = [getattr(o, name) is not None for name in group]
        npresent = sum(present)
        if npresent == 0 or npresent == len(group):
            return
        if default is not None:
            for name, pres, defval in zip(group, present, default):
                if not pres: setattr(o, name, defval)
            return
        p.error("Options missing:\n" + ", ".join(group))

    ident(["u_value", "v_value", "j_value"], [0., 0., 0.])
    ident(["bath_e", "bath_v"])
    ident(["hybr_iw", "hybr_mom"])
    o.u_choice = xor(values=["u_value", "v_value", "j_value"],
                     full=["u_full"],
                     dens=["u_densities"])
    o.hybr_choice = xor(discrete=["bath_e", "bath_v"],
                        from_iw=["hybr_iw", "hybr_mom"],
                        from_tau=["hybr_tau"],
                        none=["isolated"])

    # set default values
    try:
        if o.int_type is None:
            o.int_type = ("density", "general")[o.u_choice == "full"]
        if o.siw is None:
            o.siw = {"general":"dyson", "density":"improved"}[o.int_type]
    except KeyError, e:
        p.error("unable to detect some defaults: " % e)

    if o.meas_chiiw:
        if o.niwb is None:
            p.error("Expecting --niwb")

    if o.meas_g4iw:
        if o.g4iw_niwb is None or o.g4iw_niwf is None:
            p.error("Expecting --g4iw-niwb and --g4iw-niwf")

    return p, o, args

def read_datafile(in_file, header_lines=0, fast=False):
    """Reads a "rectangular" text data-file."""
    # read header
    header = []
    while True:
        # read the next line, if not first open the file
        try:
            line = in_file.readline()
        except AttributeError:
            in_file = file(in_file, 'r')
            line = in_file.readline()
        if not line:
            raise IOError("empty file: %s" % in_file)
        # ignore empty lines and comments in the header
        fields = line[:line.find('#')].split()
        if not fields:
            continue
        # it's not a comment, therefore it must be either header or data
        if header_lines:
            header_lines -= 1
            header.append(line)
        else:
            nfields = len(fields)
            first = np.asarray(map(float, fields))[None,:]
            break

    if fast:
        # read rest of data. This is done with fromfile for speed, which
        # unfortunately does not enforce a consistent shape of the data
        rest = np.fromfile(in_file, sep=" ").reshape(-1, nfields)
    else:
        rest = np.loadtxt(in_file, ndmin=2)

    result = np.vstack((first, rest))
    if header:
        return header_lines, result
    else:
        return result

def rmap(values, mapping, atol=None):
    """Performs an approximate reverse mapping of values to indices"""
    if (mapping.argsort() != np.arange(mapping.size)).any():
        raise ValueError("illegal mapping")
    atol = np.diff(mapping).min()/10
    mapping = np.repeat(mapping[:,None], 2, -1)
    mapping += -atol, +atol
    indices = mapping.reshape(-1).searchsorted(values, 'left')
    if not (indices % 2).all():
        problem = (indices % 2 == 0).nonzero()
        print values[problem], "\n", mapping
        raise ValueError("Inverse mapping failed: illegal values")
    return indices // 2

def array_from_pairs(keys, values, shape=None):
    """Generates an array from a list of keys and values"""
    # check for integers
    if (keys != keys.round()).any():
        raise ValueError("error converting to integer")
    keys = np.array(keys, np.int)
    # checking for proper indices [0 ... dim-1]
    if np.amin(keys) < 0:
        raise ValueError("negative indices not allowed")
    if shape is None:
        shape = np.amax(keys, axis=0) + 1
    # build up array
    full = np.zeros(shape, values.dtype)
    full[tuple(keys.T)] = values
    return full

def get_levels(opts):
    # fill site_eng with site energies: e0 - mu
    site_ham = np.zeros((opts.nflavours, opts.nflavours))

    if opts.mu is not None:
        np.fill_diagonal(site_ham, -opts.mu)

    if opts.eloc is not None:
        eloc_vals = read_datafile(file(opts.eloc, "r"))
        if eloc_vals.shape[-1] == 3:
            eloc = eloc_vals[:,2]
        elif eloc_vals.shape[-1] == 4:
            eloc = eloc_vals[:,2] + 1j * eloc_vals[:,3]
        else:
            raise ValueError("Illegal site levels file")

        site_ham += array_from_pairs(eloc_vals[:,:2], eloc, (opts.nflavours,)*2)

    return site_ham

def get_umatrix(opts):
    # fill u_dens with interaction term
    u_full = None
    if opts.u_choice == "values":
        args = opts.nflavours//2, opts.u_value, opts.v_value, opts.j_value
        u_full = system.ufull_kanamori(*args)
        u_dens = system.udensity_values(*args)
    elif opts.u_choice == "full":
        vals = read_datafile(file(opts.u_full, "r"))
        u_full = array_from_pairs(vals[:,:4], vals[:,4], (opts.nflavours//2,)*4)
        u_dens = system.udensity_from_ufull(u_full)
    elif opts.u_choice == "dens":
        vals = read_datafile(file(opts.u_densities, "r"))
        u_dens = array_from_pairs(vals[:,:2], vals[:,2], (opts.nflavours,)*2)
        u_full = system.ufull_from_udensity(u_dens)
    else:
        raise RuntimeError()

    return u_dens, u_full

def get_discrete_bath(opts):
    if opts.hybr_choice not in ("discrete", "none"):
        raise ValueError("Must have discrete bath")

    if opts.hybr_choice == "none":
        epsk = np.zeros(0)
        vki = np.zeros((0, opts.nflavours))
        return epsk, vki

    epsk_values = read_datafile(file(opts.bath_e, "r"))
    if epsk_values.shape[-1] == 1:
        epsk = epsk_values[:,0]
    elif epsk_values.shape[-1] == 2:
        epsk = epsk_values[:,0] + 1j * epsk_values[:,1]
    else:
        raise ValueError("Illegal epsk file")
    nbath = epsk.size

    vki_values = read_datafile(file(opts.bath_v, "r"))
    if vki_values.shape[-1] == 3:
        vki = vki_values[:,2]
    elif vki_values.shape[-1] == 4:
        vki = vki_values[:,2] + 1j * vki_values[:,3]
    else:
        raise ValueError("Illegal Vk file")
    vki = array_from_pairs(vki_values[:,:2], vki, (nbath, opts.nflavours))

    return epsk, vki

def get_hybr(opts):
    # Fall-back arrays that are overwritten
    oversampling = 4

    def set_niwf(niwf):
        if niwf is None:
            if opts.niwf is None:
                raise ValueError("Must specify --niwf")
        else:
            if opts.niwf is not None and niwf != opts.niwf:
                raise ValueError("--niwf is inconsistent with input function")
            opts.niwf = niwf

        if opts.ntau_hybr is None:
            opts.ntau_hybr = oversampling * opts.niwf
        return system.get_ffreq(opts.beta, opts.niwf)

    def set_ntau(ntau):
        if ntau is None:
            if opts.ntau_hybr is None:
                raise ValueError("Must specify --ntau-hybr")
        else:
            if opts.ntau_hybr is not None and ntau != opts.ntau_hybr:
                raise ValueError("--ntau-hybr is inconsistent with input function")
            opts.ntau_hybr = ntau

        if opts.niwf is None:
            opts.niwf = opts.ntau_hybr / oversampling
        return system.get_tau(opts.beta, opts.ntau_hybr)

    # fill hybr_tau and hybr_iw with Hybridisation function
    if opts.hybr_choice in ("isolated", "discrete"):
        # compute Hybridisation function from discrete bath
        iw_hybr = set_niwf(None)
        tau_hybr = set_ntau(None)

        epsk, vki = get_discrete_bath(opts)
        _, hybriw, hybrtau = system.hybr_from_sites(epsk, vki, iw_hybr, tau_hybr)

    elif opts.hybr_choice == "from_iw":
        # read in Hybridisation function on the Matsubara axis
        vals = read_datafile(file(opts.hybr_iw, "r"))
        moms = read_datafile(file(opts.hybr_mom, "r"))

        # extract the frequency axis
        read_ffreq = np.unique(vals[:,2].round(8))
        iw_hybr = set_niwf(read_ffreq.size/2)
        tau_hybr = set_ntau(None)
        iw_hybr = system.get_ffreq(opts.beta, opts.niwf)
        if not np.allclose(iw_hybr, read_ffreq, rtol=1e-4, atol=1e-4):
            raise ValueError("Illegal frequency axis")
        vals[:,2] = rmap(vals[:,2], iw_hybr)

        # check shape
        if vals.shape[-1] != 5:
            raise ValueError("Illegal hybriw file: too many columns")
        hybriw = array_from_pairs(vals[:,:3], vals[:,3] + 1j * vals[:,4])

        if moms.shape[-1] == 4:
            hybrmom = array_from_pairs(moms[:,:3], moms[:,3])
        elif moms.shape[-1] == 5:
            hybrmom = array_from_pairs(moms[:,:3], moms[:,3] + 1j * moms[:,4])
        else:
            raise ValueError("Illegal hybrmom file: too many columns")

        hybrmom = np.rollaxis(hybrmom, 2, 0)
        hybriw_model = fourier.pmodel_iw(hybrmom, iw_hybr)
        hybrtau_model = fourier.pmodel_tau(hybrmom, tau_hybr, opts.beta)
        hybrtau = fourier.tau_from_iw(hybriw - hybriw_model, opts.beta,
                                      tau_hybr.size) + hybrtau_model

    elif opts.hybr_choice == "from_tau":
        # read in Hybridisation function in tau
        vals = read_datafile(file(opts.hybr_tau, "r"))

        # detect number of tau points if not present
        read_tau = np.unique(vals[:,2].round(8))
        tau_hybr = set_ntau(read_tau.size - 1)
        iw_hybr = set_niwf(None)
        if not np.allclose(tau_hybr, read_tau):
            raise RuntimeError("Tau axis in hybrtau file is non-equispaced")
        vals[:,2] = rmap(vals[:,2], tau_hybr)

        # extract values
        if vals.shape[-1] == 4:
            hybrtau = vals[:,3]
        else:
            hybrtau = vals[:,3] + 1j * vals[:,4]

        hybrtau = array_from_pairs(vals[:,:3], hybrtau)

        hybrmom = fourier.moments_from_borders(hybrtau[...,0], hybrtau[...,-1],
                                               opts.beta)
        hybriw_model = fourier.pmodel_iw(hybrmom, iw_hybr)
        hybrtau_model = fourier.pmodel_tau(hybrmom, tau_hybr, opts.beta)
        hybriw = fourier.iw_from_tau(hybrtau - hybrtau_model, opts.beta,
                                     iw_hybr.size) + hybriw_model

    if opts.ntau is None:
        opts.ntau = opts.ntau_hybr

    return tau_hybr, hybrtau, iw_hybr, hybriw

def problem_from_config(opts):
    beta = opts.beta
    u_dens, _ = get_umatrix(opts)
    site_ham = get_levels(opts)
    tau_hybr, hybrtau, ffreq, hybriw = get_hybr(opts)
    return system.ImpurityProblem(beta, u_dens, site_ham, tau_hybr, hybrtau,
                                  ffreq, hybriw)

