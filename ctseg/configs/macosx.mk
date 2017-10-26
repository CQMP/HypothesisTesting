# Configuration for MacOS X (tested for El Capitan)
#
# Requirements:
# -------------
#  1. Anaconda Python stack (version 2.7) <https://www.continuum.io/downloads>
#  2. MacPorts package manager <https://www.macports.org/install.php>
#  3. Eigen3 library: `sudo port install eigen3-devel'
#
# Optional:
# ---------
#  4. Swig interface generator: `sudo port install swig-python'
#  5. NFFT library: `sudo port install nfft-3'

MACPORTS_PREFIX := /opt/local
ANACONDA_PREFIX := $(HOME)/anaconda

LDFLAGS :=

null :=
LD_SET_RUNPATH := -Xlinker -rpath -Xlinker $(null)

SWIG := $(MACPORTS_PREFIX)/bin/swig

EIGEN_INCLUDE := $(MACPORTS_PREFIX)/include/eigen3

PYTHON_INCLUDE := $(ANACONDA_PREFIX)/include/python2.7
NUMPY_INCLUDE := $(ANACONDA_PREFIX)/lib/python2.7/site-packages/numpy/core/include
PYTHON_LIBDIR := $(ANACONDA_PREFIX)/lib

HAVE_NFFT := 1
NFFT_INCLUDE := $(MACPORTS_PREFIX)/include
NFFT_LIBDIR := $(MACPORTS_PREFIX)/lib

DOWNLOAD := curl
