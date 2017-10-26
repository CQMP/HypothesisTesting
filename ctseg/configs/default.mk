## Default parameters for the Makefile

## DO NOT edit this file if you just need to modify the parameter values;
## modify config.mk instead (it is not written back to the repository).
## config.mk is automatically generated from this file the first time you
## run make.  It will not be overwritten by subsequent executions of make
## unless you run `make config.mk'.

CPPFLAGS :=
CPPFLAGS_DEBUG := -DMODE_DEBUG=1 -D_GLIBCXX_DEBUG
CPPFLAGS_RELEASE := -DMODE_DEBUG=0 -DNDEBUG

CXX := g++
CXXFLAGS := -fPIC -std=c++0x -pedantic -Wall -Wno-long-long
CXXFLAGS_DEBUG := -g
CXXFLAGS_RELEASE := -O3 -ffast-math

SANEFLAGS_DEBUG := -fsanitize=address,undefined
SANEFLAGS_RELEASE :=

LD := g++
LDFLAGS := -Wl,-z,defs
LD_SET_RUNPATH := -Wl,-rpath=

SWIG := swig
SWIGFLAGS := -c++ -python

PYTHON := python

MODE := release

HAVE_EIGEN := 1
EIGEN_INCLUDE := /usr/include/eigen3

HAVE_PYTHON := 1
PYTHON_INCLUDE := /usr/include/python2.7
NUMPY_INCLUDE :=
PYTHON_LIBDIR :=
PYTHON_LIBNAME := python2.7

HAVE_NFFT := 0
NFFT_INCLUDE := /usr/local/include
NFFT_LIBDIR := /usr/local/lib
NFFT_LIBNAME := nfft3

DOWNLOAD := wget -O -

HAVE_BUNDLED := 1
EIGEN_URL := https://bitbucket.org/eigen/eigen/get/3.2.9.tar.gz
