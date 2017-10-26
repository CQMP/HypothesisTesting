# ---------------------------------------------------------------------------
# Parameters for the Makefile

include configs/default.mk
include config.mk

# ---------------------------------------------------------------------------
# Configuration part for the targets

ifeq ($(HAVE_BUNDLED),1)
	CPPFLAGS += -Ibundled/include
	LDFLAGS += -Lbundled/lib $(LD_SET_RUNPATH)bundled/lib
endif

CPPFLAGS += -DHAVE_EIGEN=$(HAVE_EIGEN)
ifeq ($(HAVE_EIGEN),1)
    ifneq ($(EIGEN_INCLUDE),)
        CPPFLAGS += -I$(EIGEN_INCLUDE)
    endif
endif

CPPFLAGS += -DHAVE_PYTHON=$(HAVE_PYTHON)
ifeq ($(HAVE_PYTHON),1)
    ifneq ($(PYTHON_INCLUDE),)
        CPPFLAGS += -I$(PYTHON_INCLUDE)
    endif
    ifneq ($(NUMPY_INCLUDE),)
        CPPFLAGS += -I$(NUMPY_INCLUDE)
    endif
    ifneq ($(PYTHON_LIBDIR),)
        LDFLAGS += -L$(PYTHON_LIBDIR) $(LD_SET_RUNPATH)$(PYTHON_LIBDIR)
    endif
    LDFLAGS += -l$(PYTHON_LIBNAME)
endif

CPPFLAGS += -DHAVE_NFFT=$(HAVE_NFFT)
ifeq ($(HAVE_NFFT),1)
    ifneq ($(NFFT_INCLUDE),)
        CPPFLAGS += -I$(NFFT_INCLUDE)
    endif
    ifneq ($(NFFT_LIBDIR),)
        LDFLAGS += -L$(NFFT_LIBDIR) $(LD_SET_RUNPATH)$(NFFT_LIBDIR)
    endif
    LDFLAGS += -l$(NFFT_LIBNAME)
endif

ifeq ($(MODE),debug)
    CPPFLAGS += $(CPPFLAGS_DEBUG)
    CXXFLAGS += $(CXXFLAGS_DEBUG)
    SANEFLAGS := $(SANEFLAGS_DEBUG)
else ifeq ($(MODE),release)
    CPPFLAGS += $(CPPFLAGS_RELEASE)
    CXXFLAGS += $(CXXFLAGS_RELEASE)
    SANEFLAGS := $(SANEFLAGS_RELEASE)
else
    $(error Invalid MODE=$(MODE); must be either 'debug' or 'release')
endif

# ---------------------------------------------------------------------------
# Files

SWIG_BUNDLED := swig/numpy.inc.i
SWIG_HEADERS := $(wildcard swig/*.inc.i)
SWIG_HEADERS := $(filter-out $(SWIG_BUNDLED),$(SWIG_HEADERS))
SWIG_FILES := $(wildcard swig/*.i)
SWIG_FILES := $(filter-out $(SWIG_HEADERS) $(SWIG_BUNDLED),$(SWIG_FILES))
SWIG_PYWRAPPERS := $(patsubst swig/%.i,python/%.py,$(SWIG_FILES))
SWIG_CXXWRAPPERS := $(patsubst swig/%.i,swig/%.cc,$(SWIG_FILES))
SWIG_OBJECTS := $(patsubst swig/%.i,swig/%.o,$(SWIG_FILES))
SWIG_MODULES := $(patsubst swig/%.i,python/_%.so,$(SWIG_FILES))

CXX_HEADERS := $(wildcard src/*.hh)
CXX_TSOURCES := $(wildcard src/*.tcc)
CXXTEST_FILES := $(wildcard src/*.test.cc)
CXXTEST_OUT := $(patsubst src/%.test.cc,%.test,$(CXXTEST_FILES))

PY_MODULES := $(wildcard python/ctseg_py/*.py) $(wildcard python/manybody/*.py)
PY_SCRIPTS := python/ctseg python/ed.py
PY_TESTS := $(wildcard test/*.py)

ALLSOURCES := $(CXX_HEADERS) $(CXX_TSOURCES) $(SWIG_FILES) $(SWIG_HEADERS)
ALLSOURCES += $(PY_MODULES) $(PY_SCRIPTS) $(PY_TESTS) Makefile

ifeq ($(MODE),debug)
    DEBUG_PRECIOUS := $(SWIG_CXXWRAPPERS)
endif

MODEFILES := $(wildcard .*.mode)
EXPECTED_MODEFILE := .$(MODE).mode
ifneq ($(MODEFILES),$(EXPECTED_MODEFILE))
    OLDMODE := $(patsubst .%.mode,%,$(MODEFILES))
    SWITCHMODE := switchmode
    $(warning Switching from MODE=$(OLDMODE) to $(MODE))
endif

# ---------------------------------------------------------------------------
# Targets

all: modules testprog

dist: $(SWIG_CXXWRAPPERS) bundle

modules: modecheck $(SWIG_MODULES)

testprog: modecheck $(CXXTEST_OUT)

%.test: src/%.test.cc $(CXX_HEADERS) $(CXX_TSOURCES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SANEFLAGS) -o $@ $< $(LDFLAGS)

swig/%.cc: swig/%.i $(SWIG_HEADERS) $(CXX_HEADERS) $(CXX_TSOURCES)
	$(SWIG) $(SWIGFLAGS) -outdir python -o $@ $<

swig/%.o: swig/%.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

python/_%.so: swig/%.o
	$(LD) -shared $^ -o $@ $(LDFLAGS)

modecheck: $(SWITCHMODE)

switchmode: clean-objects $(EXPECTED_MODEFILE)

$(EXPECTED_MODEFILE):
	touch $@

bundle: bundled_eigen

bundled_eigen: bundled/include/Eigen

bundled/include/Eigen: bundled/eigen.tar.gz
	mkdir -p bundled/include
	tar -xzf $< -C bundled eigen-*/Eigen
	name=`ls -d bundled/eigen-* | head -n1` && mv $$name/* bundled/include && rmdir $$name

bundled/eigen.tar.gz: bundled
	$(DOWNLOAD) $(EIGEN_URL) > $@ || rm $@

bundled:
	mkdir -p bundled

clean: clean-dist clean-objects clean-setuptools

wipe: clean clean-config

clean-dist:
	-rm -f $(SWIG_CXXWRAPPERS) $(SWIG_PYWRAPPERS)
	-rm -rf bundled/*

clean-objects:
	-rm -f $(SWIG_OBJECTS) $(SWIG_MODULES) $(CXXTEST_OUT) $(MODEFILES)

clean-config:
	-rm -f config.mk

clean-setuptools:
	-rm -rf *.egg-info/ build/ dist/

fix-whitespace:
	sed -i -e 's/[ \t]*$$//' -- $(ALLSOURCES)

loc-count:
	wc -l -- $(ALLSOURCES)

test: modules
	PYTHONPATH=$$PWD/python:$$PYTHONPATH $(PYTHON) test/__init__.py

.PHONY: all bundle clean clean-gensrc clean-objects modules testprog test \
        fix-whitespace modecheck switchmode

.PRECIOUS: $(DEBUG_PRECIOUS)

config.mk:
	@echo "# Uncomment the corresponding lines in this file to make local" >$@
	@echo "# modifications." >> $@
	@echo "# It will not be overridden when the Makefile changes" >> $@
	@echo >> $@
	sed -e 's/^##.*$$//' -e 's/^/# /' configs/default.mk >> $@


# ---------------------------------------------------------------------------
# Cancel all the implicit rules for speed and cleaner `make -d' output

.SUFFIXES:
%: %,v
%: s.%
%: RCS/%,v
%: RCS/%
%: SCCS/s.%
