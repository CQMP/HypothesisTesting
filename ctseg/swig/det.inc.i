%{
#include "../src/det.hh"
%}

// Include stuff

template <typename ValueT = double>
class Determinant
{
public:
    static const unsigned DEFAULT_CAPACITY = 8, CAPACITY_RESERVE = 1;

public:
    Determinant(unsigned capacity = DEFAULT_CAPACITY);

    Determinant(const Determinant<ValueT> &other);

    void verify(double tol = 1e-6) const;

    // This is a bit tricky: we want to map a C++ method that returns an array
    // to Python property that does the same.  However, numpy.i only supports
    // wrapping function with outward *arguments*, so we first wrap the invmat()
    // method with _get_invmat_buffer.
    // (see http://docs.scipy.org/doc/numpy/reference/swig.interface-file.html)
    %extend {
        void _get_invmat_buffer(ValueT **ARGOUTVIEW_FARRAY2, int *DIM1, int *DIM2) {
            // Mapping function to the numpy.i interface
            *ARGOUTVIEW_FARRAY2 = self->invmat();
            *DIM1 = *DIM2 = self->capacity();
        }
    }

    ~Determinant();

    %pythoncode %{
        def _get_invmat(self):
            """Extract relevant portion of the inverse matrix buffer"""
            return self._get_invmat_buffer()[:self.order, :self.order]

        __swig_getmethods__["invmat"] = _get_invmat
        if _newclass: invmat = property(_get_invmat)

        __swig_getmethods__["_buffer"] = _get_invmat_buffer
        if _newclass: _buffer = property(_get_invmat_buffer)
    %}
};

template <typename ValueT = double>
class DetMove
{
public:
    DetMove(Determinant<ValueT> &target, unsigned max_rank);

    DetMove(const DetMove &other);

    ~DetMove();

    %extend {
        void _get_rbar(ValueT **ARGOUTVIEW_FARRAY2, int *DIM1, int *DIM2) {
            *ARGOUTVIEW_FARRAY2 = const_cast<ValueT *>(self->rbar());
            *DIM1 = self->rank();
            *DIM2 = self->target().order();
        }
        void _get_cbar(ValueT **ARGOUTVIEW_FARRAY2, int *DIM1, int *DIM2) {
            *ARGOUTVIEW_FARRAY2 = const_cast<ValueT *>(self->cbar());
            *DIM1 = self->target().order();
            *DIM2 = self->rank();
        }
        void _get_sbar(ValueT **ARGOUTVIEW_FARRAY2, int *DIM1, int *DIM2) {
            *ARGOUTVIEW_FARRAY2 = const_cast<ValueT *>(self->sbar());
            *DIM1 = *DIM2 = self->rank();
        }
    }

    %pythoncode %{
        __swig_getmethods__["_rbar"] = _get_rbar
        if _newclass: _rbar = property(_get_rbar)

        __swig_getmethods__["_cbar"] = _get_cbar
        if _newclass: _cbar = property(_get_cbar)

        __swig_getmethods__["_sbar"] = _get_sbar
        if _newclass: _sbar = property(_get_sbar)
    %}
};

template <typename ValueT = double>
class DetAppendMove
        : public DetMove<ValueT>
{
public:
    typedef ValueT Value;

public:
    DetAppendMove(Determinant<ValueT> &target, unsigned max_rank);

    // As detailled in the numpy.i FAQ
    %exception propose {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (ValueT *IN_FARRAY2, int DIM1, int DIM2) {
        (ValueT *row, int nrow, int mrow),
        (ValueT *col, int ncol, int mcol),
        (ValueT *star, int nstar, int mstar)
    }
    %extend {
        void propose(ValueT *row, int nrow, int mrow, ValueT *col, int ncol,
                     int mcol, ValueT *star, int nstar, int mstar)
        {
            const unsigned rank = nrow, order = self->target().order();

            if (unsigned(rank) > self->max_rank()) {
                PyErr_Format(PyExc_ValueError,
                             "Rank %d exceeds maximum rank %d for move",
                             rank, self->max_rank());
            } else if (unsigned(mrow) != order) {
                PyErr_Format(PyExc_ValueError,
                             "Invalid row shape (%d, %d), expecting (%d, %d)",
                             nrow, mrow, rank, order);
            } else if (unsigned(ncol) != order || unsigned(mcol) != rank) {
                PyErr_Format(PyExc_ValueError,
                             "Invalid col shape (%d, %d), expecting (%d, %d)",
                             ncol, mcol, order, rank);
            } else if (unsigned(nstar) != rank || unsigned(mstar) != rank) {
                PyErr_Format(PyExc_ValueError,
                             "Invalid star shape (%d, %d), expecting (%d, %d)",
                             nstar, mstar, rank, rank);
            } else {
                self->propose(rank, row, col, star);
            }
        }
    }

    void accept();
};

template <typename ValueT = double>
class DetRemoveMove
        : public DetMove<ValueT>
{
public:
    typedef ValueT Value;

public:
    DetRemoveMove(Determinant<ValueT> &target, unsigned max_rank);

    %exception propose {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (unsigned *IN_ARRAY1, int DIM1) {
        (unsigned *rowno, int nrowno),
        (unsigned *colno, int ncolno)
    }
    %extend {
        void unsafe_propose(unsigned *rowno, int nrowno, unsigned *colno,
                            int ncolno)
        {
            if (nrowno != ncolno) {
                PyErr_Format(PyExc_ValueError, "Matrix must remain square");
                return;
            }
            try {
                self->verify_args(nrowno, rowno, colno);
            } catch(const VerificationError &e) {
                PyErr_Format(PyExc_ValueError, "%s", e.what());
                return;
            }
            self->propose(nrowno, rowno, colno);
        }
    }

    %extend {
        void _get_rowno(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->rowno());
            *DIM1 = self->rank();
        }
        void _get_colno(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->colno());
            *DIM1 = self->rank();
        }
        void _get_rowrepl(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->rowrepl());
            *DIM1 = self->rank();
        }
        void _get_colrepl(unsigned **ARGOUTVIEW_ARRAY1, int *DIM1) {
            *ARGOUTVIEW_ARRAY1 = const_cast<unsigned *>(self->colrepl());
            *DIM1 = self->rank();
        }
    }

    void accept();

    %pythoncode %{
        # C++ code steals a reference to this ... so we need to make sure
        # Python does not destroy the arrays by holding a reference in the
        # Python part of the class.
        def propose(self, rownos, colnos):
            self._ref_rownos = rownos
            self._ref_colnos = colnos
            self.unsafe_propose(rownos, colnos)

        __swig_getmethods__["rowno"] = _get_rowno
        if _newclass: rowno = property(_get_rowno)

        __swig_getmethods__["colno"] = _get_colno
        if _newclass: colno = property(_get_colno)

        __swig_getmethods__["rowrepl"] = _get_rowrepl
        if _newclass: rowrepl = property(_get_rowrepl)

        __swig_getmethods__["colrepl"] = _get_colrepl
        if _newclass: colrepl = property(_get_colrepl)
    %}
};


template <typename ValueT = double>
class DetSetMove
{
public:
    typedef ValueT Value;

public:
    DetSetMove(Determinant<ValueT> &target, unsigned max_rank);

    %exception propose {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %extend {
        void unsafe_propose(ValueT *IN_FARRAY2, int DIM1, int DIM2)
        {
            if (DIM1 != DIM2) {
                PyErr_Format(PyExc_ValueError, "Matrix must remain square");
                return;
            }
            self->propose(DIM1, IN_FARRAY2);
        }
    }

    void accept();

    %pythoncode %{
        # C++ code steals a reference to this ... so we need to make sure
        # Python does not destroy the arrays by holding a reference in the
        # Python part of the class.
        def propose(self, new_matrix):
            self._ref_new_matrix = new_matrix
            self.unsafe_propose(new_matrix)
    %}
};

// Define all instantiations

%define DET_INSTANTIATE(suffix, ValueT...)

%attribute(Determinant<ValueT>, unsigned, order, order)
%attribute(Determinant<ValueT>, unsigned, capacity, capacity, reserve)
%attribute(Determinant<ValueT>, ValueT, weight, weight)
%template(Determinant ## suffix) Determinant<ValueT>;

%attribute(DetMove<ValueT>, unsigned, rank, rank)
%attribute(DetMove<ValueT>, unsigned, max_rank, max_rank)
%attribute(DetMove<ValueT>, unsigned, capacity, capacity)
%attribute2(DetMove<ValueT>, const Determinant<ValueT>, target, target)
%attribute(DetMove<ValueT>, ValueT, ratio, ratio)
%attribute(DetMove<ValueT>, bool, hard_reject, hard_reject)
%template(DetMove ## suffix) DetMove<ValueT>;

%template(DetAppendMove ## suffix) DetAppendMove<ValueT>;

%attribute(DetRemoveMove<ValueT>, int, perm_sign, perm_sign)
%template(DetRemoveMove ## suffix) DetRemoveMove<ValueT>;

%attribute2(DetSetMove<ValueT>, const Determinant<ValueT>, target, target)
%attribute(DetSetMove<ValueT>, ValueT, ratio, ratio)
%attribute(DetSetMove<ValueT>, bool, hard_reject, hard_reject)
%template(DetSetMove ## suffix) DetSetMove<ValueT>;

%catches(VerificationError) Determinant<ValueT>::verify();

#if MODE_DEBUG
    %catches(VerificationError) DetAppendMove<ValueT>::accept();
    %catches(VerificationError) DetRemoveMove<ValueT>::accept();
    %catches(VerificationError) DetSetMove<ValueT>::accept();
#endif

%enddef

DET_INSTANTIATE(D, double)
DET_INSTANTIATE(Z, std::complex<double>)
