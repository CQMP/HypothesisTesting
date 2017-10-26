%{
#include "../src/nfft.hh"
%}

%attribute(NDFTPlan, bool, use_nfft, use_nfft)
%attribute(NDFTPlan, unsigned, freq_dim, freq_dim)
%attribute(NDFTPlan, unsigned, nfreq, nfreq)
%attribute(NDFTPlan, unsigned, ntotfreq, ntotfreq)
%attribute(NDFTPlan, unsigned, nsample_last, nsample_last)

%numpy_typemaps(std::complex<float>, NPY_CFLOAT, int)
%numpy_typemaps(std::complex<double>, NPY_CDOUBLE, int)

class NDFTPlan
{
public:
    typedef std::complex<double> ValueT;

    static const bool NFFT_AVAILABLE = HAVE_NFFT;

public:
    NDFTPlan(unsigned freq_dim, unsigned nfreq, bool use_nfft);

    ~NDFTPlan();

    void reserve(unsigned nsample_max);

    %exception adjoint {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> **ARGOUTVIEWM_ARRAY1, int *DIM1) {
            (std::complex<double> **fhat, int *nfhat)
    }
    %apply (std::complex<double> *IN_ARRAY1, int DIM1) {
            (std::complex<double> *f, int nf)
    }
    %apply (double *IN_ARRAY2, int DIM1, int DIM2) {
            (double *x, int nx, int ndim)
    }
    %extend {
        void adjoint(std::complex<double> **fhat, int *nfhat,
                     std::complex<double> *f, int nf, double *x, int nx, int ndim)
        {
            if ((unsigned)ndim != self->freq_dim()) {
                PyErr_Format(PyExc_ValueError, "x has wrong 2nd dimension");
            } else if (nf != nx) {
                PyErr_Format(PyExc_ValueError, "f and x do not match");
            } else {
                *nfhat = self->ntotfreq();
                *fhat = (std::complex<double>*)malloc(*nfhat * sizeof(**fhat));
                self->adjoint(nf, *fhat, f, x);
            }
        }
    }
};
