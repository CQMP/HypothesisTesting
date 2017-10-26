%{
#include "../src/dndft.hh"
%}

template <typename Window>
class DeferredAdjointNDFT
{
public:
    DeferredAdjointNDFT(const Window &window);

    %exception reset {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> *INPLACE_ARRAY1, int DIM1) {
            (std::complex<double> *f_conv, int nf_conv)
    }
    %extend {
        void reset(std::complex<double> *f_conv, int nf_conv)
        {
            if (nf_conv != (int)self->ngrid()) {
                PyErr_Format(PyExc_ValueError,
                             "Array size %d does not match grid size %d",
                             nf_conv, self->ngrid());
            } else {
                self->reset(f_conv);
            }
        }
    }

    %exception compute {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> *INPLACE_ARRAY1, int DIM1) {
            (std::complex<double> *f_conv, int nf_conv)
    }
    %extend {
        void add(std::complex<double> *f_conv, int nf_conv,
                 std::complex<double> f, double x)
        {
            if (nf_conv != (int)self->ngrid()) {
                PyErr_Format(PyExc_ValueError,
                             "Array size %d does not match grid size %d",
                             nf_conv, self->ngrid());
            } else {
                self->add(f_conv, f, x);
            }
        }
    }

    %exception compute {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    %apply (std::complex<double> *IN_ARRAY1, int DIM1) {
            (std::complex<double> *f_conv, int nf_conv)
    }
    %apply (std::complex<double> **ARGOUTVIEWM_ARRAY1, int *DIM1) {
            (std::complex<double> **fhat, int *nfhat)
    }
    %extend {
        void compute(std::complex<double> **fhat, int *nfhat,
                     std::complex<double> *f_conv, int nf_conv)
        {
            if (nf_conv != (int)self->ngrid()) {
                PyErr_Format(PyExc_ValueError,
                             "Array size %d does not match grid size %d",
                             nf_conv, self->ngrid());
            } else {
                *nfhat = self->nfreq();
                *fhat = (std::complex<double>*)malloc(*nfhat * sizeof(**fhat));
                self->compute(*fhat, f_conv);
            }
        }
    }
};

class GaussianWindow
{
public:
    GaussianWindow(unsigned m, unsigned N, unsigned sigma);

    %extend {
        double phi(double t) { return phi_full(*self, t); }

        std::complex<double> phi_hat(int k) { return phi_hat_full(*self, k); }
    }
};

%attribute(GaussianWindow, unsigned, m, m)
%attribute(GaussianWindow, unsigned, N, N)
%attribute(GaussianWindow, unsigned, n, n)

%define DNDFT_INSTANTIATE(suffix, Window...)

%attribute(DeferredAdjointNDFT<Window>, unsigned, nfreq, nfreq)
%attribute(DeferredAdjointNDFT<Window>, unsigned, ngrid, ngrid)
%attribute(DeferredAdjointNDFT<Window>, unsigned, freq_dim, freq_dim)
%attribute2(DeferredAdjointNDFT<Window>, Window, window, window)
%template(DeferredAdjointNDFT ## suffix) DeferredAdjointNDFT<Window>;

%enddef

DNDFT_INSTANTIATE(Gauss, GaussianWindow)

