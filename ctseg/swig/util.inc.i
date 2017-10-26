%{
#include "../src/util.hh"
%}

template <typename ValueT>
class PiecewisePolynomial
{
public:
    %extend {
        PiecewisePolynomial(ValueT *IN_ARRAY2, int DIM1, int DIM2)
        {
            // The SWIG convention is for the constructor to return a new
            // object rather than just initialising it.
            return new PiecewisePolynomial<ValueT>(IN_ARRAY2, DIM1, DIM2);
        }
    }

    ValueT value(double x) const;
};

%define UTIL_INSTANTIATE(suffix, ValueT...)

%attribute(PiecewisePolynomial<ValueT>, unsigned, n, n);
%attribute(PiecewisePolynomial<ValueT>, unsigned, order, order);
%template(PiecewisePolynomial ## suffix) PiecewisePolynomial<ValueT>;

%enddef

UTIL_INSTANTIATE(D, double)
