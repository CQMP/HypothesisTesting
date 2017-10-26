%module cthyb

%{
#define SWIG_FILE_WITH_INIT
%}

%include "attribute.i"
%include "numpy.inc.i"

%init
%{
    // Ensures that numpy is set up properly
    import_array();
%}

%include "util.inc.i"
%include "base.inc.i"
%include "det.inc.i"
%include "bath.inc.i"
%include "seg.inc.i"
%include "dia.inc.i"
%include "driver.inc.i"
%include "nfft.inc.i"
%include "dndft.inc.i"
