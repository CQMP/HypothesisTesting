Ising solver
============

Small Ising solver used to generate the results here.

Please contact me before re-using or re-distributing this code.

Installation
------------
Needs FFTW3 and ALPSCore.

To build:

    $ mkdir build
    $ cmake .. -DCMAKE_BUILD_TYPE=Release
    $ make

To run:

    $ ./build/ising --help
    
To install:

    $ cmake .. -DCMAKE_INSTALL_PREFIX=[...]
    $ make install
    $ ising --help
