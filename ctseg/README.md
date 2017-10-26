CT-HYB
======

This is the CT-HYB(SEG) code used to produce the AIM result.

Please DO NOT re-use or re-distribute this solver without my permission.
(It is not really production-ready!)

Installation
------------
To compile:

    $ make

If something goes wrong, check out `config.mk` that was generated and modify accordingly.
On Mac OSX, please check out the `configs/macosx.mk`.

To run:

    $ python ./python/ctseg --help

To install:

    $ python setup.py install
    $ ctseg
