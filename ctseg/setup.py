#!/usr/bin/env python

# Python's deployment infrastructure is a hot, steaming mess. There is
# 'distutils', which is available through the standard library, then 
# setuptools, a superset, a now-defunct 'distutils2' and absolutely horrendous
# documentation on every count.  Please, *please*, just adopt the CMAKE
# approach and let us all live.
try:
    from setuptools.core import setup, Extension
    installer = 'setuptools'
except ImportError:
    from distutils.core import setup, Extension
    installer = 'distutils'

try:
    import numpy
except ImportError:
    print "No numpy present. You probably have to try again"
    numpy_include = []
else:
    numpy_include = [numpy.get_include()]

extra_compile_args = []


# Check that steaming mess with the Anaconda stack dependencies
try:
    import sysconfig
    import platform
except ImportError:
    pass
else:
    build_platform = sysconfig.get_platform()
    run_platform = platform.mac_ver()
    if build_platform.startswith('macosx-'):
        build_version = map(int, build_platform.split('-')[1].split('.'))
        run_version = map(int, run_platform[0].split('.'))
        if build_version < run_version:
            extra_compile_args.append('-mmacosx-version-min=%s'
                                      % run_platform[0])

cpplib = Extension(
    'cthyb',
    ['swig/cthyb.cc'], 
    include_dirs=[
        'include/',
        'bundled/include',
        ] + numpy_include,
    define_macros=[
        ('NDEBUG', '1'),
        ('MODE_DEBUG', '0'),
        ('HAVE_NFFT', '0'),
        ],
    extra_compile_args=[
        '--std=c++0x',
        ] + extra_compile_args,
    language='c++'
    )

ARGS = dict(
    name='cthyb',
    version='1.0',
    description='CT-HYB impurity solver',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        ],
    long_description='Not now, man',

    author='Markus Wallerberger',
    author_email='markus@wallerberger.at',
    url='https://wallerberger.at/cthyb',

    packages=[
        'ctseg_py',
        'manybody'
        ],
    package_dir={
        'ctseg_py': 'python/ctseg_py',
        'manybody': 'python/manybody',
        },
    ext_modules=[
        cpplib
        ],
    )

ARGS_SETUPTOOLS = dict(
    setup_requires=[
        'numpy',
        ],
    install_requires=[
        'numpy (>=1.4)',
        ],
    entry_points={
        'console_scripts': [
            'ctseg = ctseg_py.__main__:main',
            ]
        },
    )

ARGS_DISTUTILS = dict(
    scripts=[
        'python/ctseg',
        ],
    )

if installer == 'setuptools':
    ARGS.update(dict(ARGS_SETUPTOOLS))
else:
    ARGS.update(dict(ARGS_DISTUTILS))

setup(**ARGS)

