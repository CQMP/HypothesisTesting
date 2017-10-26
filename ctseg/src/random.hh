
#include <cassert>
#include <cstdlib>

// Forward declarations

class Cxx11Random;
class CStdlibRandom;

struct Cxx11RandomNotAvailable : std::exception {};

// Actual declarations

#if __cplusplus >= 201103L

#include <random>
#define HAVE_CXX11RANDOM 1

/**
 * Wrapper around C++11 standard Mersenne twister random source.
 *
 * This class wraps around the C++11 standard random source std::mt19973, as
 * defined in the `random` header.  A wrapper is used as we (1) want to
 * substitute the source with any other random source in the case that the
 * compiler does not support C++11 and (2) provide a stripped-down interface
 * to random numbers in the interval [0,1), which can then be
 * easily implemented.
 */
class Cxx11Random
{
public:
    static const unsigned ENTROPY = std::mt19937::word_size;

public:
    Cxx11Random(unsigned seed=0) : engine_(seed), distrib_(0., 1.) {}

    void seed(unsigned seed) { engine_.seed(seed); }

    double operator() () { return distrib_(engine_); }

    unsigned raw() { return engine_(); }

private:
    std::mt19937 engine_;
    std::uniform_real_distribution<double> distrib_;
};

#else

// We do not have C++11 available. Define Random source as series of stubs.
#define HAVE_CXX11RANDOM 0

class Cxx11Random {
public:
    static const unsigned ENTROPY = 0;

public:
    Cxx11Random(unsigned) { throw Cxx11RandomNotAvailable(); }

    void seed(unsigned) { throw Cxx11RandomNotAvailable(); }

    double operator() () { throw Cxx11RandomNotAvailable(); }

    unsigned raw() { throw Cxx11RandomNotAvailable(); }
};

#endif /* __cplusplus >= 201103L */


static CStdlibRandom *_current = NULL;

/**
 * Wrapper around the C standard random number generator `stdlib.h`
 *
 * This class wraps around the C standard PRNG functions `rand()` and
 * `srand()`.  This PRNG is almost always of the shift-modulo type, which
 * produces random numbers of poor quality, low entropy and short period.
 * However, since it is the only PRNG for old C++ versions, we provide a
 * wrapper here.
 *
 * This class is a singleton (it may only be instantiated *once*), since the
 * underlying C library function store the current state in a global.
 */
class CStdlibRandom
{
public:
    // 15 bits of entropy are guaranteed by the C standard
    static const unsigned ENTROPY = RAND_MAX >= 2147483648L ? 31 : 15;

public:
    CStdlibRandom(unsigned seed=0)
    {
        assert(_current == NULL);
        _current = this;
        this->seed(seed);
    }

    ~CStdlibRandom()
    {
        if (_current == this)
            _current = NULL;
    }

    void seed(unsigned seed) { srand(seed); }

    double operator() ()
    {
        assert(_current == this);
        return rand()/(RAND_MAX + 1.0);
    }

    unsigned raw()
    {
        assert(_current == this);
        return rand();
    }
};

#if HAVE_CXX11RANDOM
    typedef Cxx11Random DefaultRandom;
#else
    #pragma message "WARNING: Falling back on poor Stdlib RNG. Check results."
    typedef CStdlibRandom DefaultRandom;
#endif
