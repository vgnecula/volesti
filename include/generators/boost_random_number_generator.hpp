#ifndef GENERATORS_BOOST_RANDOM_NUMBER_GENERATOR_HPP
#define GENERATORS_BOOST_RANDOM_NUMBER_GENERATOR_HPP

#include <chrono>
#include <boost/random.hpp>

template <typename RNGType, typename NT, int... Ts>
struct BoostRandomNumberGenerator;

template <typename RNGType, typename NT>
struct BoostRandomNumberGenerator<RNGType, NT>
{
    typedef NT result_type;

    BoostRandomNumberGenerator(int d)
        : _rng(std::chrono::system_clock::now().time_since_epoch().count())
        , _urdist(0, 1)
        , _uidist(0, d-1)
        , _ndist(0, 1)
    {}

    NT operator()() {
        return sample_urdist();
    }

    NT sample_urdist()
    {
        return _urdist(_rng);
    }

    NT sample_uidist()
    {
        return _uidist(_rng);
    }

    NT sample_ndist()
    {
        return _ndist(_rng);
    }

    void seed(unsigned rng_seed) {
        _rng.seed(rng_seed);
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return 1; }

private:
    RNGType _rng;
    boost::random::uniform_real_distribution<NT> _urdist;
    boost::random::uniform_int_distribution<> _uidist;
    boost::random::normal_distribution<NT> _ndist;
};

template <typename RNGType, typename NT, int Seed>
struct BoostRandomNumberGenerator<RNGType, NT, Seed>
{
    typedef NT result_type;

    BoostRandomNumberGenerator(int d=1)
        : _rng(Seed)
        , _urdist(0, 1)
        , _uidist(0, d-1)
        , _ndist(0, 1)
    {}

    NT operator()() {
        return sample_urdist();
    }

    NT sample_urdist()
    {
        return _urdist(_rng);
    }

    NT sample_uidist()
    {
        return _uidist(_rng);
    }

    NT sample_ndist()
    {
        return _ndist(_rng);
    }

    void seed(unsigned rng_seed) {
        _rng.seed(rng_seed);
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return 1; }

private:
    RNGType _rng;
    boost::random::uniform_real_distribution<NT> _urdist;
    boost::random::uniform_int_distribution<> _uidist;
    boost::random::normal_distribution<NT> _ndist;
};

#endif // GENERATORS_BOOST_RANDOM_NUMBER_GENERATOR_HPP