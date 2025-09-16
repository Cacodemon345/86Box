#include <stdint.h>
#include <86box/random.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

// SplitMix64 generator.
static uint64_t splitmix64_x = 0;

static uint64_t splitmix64_next(void) {
	uint64_t z = (splitmix64_x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

// xorshiro256** generator
static uint64_t s[4];

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static uint64_t next(void) {
	const uint64_t result = rotl(s[1] * 5, 7) * 9;

	const uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return result;
}

uint64_t
random_generate_64_new(void)
{
    return next();
}

// 64-bit to 8-bit wrapper;
static uint64_t cur_res = 0;
static uint8_t cur_cntr = 7;

uint8_t
random_generate_new(void)
{
    if (cur_cntr >= 7) {
        cur_res = random_generate_64_new();
        cur_cntr = 0;
    } else {
        cur_cntr++;
        cur_res >>= 8;
    }

    return cur_res & 0xFF;
}

void
random_init_new(void)
{
#ifdef _WIN32
    LARGE_INTEGER res;
    QueryPerformanceCounter(&res);
    splitmix64_x = (uint64_t)res.QuadPart;
#else
    struct timespec cur_time;
    clock_gettime(CLOCK_MONOTONIC, &cur_time);
    splitmix64_x = (uint64_t)curtime.tv_nsec + ((uint64_t)curtime.tv_sec * (uint64_t)1000000000ull);
#endif
    s[0] = splitmix64_next();
    s[1] = splitmix64_next();
    s[2] = splitmix64_next();
    s[3] = splitmix64_next();
    return;
}
