/** 
 * Copyright (c) 2012-2015 Advanced Numerics Advisors Michal Rewienski and
 * Mayukh Bhattacharya. All rights reserved.
 * CONFIDENTIAL - This is an unpublished proprietary work of Advanced
 * Numerics Advisors Michal Rewienski and Mayukh Bhattacharya, and is
 * fully protected under copyright and trade secret laws. You may not
 * view, use, disclose, copy, or distribute this file or any information
 * contained herein except pursuant to a valid written license from
 * Advanced Numerics Advisors Michal Rewienski and Mayukh Bhattacharya.
 */

/** This contains functions related to random-number generation */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#include <sys/timeb.h>
#else /* WIN32 */
#include <sys/time.h>
#endif /* WIN32 */
//#include "rwengine.h"



#ifndef engineopt
#define engineopt 0
#endif

/* From mtwist.c */

#define MT_INLINE			/* Disable the inline keyword */
#define MT_EXTERN			/* Generate real code for functions */

#include "randgen.h"

/** This points to the selected random number generator function */
randfun_t * myrand = NULL;

/** This stores the vale of RAND_MAX for the currently used RNG (myrand points to it) */
UINT64_t MY_RAND_MAX;

/** This specifies how many times a RNG will be called during its warmup. */
#define WARMUP_CYCLE_LENGTH 100000

/** This returns a pseudorandom integer number from range [0,range]
    without a bias */
UINT32_t rand_range(UINT32_t range)
{
  assert(MY_RAND_MAX <= maxUINT32);
  UINT32_t divisor = MY_RAND_MAX/(range+1);
  UINT32_t outval;
  do {
    outval = myrand() / divisor;
  } while (outval > range);
  return outval;
}

/** Seeds for the JKISS RNGs */
static UINT32_t x_seed, y_seed, z_seed, c_seed, w_seed;

/** JKISS pseudorandom number generator, returns a pseudorandom unsigned
    integer number */
UINT32_t JKISS(void)
{
  UINT64_t t_seed;

  x_seed = 314527869 * x_seed + 1234567;
  y_seed ^= y_seed << 5; y_seed ^= y_seed >> 7; y_seed ^= y_seed << 22;
  t_seed = UINT64_c(4294584393) * z_seed + c_seed; c_seed = t_seed >> 32;
  z_seed = t_seed;

  return x_seed + y_seed + z_seed;
}

/** JKISS pseudorandom number generator, returns a pseudorandom unsigned
    integer number */
UINT32_t JKISS32(void)
{
  INT32_t t_seed;

  y_seed ^= (y_seed<<5); y_seed ^= (y_seed>>7); y_seed^= (y_seed<<22);
  t_seed = z_seed+w_seed+c_seed; z_seed = w_seed; c_seed = t_seed < 0;
  w_seed=t_seed&2147483647; x_seed += 1411392427;

  return x_seed + y_seed + w_seed;
}

static UINT32_t		mts_devseed(mt_state* state, char* seed_dev);
					/* Choose seed from a device */


#define RECURRENCE_OFFSET 397		/* Offset into state space for the */
					/* ..recurrence relation.  The */
					/* ..recurrence mashes together two */
					/* ..values that are separated by */
					/* ..this offset in the state */
					/* ..space. */
#define MATRIX_A	0x9908b0df	/* Constant vector A for the */
					/* ..recurrence relation.  The */
					/* ..mashed-together value is */
					/* ..multiplied by this vector to */
					/* ..get a new value that will be */
					/* ..stored into the state space. */

/*
 * Width of an unsigned int.  Don't change this even if your ints are 64 bits.
 */
#define BIT_WIDTH	32		/* Work with 32-bit words */


#define UPPER_MASK	0x80000000	/* Most significant w-r bits */
#define LOWER_MASK	0x7fffffff	/* Least significant r bits */

/*
 * Macro to simplify code in the generation loop.  This function
 * combines the top bit of x with the bottom 31 bits of y.
 */
#define COMBINE_BITS(x, y) \
			(((x) & UPPER_MASK) | ((y) & LOWER_MASK))


/*
 * Another generation-simplification macro.  This one does the magic
 * scrambling function.
 */
#define MATRIX_MULTIPLY(original, new) \
			((original) ^ ((new) >> 1) \
			  ^ matrix_decider[(new) & 0x1])


/*
 * Parameters of Knuth's PRNG (Line 25, Table 1, p. 102 of "The Art of
 * Computer Programming, Vol. 2, 2nd ed, 1981).
 */
#define KNUTH_MULTIPLIER_OLD \
			69069

/*
 * Parameters of Knuth's PRNG (p. 106 of "The Art of Computer
 * Programming, Vol. 2, 3rd ed).
 */
#define KNUTH_MULTIPLIER_NEW \
			1812433253ul
#define KNUTH_SHIFT	30		// Even on a 64-bit machine!


#define DEFAULT_SEED32_OLD \
			4357


#define DEVRANDOM	"/dev/random"
#define DEVURANDOM	"/dev/urandom"




/*
 * Many applications need only a single PRNG, so it's a nuisance to have to
 * specify a state.  For those applications, we will provide a default
 * state, and functions to use it.
 */
mt_state		mt_default_state;


/*
 * In the recurrence relation, the new value is XORed with MATRIX_A only if
 * the lower bit is nonzero.  Since most modern machines don't like to
 * branch, it's vastly faster to handle this decision by indexing into an
 * array.  The chosen bit is used as an index into the following vector,
 * which produces either zero or MATRIX_A and thus the desired effect.
 */

double			mt_32_to_double;
					/* Multiplier to convert long to dbl */
double			mt_64_to_double;
					/* Mult'r to cvt long long to dbl */


static UINT32_t	matrix_decider[2] =
			  {0x0, MATRIX_A};

/*
 * Mark a PRNG's state as having been initialized.  This is the only
 * way to set that field nonzero; that way we can be sure that the
 * constants are set properly before the PRNG is used.
 *
 * As a side effect, set up some constants that the PRNG assumes are
 * valid.  These are calculated at initialization time rather than
 * being written as decimal constants because I frankly don't trust
 * the compiler's ASCII conversion routines.
 */
void mts_mark_initialized(
    mt_state*		state)		/* State vector to mark initialized */
    {
    int			i;		/* Power of 2 being calculated */

    /*
     * Figure out the proper multiplier for long-to-double conversion.  We
     * don't worry too much about efficiency, since the assumption is that
     * initialization is vastly rarer than generation of random numbers.
     */
    mt_32_to_double = 1.0;
    for (i = 0;  i < BIT_WIDTH;  i++)
	mt_32_to_double /= 2.0;
    mt_64_to_double = mt_32_to_double;
    for (i = 0;  i < BIT_WIDTH;  i++)
	mt_64_to_double /= 2.0;

    state->initialized = 1;
    }


/*
 * Initialize a Mersenne Twist PRNG from a 32-bit seed, using
 * Matsumoto and Nishimura's newer reference implementation (Jan. 9,
 * 2002).
 */
void mts_seed32new(
    mt_state*		state,		/* State vector to initialize */
    UINT32_t		seed)		/* 32-bit seed to start from */
    {
    int			i;		/* Loop index */
    UINT32_t		nextval;	/* Next value being calculated */

    /*
     * Fill the state vector using Knuth's PRNG.  Be sure to mask down
     * to 32 bits in case we're running on a machine with 64-bit
     * ints.
     */
    state->statevec[MT_STATE_SIZE - 1] = seed & 0xffffffffUL;
    for (i = MT_STATE_SIZE - 2;  i >= 0;  i--)
	{
	nextval = state->statevec[i + 1] >> KNUTH_SHIFT;
	nextval ^= state->statevec[i + 1];
	nextval *= KNUTH_MULTIPLIER_NEW;
	nextval += (MT_STATE_SIZE - 1) - i;
	state->statevec[i] = nextval & 0xffffffffUL;
	}

    state->stateptr = MT_STATE_SIZE;
    mts_mark_initialized(state);

    /*
     * Matsumoto and Nishimura's implementation refreshes the PRNG
     * immediately after running the Knuth algorithm.  This is
     * probably a good thing, since Knuth's PRNG doesn't generate very
     * good numbers.
     */
    mts_refresh(state);
    }



/*
 * Initialize a Mersenne Twist PRNG from a 32-bit seed.
 *
 * According to Matsumoto and Nishimura's paper, the seed array needs to be
 * filled with nonzero values.  (My own interpretation is that there needs
 * to be at least one nonzero value).  They suggest using Knuth's PRNG from
 * Line 25, Table 1, p.102, "The Art of Computer Programming," Vol. 2 (2nd
 * ed.), 1981.  I find that rather odd, since that particular PRNG is
 * sensitive to having an initial seed of zero (there are many other PRNGs
 * out there that have an additive component, so that a seed of zero does
 * not generate a repeating-zero sequence).  However, one thing I learned
 * from reading Knuth is that you shouldn't second-guess mathematicians
 * about PRNGs.  Also, by following M & N's approach, we will be compatible
 * with other implementations.  So I'm going to stick with their version,
 * with the single addition that a zero seed will be changed to their
 * default seed.
 */
void mts_seed32(
    mt_state*		state,		/* State vector to initialize */
    UINT32_t		seed)		/* 32-bit seed to start from */
    {
    int			i;		/* Loop index */

    if (seed == 0)
	seed = DEFAULT_SEED32_OLD;

    /*
     * Fill the state vector using Knuth's PRNG.  Be sure to mask down
     * to 32 bits in case we're running on a machine with 64-bit
     * ints.
     */
    state->statevec[MT_STATE_SIZE - 1] = seed & 0xffffffff;
    for (i = MT_STATE_SIZE - 2;  i >= 0;  i--)
        state->statevec[i] =
          (KNUTH_MULTIPLIER_OLD * state->statevec[i + 1]) & 0xffffffff;

    state->stateptr = MT_STATE_SIZE;
    mts_mark_initialized(state);

    /*
     * Matsumoto and Nishimura's implementation refreshes the PRNG
     * immediately after running the Knuth algorithm.  This is
     * probably a good thing, since Knuth's PRNG doesn't generate very
     * good numbers.
     */
    mts_refresh(state);
    }


/*
 * Choose a seed based on some moderately random input.  Prefers
 * /dev/urandom as a source of random numbers, but uses the lower bits
 * of the current time if /dev/urandom is not available.  In any case,
 * only provides 32 bits of entropy.
 */
UINT32_t mts_seed(
    mt_state*		state)		/* State vector to seed */
    {
    return mts_devseed(state, DEVURANDOM);
    }

/*
 * Generate 624 more random values.  This function is called when the
 * state vector has been exhausted.  It generates another batch of
 * pseudo-random values.  The performance of this function is critical
 * to the performance of the Mersenne Twist PRNG, so it has been
 * highly optimized.
 */
void mts_refresh(register mt_state*	state)		/* State for the PRNG */
    {
    register int	i;		/* Index into the state */
    register UINT32_t*
			state_ptr;	/* Next place to get from state */
    register UINT32_t
			value1;		/* Scratch val picked up from state */
    register UINT32_t
			value2;		/* Scratch val picked up from state */

    /*
     * Start by making sure a random seed has been set.  If not, set
     * one.
     */
    if (!state->initialized)
	{
	mts_seed32(state, DEFAULT_SEED32_OLD);
	return;				/* Seed32 calls us recursively */
	}

    /*
     * Now generate the new pseudorandom values by applying the
     * recurrence relation.  We use two loops and a final
     * 2-statement sequence so that we can handle the wraparound
     * explicitly, rather than having to use the relatively slow
     * modulus operator.
     *
     * In essence, the recurrence relation concatenates bits
     * chosen from the current random value (last time around)
     * with the immediately preceding one.  Then it
     * matrix-multiplies the concatenated bits with a value
     * RECURRENCE_OFFSET away and a constant matrix.  The matrix
     * multiplication reduces to a shift and two XORs.
     *
     * Some comments on the optimizations are in order:
     *
     * Strictly speaking, none of the optimizations should be
     * necessary.  All could conceivably be done by a really good
     * compiler.  However, the compilers available to me aren't quite
     * smart enough, so hand optimization needs to be done.
     *
     * Shawn Cokus was the first to achieve a major speedup.  In the
     * original code, the first value given to COMBINE_BITS (in my
     * characterization) was re-fetched from the state array, rather
     * than being carried in a scratch variable.  Cokus noticed that
     * the first argument to COMBINE_BITS could be saved in a register
     * in the previous loop iteration, getting rid of the need for an
     * expensive memory reference.
     *
     * Cokus also switched to using pointers to access the state
     * array and broke the original loop into two so that he could
     * avoid using the expensive modulus operator.  Cokus used three
     * pointers; Richard J. Wagner noticed that the offsets between
     * the three were constant, so that they could be collapsed into a
     * single pointer and constant-offset accesses.  This is clearly
     * faster on x86 architectures, and is the same cost on RISC
     * machines.  A secondary benefit is that Cokus' version was
     * register-starved on the x86, while Wagner's version was not.
     *
     * I made several smaller improvements to these observations.
     * First, I reversed the contents of the state vector.  In the
     * current version of the code, this change doesn't directly
     * affect the performance of the refresh loop, but it has the nice
     * side benefit that an all-zero state structure represents an
     * uninitialized generator.  It also slightly speeds up the
     * random-number routines, since they can compare the state
     * pointer against zero instead of against a constant (this makes
     * the biggest difference on RISC machines).
     *
     * Second, I returned to Matsumoto and Nishimura's original
     * technique of using a lookup table to decide whether to xor the
     * constant vector A (MATRIX_A in this code) with the newly
     * computed value.  Cokus and Wagner had used the ?: operator,
     * which requires a test and branch.  Modern machines don't like
     * branches, so the table lookup is faster.
     *
     * Third, in the Cokus and Wagner versions the loop ends with a
     * statement similar to "value1 = value2", which is necessary to
     * carry the fetched value into the next loop iteration.  I
     * recognized that if the loop were unrolled so that it generates
     * two values per iteration, a bit of variable renaming would get
     * rid of that assignment.  A nice side effect is that the
     * overhead of loop control becomes only half as large.
     *
     * It is possible to improve the code's performance somewhat
     * further.  In particular, since the second loop's loop count
     * factors into 2*2*3*3*11, it could be unrolled yet further.
     * That's easy to do, too: just change the "/ 2" into a division
     * by whatever factor you choose, and then use cut-and-paste to
     * duplicate the code in the body.  To remove a few more cycles,
     * fix the code to decrement state_ptr by the unrolling factor, and
     * adjust the various offsets appropriately.  However, the payoff
     * will be small.  At the moment, the x86 version of the loop is
     * 25 instructions, of which 3 are involved in loop control
     * (including the decrementing of state_ptr).  Further unrolling by
     * a factor of 2 would thus produce only about a 6% speedup.
     *
     * The logical extension of the unrolling
     * approach would be to remove the loops and create 624
     * appropriate copies of the body.  However, I think that doing
     * the latter is a bit excessive!
     *
     * I suspect that a superior optimization would be to simplify the
     * mathematical operations involved in the recurrence relation.
     * However, I have no idea whether such a simplification is
     * feasible.
     */
    state_ptr = &state->statevec[MT_STATE_SIZE - 1];
    value1 = *state_ptr;
    for (i = (MT_STATE_SIZE - RECURRENCE_OFFSET) / 2;  --i >= 0;  )
	{
	state_ptr -= 2;
	value2 = state_ptr[1];
	value1 = COMBINE_BITS(value1, value2);
	state_ptr[2] =
	  MATRIX_MULTIPLY(state_ptr[-RECURRENCE_OFFSET + 2], value1);
	value1 = state_ptr[0];
	value2 = COMBINE_BITS(value2, value1);
	state_ptr[1] =
	  MATRIX_MULTIPLY(state_ptr[-RECURRENCE_OFFSET + 1], value2);
	}
    value2 = *--state_ptr;
    value1 = COMBINE_BITS(value1, value2);
    state_ptr[1] =
      MATRIX_MULTIPLY(state_ptr[-RECURRENCE_OFFSET + 1], value1);

    for (i = (RECURRENCE_OFFSET - 1) / 2;  --i >= 0;  )
	{
	state_ptr -= 2;
	value1 = state_ptr[1];
	value2 = COMBINE_BITS(value2, value1);
	state_ptr[2] =
	  MATRIX_MULTIPLY(state_ptr[MT_STATE_SIZE - RECURRENCE_OFFSET + 2],
	    value2);
	value2 = state_ptr[0];
	value1 = COMBINE_BITS(value1, value2);
	state_ptr[1] =
	  MATRIX_MULTIPLY(state_ptr[MT_STATE_SIZE - RECURRENCE_OFFSET + 1],
	    value1);
	}

    /*
     * The final entry in the table requires the "previous" value
     * to be gotten from the other end of the state vector, so it
     * must be handled specially.
     */
    value1 = COMBINE_BITS(value2, state->statevec[MT_STATE_SIZE - 1]);
    *state_ptr =
      MATRIX_MULTIPLY(state_ptr[MT_STATE_SIZE - RECURRENCE_OFFSET], value1);

    /*
     * Now that refresh is complete, reset the state pointer to allow more
     * pseudorandom values to be fetched from the state array.
     */
    state->stateptr = MT_STATE_SIZE;
    }

/*
 * Choose a seed based on a random-number device given by the caller.
 * If that device can't be opened, use the lower 32 bits from the
 * current time.
 */
static UINT32_t mts_devseed(
    mt_state*		state,		/* State vector to seed */
    char*		seed_dev)	/* Device to seed from */
    {
    int			bytesread;	/* Byte count read from device */
    int			nextbyte;	/* Index of next byte to read */
    FILE*		ranfile;	/* Access to device */
    union
	{
	char		ranbuffer[sizeof (UINT32_t)];
					/* Space for reading random int */
	UINT32_t	randomvalue;	/* Random value for initialization */
	}
			randomunion;	/* Union for reading random int */
#ifdef WIN32
    struct _timeb	tb;		/* Time of day (Windows mode) */
#else /* WIN32 */
    struct timeval	tv;		/* Time of day */
    struct timezone	tz;		/* Dummy for gettimeofday */
#endif /* WIN32 */

    ranfile = fopen(seed_dev, "rb");
    /*
     * Some machines have /dev/random but not /dev/urandom.  On those
     * machines, /dev/random is nonblocking, so we'll try it before we
     * fall back to using the time.
     */
    if (ranfile == NULL)
	ranfile = fopen(DEVRANDOM, "rb");
    if (ranfile != NULL)
	{
	setbuf(ranfile, NULL);
	for (nextbyte = 0;
	  nextbyte < (int)sizeof randomunion.ranbuffer;
	  nextbyte += bytesread)
	    {
	    bytesread = fread(&randomunion.ranbuffer[nextbyte], (size_t)1,
	      sizeof randomunion.ranbuffer - nextbyte, ranfile);
	    if (bytesread == 0)
		break;
	    }
	fclose(ranfile);
	if (nextbyte == sizeof randomunion.ranbuffer)
	    {
	    mts_seed32new(state, randomunion.randomvalue);
	    return randomunion.randomvalue;
	    }
	}

    /*
     * The device isn't available.  Use the time.  We will
     * assume that the time of day is accurate to microsecond
     * resolution, which is true on most modern machines.
     */
#ifdef WIN32
    (void) _ftime (&tb);
#else /* WIN32 */
    (void) gettimeofday (&tv, &tz);
#endif /* WIN32 */

    /*
     * We just let the excess part of the seconds field overflow
     */
#ifdef WIN32
    randomunion.randomvalue = tb.time * 1000 + tb.millitm;
#else /* WIN32 */
    randomunion.randomvalue = tv.tv_sec * 1000000 + tv.tv_usec;
#endif /* WIN32 */
    mts_seed32new(state, randomunion.randomvalue);
    return randomunion.randomvalue;
    }


/*
 * Initialize the default Mersenne Twist PRNG from a 32-bit seed.
 *
 * See mts_seed32new for full commentary.
 */
void mt_seed32new(UINT32_t seed)		/* 32-bit seed to start from */
    {
    mts_seed32new(&mt_default_state, seed);
    }


/*
 * Initialize the PRNG from random input.  See mts_seed.
 */
UINT32_t mt_seed(void)
    {
    return mts_seed(&mt_default_state);
    }

/** This seeds the JKISS/JKISS32 generators. For JKISS set w = 0, for JKISS32 set c = 0 */
void JKISS_seed(UINT32_t x, UINT32_t y, UINT32_t z, UINT32_t c, UINT32_t w)
{
  x_seed = x;
  y_seed = y;
  z_seed = z;
  c_seed = c;
  w_seed = w;
}

static UINT32_t read_devrand(void)
{
  UINT32_t outrand;
  FILE * randf;
  randf = fopen(DEVURANDOM, "rb");
  if (randf == NULL)
    exit(-1);
  if(fread(&outrand, (size_t) 1, 4, randf) != 4)
    exit(-1);
  fclose(randf);
  return outrand;
}

/** This seeds the JKISS/JKISS32 generators. For JKISS set w = 0, for JKISS32 set c = 0 */
static void JKISS_rand_seed(void)
{
  x_seed = read_devrand();
  while(!(y_seed = read_devrand()));
  z_seed = read_devrand();
  c_seed = read_devrand() % 698769068 + 1;
  w_seed = read_devrand();
}

/** This seeds and warms up a random number generator. */
void rng_init(UINT32_t rngtype)
{
  /* Seed the generators */
  /* Note: Because we use the function pointer myrand,
     Mersenne twister is not inlined. 2X speedup can
     be achieved if this is fixed (by not using the pointer) */

  switch(rngtype) {
  case 0: /* rand */
    if (engineopt)
      srand(time(NULL));
    else
      srand(536870911);
      /* srand(2013); */ /* maybe a bigger number would be better? */
    myrand = (randfun_t *) rand;
    MY_RAND_MAX = UINT64_c(2147483647);
    break;
  case 1: /* JKISS */
    if (engineopt)
      JKISS_rand_seed();
    else
      JKISS_seed(536870911,134217727,268435455,67108863, 0);
    /* JKISS_seed(123456789,987654321,4321987,6543217,0); */
    myrand = JKISS;
    MY_RAND_MAX = UINT64_c(4294967295);
    break;
  case 2: /* JKISS32 */
    if (engineopt)
      JKISS_rand_seed();
    else
      JKISS_seed(536870911,134217727,268435455,0,67108863);
    /* JKISS_seed(123456789,234567891,345678912,0,456789123); */
    myrand = JKISS32;
    MY_RAND_MAX = UINT64_c(4294967295);
    break;
  case 3: /* Mersenne twister */
    if (engineopt)
      mt_seed();
    else
      mt_seed32new(536870911); /* maybe a bigger number would be better ? */
    /* mt_seed32new(2013); */ /* maybe a bigger number would be better ? */
    myrand = mt_lrand;
    MY_RAND_MAX = UINT64_c(4294967295);
    break;
  default:
    assert(0);
  }

  /* Warm up the generator */
  UINTpt_t i;
  for (i = 0; i < WARMUP_CYCLE_LENGTH; i++)
    myrand();
}
