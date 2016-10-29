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

#ifndef PORTTYPE_H
#define PORTTYPE_H

/** This file contains definitions for portable data types */

#ifndef BITS32
#define BITS32
#endif

#include <math.h>
#include <assert.h>
#include <limits.h>

/** redefine assert() - do not perform checks if not in
    debug mode */
#ifdef NODEBUG
#ifdef assert
#undef assert
#endif /* undefine assert */
#define assert(_STAT_) ((void) 0)
#endif /* NODEBUG */

/** 32-bit floating point number */
typedef float REAL32_t;

/** 64-bit floating point number */
typedef double REAL64_t;

/** 8-bit unsigned integer */
typedef unsigned char UINT8_t;

/** 16-bit unsigned integer */
typedef unsigned short int UINT16_t;

/** 32-bit unsigned integer */
typedef unsigned int UINT32_t;

/** 8-bit signed integer */
typedef signed char INT8_t;

/** 16-bit signed integer */
typedef signed short int INT16_t;

/** 32-bit signed integer */
typedef signed int INT32_t;

/** *** 32-bit architecture *** */
#ifdef BITS32

/** for the 32-bit version */
#define _BITVER_ 32

/** 64-bit signed integer */
typedef signed long long int INT64_t;

/** Signed int of the same size as pointer */
typedef INT32_t INTpt_t;

/** 64-bit unsigned integer */
typedef unsigned long long int UINT64_t;

/* Unsigned int of the same size as pointer */
typedef UINT32_t UINTpt_t;

/* macros for specifying constants */

/** 32-bit unsigned int constant */
#define UINT32_c(_const_) ((UINT32_t) (_const_ ## u))

/** 64-bit unsigned int constant */
#define UINT64_c(_const_) ((UINT64_t) (_const_ ## ull))

/** pointer-size unsigned int constant */
#define UINTpt_c(_const_) UINT32_c(_const_)

#else /* BITS32 */

/** *** 64-bit architecture *** */

#ifndef BITS64
#error BITS32 or BITS64 must be specified
#endif /* BITS64 */

/** for the 64-bit version */
#define _BITVER_ 64

/** 64-bit signed integer */
typedef signed long int INT64_t;

/** Signed int of the same size as pointer */
typedef INT64_t INTpt_t;

/** 64-bit unsigned integer */
typedef unsigned long int UINT64_t;

/* Unsigned int of the same size as pointer */
typedef UINT64_t UINTpt_t;

/* macros for specifying constants */

/** 32-bit unsigned int constant */
#define UINT32_c(_const_) ((UINT32_t) (_const_ ## u))

/** 64-bit unsigned int constant */
#define UINT64_c(_const_) ((UINT64_t) (_const_ ## ul))

/** pointer-size unsigned int constant */
#define UINTpt_c(_const_) UINT64_c(_const_)

#endif /* BITS32 */

/* The largest value for 32-bit unsigned integer */
#define maxUINT32 UINT32_c(0xFFFFFFFF)

/* The largest value for 64-bit unsigned integer */
#define maxUINT64 UINT64_c(0xFFFFFFFFFFFFFFFF)

#ifdef BITS32

#define maxUINTpt maxUINT32

#else /* NITS32 */

#define maxUINTpt maxUINT64

#endif /* BITS32 */

/** infinity definition according to IEEE standard */

/** this type is to store IEEE infinity */
typedef union IEEE_INF_t
{
  UINT8_t infbyte[8]; /**< this will contain byte image of infinity */
  REAL64_t infreal; /**< double precision */
} IEEE_INF_t;

/** This variable is specified in porttype.c */
extern IEEE_INF_t ieee_inf;

/** This macro specifies IEEE infinity constant */
#define INF (ieee_inf.infreal)

/* Value of PI */
#define PI 3.141592653589793

#endif /* PORTTYPE_H */
