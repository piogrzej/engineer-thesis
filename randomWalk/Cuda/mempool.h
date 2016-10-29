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

#ifndef MEMPOOL_H
#define MEMPOOL_H

#define NULL 0

/** This contains memory allocation utilities */

#include <malloc.h>
#include <memory.h>
#include "porttype.h"

/** This is a handle for memory pool */
typedef struct memh_t
{
  struct mempool_t *pool; /**< points to mem pool data */
} memh_t;

/** Memory pool initialization */
void memInit(memh_t * mh);

/** Frees memory pool */
void memFree(memh_t * mh);

/** This frees and unused blocks or memory pool structures */
void memCleanup(void);

/** Allocates memory from memory pool. Not to be used directly, only through memGet */
void *memGetInternal(memh_t * mh, UINT64_t count);

/** Allocates memory from memory pool. */
#define memGet(outptr, memh, datatype, count)	\
do { \
  UINT64_t _cnt; \
  _cnt = ((UINT64_t) (count)) * sizeof(datatype);    \
  (outptr) = (datatype *) memGetInternal(memh, _cnt); \
 } while(0);

/** Simple wrapper around malloc for now */
#define rwMalloc(outptr, type, n)                  \
do {                                         \
  assert(n > 0);                            \
  UINT64_t _nn = sizeof(type) * ((UINT64_t) (n)); \
  assert(_nn == (UINTpt_t) _nn);              \
  outptr = (type *) malloc((UINTpt_t) _nn);    \
 } while(0)

/** Simple wrapper for free */
#define rwFree(_ptr_)  \
do {               \
  if (_ptr_)       \
    {              \
      free(_ptr_); \
      (_ptr_) = NULL;				\
    }              \
 } while (0)

/** Zeros an array of size _count_ */
#define setZero(_ptr_, _count_) memset ((char *) (_ptr_), 0, sizeof(*(_ptr_))*(_count_))

#endif /* MEMPOOL_H" */
