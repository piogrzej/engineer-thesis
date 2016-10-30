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

/** This contains code for allocating memory using memory pools. */

#include "mempool.h"

/** This stores information about an allocated block */
typedef struct memblock_t
{
  struct memblock_t *next; /**< points to the next block */
  UINTpt_t size; /**< size of this block */
  UINT8_t *datablock; /**< points to data block in this block */
  UINTpt_t unused; /**< ensures 16-byte alignment */
} memblock_t;

/** This stores memory pool data */
typedef struct mempool_t
{
  struct mempool_t *nextpool; /**< points to a next pool */
  UINTpt_t available_mem_size; /**< memory available in the pool, at pointer below */
  UINT8_t *memptr; /**< pointer to available memory */
  memblock_t *blocks; /**< points to blocks allocated in this mem pool */
} mempool_t;

/** This specifies default block size to be allocated - currently 16K, minus
    malloc overhead (24b for linux) and memory storing block information, at the beginning
    of the block */
#define BLOCK_SIZE ((UINTpt_t) (16384-24-sizeof(memblock_t)))

/** This is for 8b alignment */
#define ALIGN_MASK ((UINT64_t) (7))

/** Default block size, aligned */
#define BLOCK_SIZE_ALIGNED ((UINTpt_t) (BLOCK_SIZE & ~ALIGN_MASK))

/** This is for reusing memory pools */

/** Max number of stored reused memory pool structures */
#define MAX_REUSED_POOLS 500

/** Previously used pools are stored here */
static mempool_t *_available_pools = NULL;

/** No. available memory pools */
static UINTpt_t _available_pool_cnt = 0;

/** This is for reusing memory blocks */

/** Max number of stored reused memory blocks  */
#define MAX_REUSED_BLOCKS 5

/** Previously used blocks are stored here */
static memblock_t *_available_blocks = NULL;

/** No. available memory blocks */
static UINTpt_t _available_block_cnt = 0;

/** Memory pool initialization */
void memInit(memh_t * mh)
{
  mempool_t *pool;
  if (_available_pools)
    {
      pool = _available_pools;
      _available_pools = pool->nextpool;
      _available_pool_cnt--;
    }
  else
    rwMalloc(pool, mempool_t, 1);
  setZero(pool, 1);
  mh->pool = pool;
}

void memFree(memh_t * mh)
{
  /* First, free memory blocks */
  mempool_t *pool = mh->pool;
  memblock_t *this_block, *next_block;
  for (this_block = pool->blocks; this_block; this_block=next_block)
    {
      next_block = this_block->next;
      /* save the block for reuse or free it */
      if ((this_block->size != BLOCK_SIZE_ALIGNED) ||
	  _available_block_cnt >= MAX_REUSED_BLOCKS)
	rwFree(this_block);
      else
	{
	  /* add to list */
	  this_block->next = _available_blocks;
	  _available_blocks = this_block;
	  _available_block_cnt++;
	}
    }

  /* Then, free (or save for reuse the pool structure itself */
  if (_available_pool_cnt >= MAX_REUSED_POOLS)
    rwFree(pool);
  else
    {
      pool->nextpool = _available_pools;
      _available_pools = pool;
      _available_pool_cnt++;
    }
}

/** This frees and unused blocks or memory pool structures */
void memCleanup(void)
{
  mempool_t *pool, *nextpool;
  for (pool=_available_pools ;pool; pool = nextpool)
    {
      nextpool = pool->nextpool;
      rwFree(pool);
    }

  memblock_t *block, *nextblock; 
  for (block = _available_blocks; block; block = nextblock)
    {
      nextblock = block->next;
      rwFree(block);
    }
}

void *memGetInternal(memh_t * mh, UINT64_t count)
{
  void * outmem;
  
  assert(count);
  mempool_t *pool = mh->pool;

  /* increase memory size to be aligned to 8b */
  count = (count+ALIGN_MASK) & (~ALIGN_MASK);

  /* check if memory is already available in the pool */
  if (count > pool->available_mem_size)
    {
      memblock_t *newblock;
      /* allocate new memory */
      /* the allocated block should have at least the size BLOCK_SIZE_ALIGNED */
      UINTpt_t blocksize = (count < BLOCK_SIZE_ALIGNED) ? BLOCK_SIZE_ALIGNED :
	count;

      /* check if there are available blocks */
      if ((blocksize == BLOCK_SIZE_ALIGNED) && _available_block_cnt)
	{
	  newblock = _available_blocks;
	  _available_blocks = newblock->next;
	  _available_block_cnt--;
	}
      else
	{
	  /* allocate memory */
	  UINTpt_t blocksizeadj = (blocksize == BLOCK_SIZE_ALIGNED) ?
	    BLOCK_SIZE : blocksize;
	  blocksizeadj += sizeof(memblock_t);
	  UINT8_t *outblk;
	  rwMalloc(outblk, UINT8_t, blocksizeadj);
	  newblock = (memblock_t *) outblk;
	  newblock->datablock = outblk+sizeof(memblock_t);
	  newblock->size = blocksize;
	}

      /* add the new block to the pool */
      UINT8_t oldontop = 0;
      if (pool->blocks)
	if (pool->available_mem_size > (BLOCK_SIZE_ALIGNED-count) ||
	    (count >= BLOCK_SIZE_ALIGNED))
	  oldontop = 1;
      
      if (oldontop)
	{
	  /* previous block stays on top of the list to save memory */
	  newblock->next = pool->blocks->next;
	  pool->blocks->next = newblock;
	}
      else
	{
	  /* new block goes to the top */
	  newblock->next = pool->blocks;
	  pool->blocks = newblock;
	  pool->memptr = newblock->datablock+count;
	  pool->available_mem_size = blocksize-count;
	}

      outmem = newblock->datablock;
    }
  else
    {
      /* use available memory from the pool */
      outmem = pool->memptr;
      pool->available_mem_size -= (UINTpt_t) count;
      pool->memptr += (UINTpt_t) count;
    }

  return outmem;
}
