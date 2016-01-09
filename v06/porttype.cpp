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

/** This file contains global variables for portable variables e.g. IEEE infinity */

#include "porttype.h"

/** IEEE infinity in little endian systems such as Linux, Windows */
IEEE_INF_t ieee_inf = {{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x7f}};

/* for big endian */
/* IEEE_INF_t ieee_inf = {{0x7f, 0xf0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}}; */
