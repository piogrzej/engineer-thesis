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
#ifndef GREEN_H
#define GREEN_H

#include "mempool.h"
#include "randgen.h"

/** This computes a value for 2D Green's function specified on a 
    square. a equals the side of the square and eta is a variable
    used to parametrize the square perimeter: eta varies from 0 to 4*a, 
    starting in the lower left corner and going clockwise. */
REAL64_t green_square(REAL64_t a, REAL64_t eta);

/** This computes derivatives for Green's function specified on a square.
    dg/dx and dg/dy are returned in dg[0] and dg[1], respectively. */
void deriv_green_square(REAL64_t * dg, REAL64_t a, REAL64_t eta);

/* This precomputes values of surface Green's function at 1/2,1/2 and its derivatives
   for a unit square. */

/*
Najbardziej przyda siê Panu funkcja w pliku green.c:

precompute_unit_square_green() - ta funkcja liczy rozk³ad
prawdopodobieñstwa w przypadku 2D (dla brzegu kwadratu):
Dzielimy obwód kwadratu (wszystkie 4 boki) na Nsample czêœci (np. Nsample = 200) -
tablica "g" (o d³ugoœci Nsample) okreœla prawdopodobieñstwo przeskoku z œrodka kwadratu na kolejne odcinki na obwodzie kwadratu.
Funkcja zwraca te¿ zca³kowany rozk³ad prawdopodobieñstwa "intg" (tablica o d³ugoœci Nsample+1), który stosuje siê nastêpuj¹co:
1) wybieramy losowo liczbê "p" od 0 do 1;
2) Wyszukujemy w tablicy intg (jest ju¿ posortowana) indeks i taki, ¿e intg[i] <= p < intg[i].
3) index "i" okreœla, na który odcinek na kwadracie przeskoczymy w kolejnym kroku.

Do³¹czy³em jeszcze kody do generacji liczb pseudolosowych - w pliku
randgen.c. Nale¿y najpierw "rozgrzaæ" generator pseudolosowy za pomoc¹
funkcji: rng_init( typ) gdzie typ=0,1,2,3 (polecam 1 lub 2, ale mo¿e byæ
te¿ 3 (Mersenne twister)) W funkcji rng_init pojawia siê flaga
"engineopt.rng_random_seed" - by uzyskaæ deterministyczne wyniki pomiêdzy
wykonaniami nale¿y ustawiæ tê flagê na 0. Nastêpnie liczbê pseudolosow¹
mo¿na uzyskaæ wywo³uj¹c "myrand()"
*/
void precompute_unit_square_green(REAL64_t * g, REAL64_t * dgdx, REAL64_t * dgdy,
				  REAL64_t * intg, UINTpt_t Nsample);

/* This precomputes integrated Green's function for a metal fill rectangle of size:
   flength (x-coord) x fwidth (y-coord). When computing values for Green's function,
   it is assumed that there is a rectangular Gaussian surface which surrounds the fill rectange, and
   the distance between the Gaussian surface and the fill surface equals to "spacing" in terms of
   Manhattan norm. It is also assumed that the fill is surrounded by free space with permittivity
   = epsilon_0. If a dielectric with relative permittivity eps_r surrounds the metal fill, the
   value for Green's function should be multiplied by e_r.
*/
void precompute_green_fill(REAL64_t * intg_fill, REAL64_t flength, REAL64_t fwidth,
			   REAL64_t spacing, UINTpt_t Nsamp);

/** 3D GREEN FUNCTIONS */

/** This computes a value for 3D Green's function specified on a 
    cube. a is the edge length for the cube  and eta, dzeta are variables
    used to parametrize the square face for the cube: eta, dzeta vary from 0 to 6*a.
    FACE1 = (eta,dzeta,0), FACE2=(a,eta-a,dzeta-a), FACE3 = (3a-dzeta, a, eta-2a),
    FACE4 = (4a-eta, 4a-dzeta, a), FACE5=(0, 5a-eta, 5a-dzeta), FACE6=(dzeta-5a, 0, 6a-eta). */
REAL64_t green_cube(REAL64_t a, REAL64_t eta, REAL64_t dzeta);

/** This computes derivatives for Green's function specified on a cube.
    dg/dx, dg/dy and dg/dz are returned in dg[0], dg[1] and dg[2], respectively. */
void deriv_green_cube(REAL64_t * dg, REAL64_t a, REAL64_t eta, REAL64_t dzeta);

/* This precomputes values of surface Green's function at 1/2,1/2 and its derivatives
   for a unit square. */
void precompute_unit_cube_green(REAL64_t * g, REAL64_t * dgdx, REAL64_t * dgdy, REAL64_t * dgdz,
				REAL64_t * intg, UINTpt_t Nsample);


/** This precomputes integrated Green's function for a metal fill rectangle of size:
   flength (x-coord) x fwidth (y-coord). When computing values for Green's function,
   it is assumed that there is a rectangular Gaussian surface which surrounds the fill rectange, and
   the distance between the Gaussian surface and the fill surface equals to "spacing" in terms of
   Manhattan norm. It is also assumed that the fill is surrounded by free space with permittivity
   = epsilon_0. If a dielectric with relative permittivity eps_r surrounds the metal fill, the
   value for Green's function should be multiplied by e_r.
*/
void precompute_green_fill3D(REAL64_t * intg_fill, REAL64_t flength, REAL64_t fwidth, REAL64_t fheight,
			     REAL64_t spacing, UINTpt_t Nsamp);

#endif /* GREEN_H */
