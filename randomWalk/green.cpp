#define _USE_MATH_DEFINES

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
/** This isInBounds code for computing 2D Green's functions on a square. */

#include "porttype.h"
#include "green.h"
#include "phys.h"

/** This specifies the number of terms in the series used to compute
    Green's function. TODO: investigate series convergence */
#define NTERMS 20 

/** This computes a value for 2D Green's function specified on a 
    square. a equals the side of the square and eta is a variable
    used to parametrize the square perimeter: eta varies from 0 to 4*a, 
    starting in the lower left corner and going clockwise. */
REAL64_t green_square(REAL64_t a, REAL64_t eta)
{
  assert(eta <= 4*a);
  assert(eta >= 0);

  REAL64_t g = 0.0;
  REAL64_t eta_scaled = M_PI*eta/a;
  REAL64_t sign = 1.0;
  REAL64_t expy, oneminusexp;
  UINT64_t n;
  for (n = 1; n <= NTERMS; n += 2) 
    {
      expy = exp(-0.5*n*M_PI);
      oneminusexp = 1.0 - exp(-2.0*M_PI*n);
      g += sign*(expy*(1.0-expy*expy)/oneminusexp)*sin(n*eta_scaled);
      sign *= (-1);
    }
 
    if (eta <= a)
      g *= 2/a;
    else if (eta <= 2*a)
      g *= -2/a;
    else if (eta <= 3*a)
      g *= 2/a;
    else
      g *= -2/a;
    return g;
}

/** This computes derivatives for Green's function specified on a square.
    dg/dx and dg/dy are returned in dg[0] and dg[1], respectively. */
void deriv_green_square(REAL64_t * dg, REAL64_t a, REAL64_t eta)
{
  assert(eta <= 4*a);
  assert(eta >= 0);
  assert(a > 0);

  dg[0] = 0.0;
  dg[1] = 0.0;
  REAL64_t eta_scaled = M_PI*eta/a;
  REAL64_t x_scaled = 0.5*M_PI;
  UINT16_t n;

  if (eta <= a)
    {
      for (n = 1; n <= NTERMS; n++)
	{
	  REAL64_t expy = exp(-1.0*n*x_scaled);
	  REAL64_t oneminusexp = 1.0 - exp(-2.0*M_PI*n);
	  assert(oneminusexp);
	  dg[0] += n*(expy*(1.0-expy*expy)/oneminusexp)*cos(n*x_scaled)*sin(n*eta_scaled);
	  dg[1] += n*(expy*(1.0+expy*expy)/oneminusexp)*sin(n*x_scaled)*sin(n*eta_scaled);
	}
      dg[0] *= -2*M_PI/(a*a);
      dg[1] *= 2*M_PI/(a*a);
    }
  else if (eta <= 2*a)
    {
      for (n = 1; n <= NTERMS; n++)
	{
	  REAL64_t expx = exp(-1.0*n*x_scaled);
	  REAL64_t oneminusexp = 1.0 - exp(-2.0*M_PI*n);
	  assert(oneminusexp);
	  dg[0] += n*(expx*(1.0+expx*expx)/oneminusexp)*sin(n*x_scaled)*sin(n*eta_scaled-n*M_PI);
	  dg[1] += n*(expx*(1.0-expx*expx)/oneminusexp)*cos(n*x_scaled)*sin(n*eta_scaled-n*M_PI);
	}
      dg[0] *= -2*M_PI/(a*a);
      dg[1] *= -2*M_PI/(a*a);
    }
  else if (eta <= 3*a)
    {
      for (n = 1; n <= NTERMS; n++)
	{
	  REAL64_t expy = exp(-1.0*n*x_scaled);
	  REAL64_t oneminusexp = 1.0 - exp(-2.0*M_PI*n);
	  dg[0] += n*(expy*(1.0-expy*expy)/oneminusexp)*cos(n*x_scaled)*sin(n*eta_scaled);
	  dg[1] += n*(expy*(1.0+expy*expy)/oneminusexp)*sin(n*x_scaled)*sin(n*eta_scaled);
	}
      dg[0] *= 2*M_PI/(a*a);
      dg[1] *= -2*M_PI/(a*a);
    }
  else
    {    
      for (n = 1; n <= NTERMS; n++)
	{
	  REAL64_t expx = exp(-1.0*n*x_scaled);
	  REAL64_t oneminusexp = 1.0 - exp(-2.0*M_PI*n);
	  assert(oneminusexp);
	  dg[0] += n*(expx*(1.0+expx*expx)/oneminusexp)*sin(n*x_scaled)*sin(n*eta_scaled-n*M_PI);
	  dg[1] += n*(expx*(1.0-expx*expx)/oneminusexp)*cos(n*x_scaled)*sin(n*eta_scaled-n*M_PI);
	}
      dg[0] *= 2*M_PI/(a*a);
      dg[1] *= 2*M_PI/(a*a);
    }
}

/* This precomputes values of surface Green's function at 1/2,1/2 and its derivatives
   for a unit square. */
void precompute_unit_square_green(REAL64_t * g, REAL64_t * dgdx, REAL64_t * dgdy,
				  REAL64_t * intg, UINTpt_t Nsample)
{
  REAL64_t eta, delta_eta, dg[2];
  delta_eta = 4.0/Nsample;

  UINTpt_t i;
  for (i = 0; i < Nsample; i++)
    {
      eta = (i + 0.5) * delta_eta;
      assert(eta < 4.0);
      g[i] = green_square(1.0, eta);
      deriv_green_square(dg, 1.0, eta);
      dgdx[i] = dg[0];
      dgdy[i] = dg[1];
    }

  /* Integrate g over the unit square perimeter. This
     can also be computed analytically if there is a need */
  intg[0] = 0.0;
  for (i=1; i <= Nsample; i++)
    intg[i] = intg[i-1]+g[i-1]*delta_eta;
  
  /* normalize - required unless analytical integration is used */
  REAL64_t alpha = 1.0/intg[Nsample];
  for (i = 0; i <= Nsample; i++)
    intg[i] *= alpha;
}

/*  flength = fill_surf(1,2,1)-fill_surf(1,1,1);
    fwidth = fill_surf(1,3,2)-fill_surf(1,2,2); */
/* This precomputes integrated Green's function for a metal fill rectangle of size:
   flength (x-coord) x fwidth (y-coord). When computing values for Green's function,
   it is assumed that there is a rectangular Gaussian surface which surrounds the fill rectange, and
   the distance between the Gaussian surface and the fill surface equals to "spacing" in terms of
   Manhattan norm. It is also assumed that the fill is surrounded by free space with permittivity
   = epsilon_0. If a dielectric with relative permittivity eps_r surrounds the metal fill, the
   value for Green's function should be multiplied by e_r.
*/
void precompute_green_fill(REAL64_t * intg_fill, REAL64_t flength, REAL64_t fwidth,
			   REAL64_t spacing, UINTpt_t Nsamp)
{
  memh_t locmem;
  memInit(&locmem);

  UINTpt_t i;
  REAL64_t delta_xi, length, templ;
  delta_xi = 2*(flength+fwidth)/Nsamp;

  /* values for Green's function are computed along fill's perimeter, parametrized by variable
     "length" */
  REAL64_t * green_fill;
  UINTpt_t halfnsamp = floor((REAL64_t)Nsamp/2);
  memGet(green_fill, &locmem, REAL64_t, halfnsamp);
  for (i = 0; i < halfnsamp; i++)
    {
      length = (i+0.5)*delta_xi;
      if (length < spacing)
	{
	  templ = spacing - length;
	  green_fill[i] = permittivity / sqrt(spacing*spacing+templ*templ);
	}
      else if (length < flength-spacing)
	green_fill[i] = permittivity / spacing;
      else if (length < flength)
	{
	  templ = length - (flength-spacing);	
	  green_fill[i] = permittivity / sqrt(spacing*spacing+templ*templ);
	}
      else if (length < flength+spacing)
	{
	  templ = spacing - (length - flength);	
	  green_fill[i] = permittivity / sqrt(spacing*spacing+templ*templ);
	}
      else if (length < flength + fwidth - spacing)
	green_fill[i] = permittivity / spacing;
      else
	{
	  templ = length - (flength+fwidth-spacing);	
	  green_fill[i] = permittivity / sqrt(spacing*spacing+templ*templ);
	}
    }
  
  /* integrate */
  intg_fill[0] = 0.0;
  
  for (i = 1; i <= halfnsamp; i++)
    intg_fill[i] = intg_fill[i-1]+green_fill[i-1]*delta_xi;
  for (i = halfnsamp+1; i <= 2*halfnsamp; i++)
    intg_fill[i] = intg_fill[i-1]+green_fill[i-halfnsamp-1]*delta_xi;
  if ((2*halfnsamp) < Nsamp)
    intg_fill[Nsamp] = intg_fill[Nsamp - 1]
	+ permittivity / (sqrt((REAL64_t)2)*spacing) * delta_xi;

  memFree(&locmem);

  /* normalize */
  REAL64_t alpha = 1.0/intg_fill[Nsamp];
  for (i = 0; i <= Nsamp; i++)
    intg_fill[i] *= alpha;
}

/** This computes a value for 3D Green's function specified on a 
    cube. a is the edge length for the cube  and eta, dzeta are variables
    used to parametrize the square face for the cube: eta, dzeta vary from 0 to 6*a.
    FACE1 = (eta,dzeta,0), FACE2=(a,eta-a,dzeta-a), FACE3 = (3a-dzeta, a, eta-2a),
    FACE4 = (4a-eta, 4a-dzeta, a), FACE5=(0, 5a-eta, 5a-dzeta), FACE6=(dzeta-5a, 0, 6a-eta). */
REAL64_t green_cube(REAL64_t a, REAL64_t eta, REAL64_t dzeta)
{
  UINT8_t i;
  assert(eta <= 6*a);
  assert(eta >= 0);
  assert(dzeta <= 6*a);
  assert(dzeta >= 0);
  for (i = 1; i < 6; i++)
    {
      if (eta < i*a)
	assert(dzeta < i*a);
      if (dzeta < i*a)
	assert(eta < i*a);
    }
  
  REAL64_t g = 0.0;
  REAL64_t eta_scaled = M_PI*eta/a;
  REAL64_t dzeta_scaled = M_PI*dzeta/a;
  REAL64_t signm, signn;
  REAL64_t expy, oneminusexp;
  UINT64_t n,m;
  for (m = 1, signm = 1.0; m <= NTERMS; m += 2, signm *= (-1))
    {
      REAL64_t sineta = signm * sin(m*eta_scaled);
      for (n = 1, signn = 1.0; n <= NTERMS; n += 2, signn *= (-1)) 
	{
		  REAL64_t sqrtmn_scaled = -0.5*sqrt((REAL64_t)(m*m + n*n))*M_PI;
	  expy = exp(sqrtmn_scaled);
	  oneminusexp = 1.0 - exp(4.0*sqrtmn_scaled);
	  g += signn*(expy*(1.0-expy*expy)/oneminusexp)*sin(n*dzeta_scaled)*sineta;
	}
    }

  g *= (4/(a*a));
  return g;
}

/** This computes 'generic' derivaties for Green's function for a face (side) of a cube  */
static void dgdxyz(REAL64_t * dg1, REAL64_t * dg2, REAL64_t * dg3, REAL64_t etascaled, REAL64_t dzetascaled)
{
  REAL64_t signsinm, signcosm, signsinn, signcosn, expy, oneminusexp;
  UINT32_t m, n;
  (*dg1) = (*dg2) = (*dg3) = 0.0;
  for (m = 1, signsinm = 1.0, signcosm = -1.0; m <= NTERMS; m++)
    {
      REAL64_t sineta = sin(m*(etascaled));
      for (n = 1, signsinn = 1.0, signcosn = -1.0; n <= NTERMS; n++)
	{
		  REAL64_t sqrtmn_scaled = -0.5*sqrt((REAL64_t)(m*m + n*n))*M_PI;
	  expy = exp(sqrtmn_scaled);
	  oneminusexp = 1.0 - exp(4.0*sqrtmn_scaled);
	  if (n % 2)
	    {
	      if (m % 2)
			  (*dg1) += sqrt((REAL64_t)(m*m + n*n))*(expy*(1.0 + expy*expy) / oneminusexp)
		  *sin(n*(dzetascaled))*sineta*signsinm*signsinn;
	      else
		(*dg2) += m*(expy*(1.0-expy*expy)/oneminusexp)
		  *sin(n*(dzetascaled))*sineta*signcosm*signsinn;
	      signsinn *= (-1); 
	    }
	  else
	    {
	      if (m % 2) /* add to dgdy */
		(*dg3) += n*(expy*(1.0-expy*expy)/oneminusexp)
		  *sin(n*(dzetascaled))*sineta*signsinm*signcosn;
	      signcosn *= (-1);
	    }
	}
      if (m % 2)
	signsinm *= (-1);
      else
	signcosm *= (-1);
    }
}

/** This computes derivatives for Green's function specified on a cube.
    -dg/dx, -dg/dy and -dg/dz are returned in dg[0], dg[1] and dg[2], respectively.
    Please note the minus sign! */
void deriv_green_cube(REAL64_t * dg, REAL64_t a, REAL64_t eta, REAL64_t dzeta)
{
  UINT8_t i;
  assert(eta <= 6*a);
  assert(eta >= 0);
  assert(dzeta <= 6*a);
  assert(dzeta >= 0);
  for (i = 1; i < 6; i++)
    {
      if (eta < i*a)
	assert(dzeta < i*a);
      if (dzeta < i*a)
	assert(eta < i*a);
    }
  
  REAL64_t scaling = -4.0*M_PI/(a*a*a);
  REAL64_t eta_scaled, dzeta_scaled;
  if (eta < a)
    {
      assert(dzeta < a);
      eta_scaled = M_PI*eta/a;
      dzeta_scaled = M_PI*dzeta/a;
      dgdxyz(dg+2, dg+0, dg+1, eta_scaled, dzeta_scaled);
      dg[2] *= (-1);
    }
  else if (eta < 2*a)
    {
      assert(dzeta >= a);
      assert(dzeta < 2*a);
      eta_scaled = M_PI*(eta-a)/a;
      dzeta_scaled = M_PI*(dzeta-a)/a;
      dgdxyz(dg+0, dg+1, dg+2, eta_scaled, dzeta_scaled);
    }
  else if (eta < 3*a)
    {
      assert(dzeta >= 2*a);
      assert(dzeta < 3*a);
      eta_scaled = M_PI*(eta-2*a)/a;
      dzeta_scaled = M_PI*(dzeta-2*a)/a;
      dgdxyz(dg+1, dg+0, dg+2, dzeta_scaled, eta_scaled);
      dg[0] *= (-1);
    }
  else if (eta < 4*a)
    {
      assert(dzeta >= 3*a);
      assert(dzeta < 4*a);
      eta_scaled = M_PI*(eta-3*a)/a;
      dzeta_scaled = M_PI*(dzeta-3*a)/a;
      dgdxyz(dg+2, dg+0, dg+1, eta_scaled, dzeta_scaled);
      dg[0] *= (-1);
      dg[1] *= (-1);
    }
  else if (eta < 5*a)
    {
      assert(dzeta >= 4*a);
      assert(dzeta < 5*a);
      eta_scaled = M_PI*(eta-4*a)/a;
      dzeta_scaled = M_PI*(dzeta-4*a)/a;
      dgdxyz(dg+0, dg+1, dg+2, eta_scaled, dzeta_scaled);
      dg[0] *= (-1);
      dg[1] *= (-1);
      dg[2] *= (-1);
    }
  else 
    {
      assert(dzeta >= 5*a);
      assert(dzeta < 6*a);
      eta_scaled = M_PI*(eta-5*a)/a;
      dzeta_scaled = M_PI*(dzeta-5*a)/a;
      dgdxyz(dg+1, dg+0, dg+2, dzeta_scaled, eta_scaled);
      dg[1] *= (-1);
      dg[2] *= (-1);
    }

  dg[0] *= scaling;
  dg[1] *= scaling;
  dg[2] *= scaling;
}

/* This precomputes values of surface Green's function at 1/2,1/2 and its derivatives
   for a unit square. */
void precompute_unit_cube_green(REAL64_t * g, REAL64_t * dgdx, REAL64_t * dgdy, REAL64_t * dgdz,
				REAL64_t * intg, UINTpt_t Nsample)
{
  REAL64_t eta, dzeta, delta_eta, delta_dzeta, dg[3];
  delta_dzeta = delta_eta = 1.0/Nsample;

  UINTpt_t i, j, k;
  UINTpt_t ldg = Nsample;
  REAL64_t prevint, prevg;
  
  prevint = 0.0;
  prevg = 0.0;
  for (k = 0; k < 6; k++)
    for (i = k*Nsample; i < (k+1)*Nsample; i++)
      for (j = 0; j < Nsample; j++)
	{
	  eta = (i + 0.5) * delta_eta;
	  dzeta = (k*Nsample+j+0.5)*delta_dzeta;
	  assert(eta < 6.0);
	  assert(dzeta < 6.0);
	  assert(i*ldg+j < 6*Nsample*Nsample);
	  g[i*ldg+j] = green_cube(1.0, eta, dzeta);
	  deriv_green_cube(dg, 1.0, eta, dzeta);
	  dgdx[i*ldg+j] = dg[0];
	  dgdy[i*ldg+j] = dg[1];
	  dgdz[i*ldg+j] = dg[2];

	  /* Compute int(g) over the unit cube surface. This
	     can also be computed analytically if there is a need */
	  intg[i*ldg+j] = prevint + prevg*delta_eta*delta_dzeta;
	  prevint = intg[i*ldg+j];
	  prevg = g[i*ldg+j];
	}
  intg[6*Nsample*Nsample] = prevint + prevg*delta_eta*delta_dzeta;
    
  /* normalize - required unless analytical integration is used */
  REAL64_t alpha = 1.0/intg[6*Nsample*Nsample];
  for (i = 0; i <= 6*Nsample*Nsample; i++)
    intg[i] *= alpha;
}

/** This precomputes integrated Green's function for a metal fill rectangle of size:
   flength (x-coord) x fwidth (y-coord). When computing values for Green's function,
   it is assumed that there is a rectangular Gaussian surface which surrounds the fill rectange, and
   the distance between the Gaussian surface and the fill surface equals to "spacing" in terms of
   Manhattan norm. It is also assumed that the fill is surrounded by free space with permittivity
   = epsilon_0. If a dielectric with relative permittivity eps_r surrounds the metal fill, the
   value for Green's function should be multiplied by e_r.
*/
void precompute_green_fill3D(REAL64_t * intg_fill, REAL64_t flength, REAL64_t fwidth, REAL64_t fheight,
			   REAL64_t spacing, UINTpt_t Nsamp)
{
  memh_t locmem;
  memInit(&locmem);

  UINTpt_t i, j, k;

  /* values for Green's function are computed on the Gaussian surface around the fill surface, parametrized by variable
     "length" */
  REAL64_t * green_fill;
  REAL64_t etadim, dzetadim, deltaeta, deltadzeta;
  UINTpt_t halfnsamp = 3*Nsamp*Nsamp;
  memGet(green_fill, &locmem, REAL64_t, halfnsamp);
  for (k = 0; k < 3; k++)
    {
      switch (k) {
      case 0:
	etadim = flength;
	dzetadim = fwidth;
	break;
      case 1:
	etadim = fwidth;
	dzetadim = fheight;
	break;
      case 2:
	etadim = fheight;
	dzetadim = flength;
	break;
      default:
	assert(0);
	break;
      }
      deltaeta = etadim/Nsamp;
      deltadzeta = dzetadim/Nsamp;
      for (i = 0; i < Nsamp; i++)
	{
	  REAL64_t eta, dzeta, tempeta, tempdzeta;
	  eta = (i + 0.5)*deltaeta;
	  if (eta < spacing)
	    tempeta = spacing - eta;
	  else if (eta < etadim-spacing)
	    tempeta = 0.0;
	  else
	    tempeta = eta - (etadim-spacing);
	    
	  for (j = 0; j < Nsamp; j++)
	    {
	      dzeta = (j+0.5)*deltadzeta;
	      if (dzeta < spacing)
		tempdzeta = spacing - dzeta;
	      else if (dzeta < dzetadim-spacing)
		tempdzeta = 0.0;
	      else
		tempdzeta = dzeta - (dzetadim-spacing);

	      assert(k*Nsamp*Nsamp+i*Nsamp+j < halfnsamp);
	      green_fill[k*Nsamp*Nsamp+i*Nsamp+j]
		= permittivity/sqrt(spacing*spacing+tempeta*tempeta+
				    tempdzeta*tempdzeta);
	    }
	}
    }
  
  /* integrate */
  intg_fill[0] = 0.0;

  UINTpt_t totalidx = 1;
  for (k = 0; k < 3; k++)
    {
      switch (k) {
      case 0:
	etadim = flength;
	dzetadim = fwidth;
	break;
      case 1:
	etadim = fwidth;
	dzetadim = fheight;
	break;
      case 2:
	etadim = fheight;
	dzetadim = flength;
	break;
      default:
	assert(0);
	break;
      }
      deltaeta = etadim/Nsamp;
      deltadzeta = dzetadim/Nsamp;
      for (i = 0; i < Nsamp; i++)
	for (j = 0; j < Nsamp; j++)
	  {
	    assert(totalidx <= halfnsamp);
	    intg_fill[totalidx] = intg_fill[totalidx-1]
	      +green_fill[totalidx-1]*(deltaeta*deltadzeta);
	    totalidx++;
	  }
    }
  UINTpt_t idxoffset = totalidx-1;
  assert(idxoffset == (3*Nsamp*Nsamp));
  for (i = 1; i <= 3*Nsamp*Nsamp; i++)
    intg_fill[idxoffset+i] = intg_fill[idxoffset]+intg_fill[i];
  assert((idxoffset+i) == (6*Nsamp*Nsamp+1));
  
  memFree(&locmem);
     
  /* normalize */
  REAL64_t alpha = 1.0/intg_fill[6*Nsamp*Nsamp];
for (i = 0; i <= (6*Nsamp*Nsamp); i++)
    intg_fill[i] *= alpha;
}
