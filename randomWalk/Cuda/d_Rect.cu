#include "d_Rect.h"

__host__ __device__ d_Rect::d_Rect(floatingPoint xTL, floatingPoint yTL, floatingPoint xBR, floatingPoint yBR)
{
	topLeft.x=xTL;
	topLeft.y=yTL;
	bottomRight.x =xBR;
	bottomRight.y=yBR;
}
__device__ bool d_Rect::rectsCollision(d_Rect const& r2) const{
    if (bottomRight.x >= r2.topLeft.x &&
            r2.bottomRight.x >= topLeft.x &&
            bottomRight.y >= r2.topLeft.y &&
            r2.bottomRight.y >= topLeft.y)
            return true;

        return false;
}

__device__ floatingPoint d_Rect::getWidth() const
{
	return (bottomRight.x - topLeft.x);
}
__device__ floatingPoint d_Rect::getHeigth() const
{
	return (bottomRight.y - topLeft.y);
}

__device__ floatingPoint d_Rect::getPerimeter() const
{
	return (2 * (this->bottomRight.x - this->topLeft.x) + 2 * (this->bottomRight.y - this->topLeft.y));
}

__device__ point2 d_Rect::getPointFromNindex(int index, int Nsample)
{
	floatingPoint perimeter = this->getPerimeter();
	floatingPoint vector = (floatingPoint)perimeter / (floatingPoint)Nsample;
	floatingPoint heigth = this->getHeigth();
	floatingPoint width = this->getWidth();
	point2 ret;
	if (index*vector < width)
	{
		ret.x = (int)((index-1)*vector +vector / 2 + this->topLeft.x);
		ret.y = this->topLeft.y;
		return ret;
	}
	else if ((index - 1)*vector < width &&index*vector > width)                 //JEZELI PRZECHODZI PRZEZ KRAWEDZ TO DAJE WIESZCHOLEK
	{
		ret.x = this->bottomRight.x;
		ret.y = this->topLeft.y;
		return ret;
	}
	else if (index*vector < (width + heigth))
	{
		ret.x = this->bottomRight.x;
		if ((index - 1)*vector > width)
		{
			ret.y = this->topLeft.y + (index - 1)*vector - width + vector / 2;
		}
		else
		{
			ret.y = this->topLeft.y + index*vector - width + (index*vector - width)/2;
		}
		return ret;
	}
	else if ((index - 1)*vector < (width + heigth) && vector*index>(width + heigth))
	{
		return this->bottomRight;
	}
	else if (index*vector < (2 * width + heigth))
	{
		ret.y = this->bottomRight.y;
		if ((index - 1)*vector > width + heigth)
		{
			ret.x = this->bottomRight.x - ((index - 1)*vector - width - heigth + vector / 2 );
		}
		else
		{
			ret.x = this->bottomRight.x - (index*vector - width - heigth + (index*vector - width - heigth) / 2);
		}
		return ret;
	}
	else if ((index - 1)*vector < (2 * width + heigth) && vector*index>(2 * width + heigth))
	{
		ret.x = this->topLeft.x;
		ret.y = this->bottomRight.y;
		return ret;
	}
	else{
		ret.x = this->topLeft.x;
		if ((index - 1)*vector > 2*width + heigth)
		{
			ret.y = this->bottomRight.y - ((index - 1)*vector - 2*width - heigth + vector / 2);
		}
		else
		{
			ret.y = this->bottomRight.y - (index*vector - 2*width - heigth + (index*vector - 2*width - heigth) / 2);
		}
		return ret;
	}
}

__device__ d_Rect d_Rect::createGaussianSurface(floatingPoint factorX, floatingPoint factorY) const
{
    floatingPoint middleX = floatingPoint(topLeft.x + bottomRight.x) / 2.;
    floatingPoint middleY = floatingPoint(topLeft.y + bottomRight.y) / 2.;
    floatingPoint vectorX = floatingPoint(topLeft.x) - middleX;
    floatingPoint vectorY = floatingPoint(topLeft.y) - middleY;
    d_Rect gaussSurface;
    vectorX *= factorX;
    vectorY *= factorY;

    gaussSurface.topLeft.x = middleX + vectorX;
    gaussSurface.topLeft.y = middleY + vectorY;
    gaussSurface.bottomRight.x = middleX - vectorX;
    gaussSurface.bottomRight.y = middleY - vectorY;

    return gaussSurface;
}

__device__ bool d_Rect::operator==(const d_Rect & r2) const
{
    if (r2.topLeft.x == topLeft.x &&
        r2.topLeft.y == topLeft.y &&
        r2.bottomRight.x == bottomRight.x &&
        r2.bottomRight.y == bottomRight.y)
        return true;
    else
        return false;
}
