#include "Rect.h"

#include <math.h>

Rect::Rect()
{
}

Rect::Rect(int topLeftX, int topLeftY, int bottomRightX, int bottomRightY)
{
	this->topLeft.x = topLeftX;
	this->topLeft.y = topLeftY;
	this->bottomRight.x = bottomRightX;
	this->bottomRight.y = bottomRightY;
}

Rect::Rect(point topLeft,point bottomRight)
{
	this->topLeft = topLeft;
	this->bottomRight = bottomRight;
}


Rect::~Rect()
{
}

int Rect::getWidth() const {
	return (bottomRight.x - topLeft.x);
}

int Rect::getHeigth() const {
	return (bottomRight.y - topLeft.y);
}

bool Rect::cmpRect(Rect const& r2) const {
	if (this->bottomRight.x == r2.bottomRight.x &&
		this->bottomRight.y == r2.bottomRight.y &&
		this->topLeft.x == r2.topLeft.x &&
		this->topLeft.y == r2.topLeft.y)
		return true;
	else
		return false;
}

bool Rect::rectsCollision(Rect const& r2) const {
	/*ponizsze warunki powinny takze wykryc sytuacje gdy r2 calkowiecie zawiera sie w r1*/
	/*jezeli r2 upper/lower y zawiera sie miedzy r1 lower y i upper y*/
	if ((this->topLeft.y <= r2.topLeft.y && r2.topLeft.y <= this->bottomRight.y) ||
		(this->topLeft.y <= r2.bottomRight.y && r2.bottomRight.y <= this->bottomRight.y))
	{
		/*jezeli r2 left/right zawiera sie miedzy r1 left i right
				+----------+
				|          |
		+-------+--+  r2   |
		|       |  |       |
		|    r1 +--+-------+
		|          |
		+----------+
		*/
		if ((this->topLeft.x <= r2.topLeft.x && r2.topLeft.x <= this->bottomRight.x) ||
			(this->topLeft.x <= r2.bottomRight.x && r2.bottomRight.x <= this->bottomRight.x)) return true;
		/*jezeli r2 left jest mniejsze od r1 left i r2 right jest wieksze od r2 right
		+---------------------+
		|                     |
		|  +----------+  r2   |
		+--+----------+-------+
		   |    r1	  |
		   |          |
		   +----------+
		*/
		if (this->topLeft.x <= r2.topLeft.x && r2.bottomRight.x >= this->bottomRight.x) return true;
	}
	/*jezeli r2 left/right zawiera sie miedzy r1 left i right*/
	else if ((this->topLeft.x <= r2.topLeft.x && r2.topLeft.x <= this->bottomRight.x) ||
		(this->topLeft.x <= r2.bottomRight.x && r2.bottomRight.x <= this->bottomRight.x))
	{
		/*jezeli r2 upper/lower zaiwra sie miedzy r1 lower i upper
		patrz rysunek wyzej*/
		if ((this->topLeft.y <= r2.topLeft.y && r2.topLeft.y <= this->bottomRight.y) ||
			(this->topLeft.y <= r2.bottomRight.y && r2.bottomRight.y <= this->bottomRight.y)) return true;
		/*jezeli r2 upper jest mniejszy od r1 upper i r2 lower jest wiekszy od r1 lower*/
		if ((this->topLeft.y >= r2.topLeft.y && r2.bottomRight.y >= this->bottomRight.y)) return true;

	}
	return false;
}

bool Rect::rectContains(point p) const {
	if ((topLeft.x <= p.x) && (topLeft.y <= p.y) && (bottomRight.x >= p.x) && (bottomRight.y >= p.y)) return true;
	else return false;
}

Rect Rect::createGaussianSurface(double factor) const {
	double middleX = double(topLeft.x + bottomRight.x) / 2.;
	double middleY = double(topLeft.y + bottomRight.y) / 2.;
	double vectorX = double(topLeft.x) - middleX;
	double vectorY = double(topLeft.y) - middleY;
	Rect gaussSurface;
	vectorX *= factor;
	vectorY *= factor;
	
	gaussSurface.topLeft.x = int(round(middleX + vectorX));
	gaussSurface.topLeft.y = int(round(middleY + vectorY));
	gaussSurface.bottomRight.x = int(round(middleX - vectorX));
	gaussSurface.bottomRight.y = int(round(middleY - vectorY));

	return gaussSurface;
}

int Rect::getPerimeter() const
{
	return (2 * (this->bottomRight.x - this->topLeft.x) + 2 * (this->bottomRight.y - this->topLeft.y));
}

point Rect::getPointFromNindex(int index, int Nsample) const {//MAM NADZIEJE ZE NIGDZIE SIE NIE WALNALEM BO TO SKOMPLIKOWANA GEOMETRIA
	int perimeter = this->getPerimeter();
	double vector = (double)perimeter / (double)Nsample;
	int heigth = this->getHeigth();
	int width = this->getWidth();
	point ret;
	if (index*vector < width){
		ret.x = (int)((index-1)*vector +vector / 2 + this->topLeft.x);
		ret.y = this->topLeft.y;
		return ret;
	}
	else if ((index - 1)*vector < width &&index*vector > width){//JEZELI PRZECHODZI PRZEZ KRAWEDZ TO DAJE WIESZCHO£EK
		ret.x = this->bottomRight.x;
		ret.y = this->topLeft.y;
		return ret;
	}
	else if (index*vector < (width + heigth)){
		ret.x = this->bottomRight.x;
		if ((index - 1)*vector > width){
			ret.y = this->topLeft.y + (index - 1)*vector - width + vector / 2;
		}
		else{
			ret.y = this->topLeft.y + index*vector - width + (index*vector - width)/2;
		}
		return ret;
	}
	else if ((index - 1)*vector < (width + heigth) && vector*index>(width + heigth)){
		return this->bottomRight;
	}
	else if (index*vector < (2 * width + heigth)){
		ret.y = this->bottomRight.y;
		if ((index - 1)*vector > width + heigth){
			ret.x = this->bottomRight.x - ((index - 1)*vector - width - heigth + vector / 2 );
		}
		else{
			ret.x = this->bottomRight.x - (index*vector - width - heigth + (index*vector - width - heigth) / 2);
		}
		return ret;
	}
	else if ((index - 1)*vector < (2 * width + heigth) && vector*index>(2 * width + heigth)){
		ret.x = this->topLeft.x;
		ret.y = this->bottomRight.y;
		return ret;
	}
	else{
		ret.x = this->topLeft.x;
		if ((index - 1)*vector > 2*width + heigth){
			ret.y = this->bottomRight.y - ((index - 1)*vector - 2*width - heigth + vector / 2);
		}
		else{
			ret.y = this->bottomRight.y - (index*vector - 2*width - heigth + (index*vector - 2*width - heigth) / 2);
		}
		return ret;
	}
}
ostream& operator<< (ostream &wyjscie, Rect const& ex)
{
	wyjscie << "Rect TL: x:" << ex.topLeft.x<<", y:" << ex.topLeft.y << "  BR: x:" << ex.bottomRight.x << ", y:" << ex.bottomRight.y << endl;
	return wyjscie;
}