#include "Rect.h"

#include <math.h>

Rect::Rect()
{
}


Rect::~Rect()
{
}

int Rect::getWidth() {
	return (bottomRight.x - topLeft.x);
}

int Rect::getHeigth() {
	return (bottomRight.y - topLeft.y);
}

bool Rect::cmpRect(Rect r2) {
	if (this->bottomRight.x == r2.bottomRight.x &&
		this->bottomRight.y == r2.bottomRight.y &&
		this->topLeft.x == r2.topLeft.x &&
		this->topLeft.y == r2.topLeft.y)
		return true;
	else
		return false;
}

bool Rect::rectsCollision(Rect r2) {
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

bool Rect::rectContains(point p) {
	if ((topLeft.x <= p.x) && (topLeft.y <= p.y) && (bottomRight.x >= p.x) && (bottomRight.y >= p.y)) return true;
	else return false;
}

Rect Rect::createGaussianSurface(double factor) {
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