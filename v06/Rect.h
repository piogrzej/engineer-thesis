#ifndef RECT_H
#define RECT_H

struct point {
	int x;
	int y;
};

class Rect
{
public:
	Rect();
	~Rect();
	point topLeft;
	point bottomRight;

	int getWidth();
	int getHeigth();
	bool cmpRect(Rect r2);
	bool rectsCollision(Rect r2);
	bool rectContains(point p);
	Rect createGaussianSurface(double factor);
	int getPerimeter();
	point getPointFromNindex(int index, int Nsample);
};

#endif