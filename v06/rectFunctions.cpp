#include "rectFunctions.h"

int getWidth(rect r){
	return (r.bottom_right.x - r.top_left.x);
}

int getHeigth(rect r){
	return (r.bottom_right.y - r.top_left.y);
}

bool cmpRect(rect r1, rect r2){
	if (r1.bottom_right.x == r2.bottom_right.x &&
		r1.bottom_right.y == r2.bottom_right.y &&
		r1.top_left.x == r2.top_left.x &&
		r1.top_left.y == r2.top_left.y)
		return true;
	else
		return false;
}

bool rectsCollision(rect r1, rect r2){
	/*ponizsze warunki powinny takze wykryc sytuacje gdy r2 calkowiecie zawiera sie w r1*/
	/*jezeli r2 upper/lower y zawiera sie miedzy r1 lower y i upper y*/
	if ((r1.top_left.y <= r2.top_left.y && r2.top_left.y <= r1.bottom_right.y) ||
		(r1.top_left.y <= r2.bottom_right.y && r2.bottom_right.y <= r1.bottom_right.y))
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
		if ((r1.top_left.x <= r2.top_left.x && r2.top_left.x <= r1.bottom_right.x) ||
			(r1.top_left.x <= r2.bottom_right.x && r2.bottom_right.x <= r1.bottom_right.x)) return true;
		/*jezeli r2 left jest mniejsze od r1 left i r2 right jest wieksze od r2 right
			 +---------------------+
			 |                     |
			 |  +----------+  r2   |
			 +--+----------+-------+
				|    r1	   |
				|          |
				+----------+
		*/
		if (r1.top_left.x <= r2.top_left.x && r2.bottom_right.x >= r1.bottom_right.x) return true;
	}
	/*jezeli r2 left/right zawiera sie miedzy r1 left i right*/
	else if ((r1.top_left.x <= r2.top_left.x && r2.top_left.x <= r1.bottom_right.x) ||
		(r1.top_left.x <= r2.bottom_right.x && r2.bottom_right.x <= r1.bottom_right.x))
	{
		/*jezeli r2 upper/lower zaiwra sie miedzy r1 lower i upper
			patrz rysunek wyzej*/
		if ((r1.top_left.y <= r2.top_left.y && r2.top_left.y <= r1.bottom_right.y) ||
			(r1.top_left.y <= r2.bottom_right.y && r2.bottom_right.y <= r1.bottom_right.y)) return true;
		/*jezeli r2 upper jest mniejszy od r1 upper i r2 lower jest wiekszy od r1 lower*/
		if ((r1.top_left.y >= r2.top_left.y && r2.bottom_right.y >= r1.bottom_right.y)) return true;

	}
	return false;
}

bool rectContains(rect r, point p){
	if ((r.top_left.x <= p.x) && (r.top_left.y <= p.y) && (r.bottom_right.x >= p.x) && (r.bottom_right.y >= p.y)) return true;
	else return false;
}