#include "mainFunctions.h"


void pointsFormLine(point * topLeft, point * bottomRight, char * line)
{
	//zastepuje znaki ' ' bajtami 0 aby uzyc atoi
	//toeretycznie mozna to zrobic "na sztywno" i takie rozwiazanie byloby najsprawniejsze
	//lecz niewadomo czy struktura pliku wejsciowego sie nie zmieni
	int tmp[4];//lokalizacje liczb
	int j = 0;
	int i = 0;
	while (line[i] != '\n'){
		if (line[i] == ' '){//this is potencially unsafe EXAMPLE: "rect 277 250 371 311 " <-this space at the end of line
			line[i] = 0;
			tmp[j] = i + 1;
			j++;
		}
		i++;
	}
	topLeft->x = atoi((line + tmp[0]));
	topLeft->y = atoi((line + tmp[1]));
	bottomRight->x = atoi((line + tmp[2]));
	bottomRight->y = atoi((line + tmp[3]));
}

//okreslanie rozmiaru przestrzeni
Rect layerSpaceSize(FILE * pFile){
	point tmpTopLeft, tmpBottomRight;
	Rect spaceSize;
	char linebuffor[LINE_BUFFOR_SIZE];//buffor na linie

	while (1){//wyszukiwanie warstwy
		fgets(linebuffor, LINE_BUFFOR_SIZE, pFile);
		if (linebuffor[0] == '<'){
			if (strcmp(linebuffor, "<< metal3 >>\n") == 0){//narazie na sztywno, jedna warstwa
				break;
			}
		}
	}
	//ustawiam na takie wartosci zeby bylo wiadomo ktory obieg jest pierwszy
	bool start = true;
	while (1)
	{
		fgets(linebuffor, LINE_BUFFOR_SIZE, pFile);
		if (linebuffor[0] == 'r')
		{
			pointsFormLine(&tmpTopLeft, &tmpBottomRight, linebuffor);
			if (!start){
				if (tmpTopLeft.x < spaceSize.topLeft.x){
					spaceSize.topLeft.x = tmpTopLeft.x;
				}
				if (tmpTopLeft.y < spaceSize.topLeft.y)
				{
					spaceSize.topLeft.y = tmpTopLeft.y;
				}
				if (tmpBottomRight.x > spaceSize.bottomRight.x){
					spaceSize.bottomRight.x = tmpBottomRight.x;
				}
				if (tmpBottomRight.y > spaceSize.bottomRight.y){
					spaceSize.bottomRight.y = tmpBottomRight.y;
				}
			}
			else
			{
				spaceSize.topLeft.x = tmpTopLeft.x;
				spaceSize.topLeft.y = tmpTopLeft.y;
				spaceSize.bottomRight.x = tmpBottomRight.x;
				spaceSize.bottomRight.y = tmpBottomRight.y;
				start = false;
			}

		}
		else if (linebuffor[0] == '<')
		{
			break;
		}
	}

	//powiekszanie space size zeby miec pewnosc ze elementy nie beda "wystwa³y", nie wplywa to na wydajnosc, a moze pomoc
	spaceSize.topLeft.y -= 10;
	spaceSize.topLeft.x -= 10;
	spaceSize.bottomRight.y += 10;
	spaceSize.bottomRight.x += 10;

	return spaceSize;
}

void createTree(QuadTree * mainTree, Layer const& layer){
	for(Rect const& rect : layer)
	{
		mainTree->insert(rect);
	}
}

int getIndex(REAL64_t intg[NSAMPLE + 1], double rand){
	for (int i = 0; i <= NSAMPLE; ++i){
		if (intg[i] <= rand && intg[i + 1] > rand) return i;
	}
}

Rect RandomWalk(Rect R, QuadTree* mainTree)
{
	REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE], intg[NSAMPLE + 1];
	UINT32_t Nsample = NSAMPLE;

	precompute_unit_square_green(g, dgdx, dgdy, intg, Nsample);//wyliczanie funkcji greena

	rng_init(1);//inicjalizacja genaeratora

	point p;
	double rand;
	int index;

	int counter = 0;

	Rect output, square = R.createGaussianSurface(1.1);

	do
	{
		rand = myrand() / (double)(MY_RAND_MAX);
		index = getIndex(intg, rand);
		p = square.getPointFromNindex(index, NSAMPLE);
		square = mainTree->drawBiggestSquareAtPoint(p);
		printf("%d\n",counter++);
	}
	while (mainTree->checkCollisons(p, output));

	//narazie pusty output
	return output;
}

void debugFunction()
{
	rng_init(1);
	for (int i = 0; i < 10; i++)
		std::cout << myrand() / (double)(MY_RAND_MAX) << std::endl;
}