#include "mainFunctions.h"

void pointsFormLine(point * topLeft, point * bottomRight, char * line){
	//zastepuje znaki ' ' bajtami 0 aby uzyc atoi
	//toeretycznie mozna to zrobic "na sztywno" i takie rozwiazanie byloby najsprawniejsze
	//lecz niewadomo czy struktura pliku wejsciowego sie nie zmieni
	int tab[4];//lokalizacje liczb
	int j = 0;
	int i = 0;
	while (line[i] != '\n'){
		if (line[i] == ' '){
			line[i] = 0;
			tab[j] = i + 1;
			j++;
		}
		i++;
	}
	topLeft->x = atoi((line + tab[0]));
	topLeft->y = atoi((line + tab[1]));
	bottomRight->x = atoi((line + tab[2]));
	bottomRight->y = atoi((line + tab[3]));
}

//okreslanie rozmiaru przestrzeni
rect layerSpaceSize(FILE * pFile){
	point tmpTopLeft, tmpBottomRight;
	rect spaceSize;
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
	while (1){
		fgets(linebuffor, LINE_BUFFOR_SIZE, pFile);
		if (linebuffor[0] == 'r'){
			pointsFormLine(&tmpTopLeft, &tmpBottomRight, linebuffor);
			if (!start){
				if (tmpTopLeft.x < spaceSize.top_left.x){
					spaceSize.top_left.x = tmpTopLeft.x;
				}
				if (tmpTopLeft.y < spaceSize.top_left.y){
					spaceSize.top_left.y = tmpTopLeft.y;
				}
				if (tmpBottomRight.x > spaceSize.bottom_right.x){
					spaceSize.bottom_right.x = tmpBottomRight.x;
				}
				if (tmpBottomRight.y > spaceSize.bottom_right.y){
					spaceSize.bottom_right.y = tmpBottomRight.y;
				}
			}
			else{
				spaceSize.top_left.x = tmpTopLeft.x;
				spaceSize.top_left.y = tmpTopLeft.y;
				spaceSize.bottom_right.x = tmpBottomRight.x;
				spaceSize.bottom_right.y = tmpBottomRight.y;
				start = false;
			}

		}
		else if (linebuffor[0] == '<'){
			break;
		}
	}

	//powiekszanie space size zeby miec pewnosc ze elementy nie beda "wystwa�y", nie wplywa to na wydajnosc, a moze pomoc
	spaceSize.top_left.y -= 10;
	spaceSize.top_left.x -= 10;
	spaceSize.bottom_right.y += 10;
	spaceSize.bottom_right.x += 10;

	return spaceSize;
}

void createTree(quadTree * mainTree, FILE * pFile){
	point tmpTopLeft, tmpBottomRight;
	char linebuffor[LINE_BUFFOR_SIZE];//buffor na linie

	while (1){//wyszukiwanie warstwy
		fgets(linebuffor, LINE_BUFFOR_SIZE, pFile);
		if (linebuffor[0] == '<'){
			if (strcmp(linebuffor, "<< metal3 >>\n") == 0){//narazie na sztywno, jedna warstwa
				break;
			}
		}
	}
	linebuffor[0] = 'A';// set first line buffor char to sth not equal to '<'

	//zmienna debug usunac
	int debug1 = 0;

	//fill tree
	rect tmprect;
	while (linebuffor[0] != '<'){
		fgets(linebuffor, LINE_BUFFOR_SIZE, pFile);
		if (linebuffor[0] == 'r'){
			pointsFormLine(&tmpTopLeft, &tmpBottomRight, linebuffor);
			tmprect.top_left = tmpTopLeft;
			tmprect.bottom_right = tmpBottomRight;
			mainTree->insert(tmprect);
			debug1++;
		}
	}
	printf("%d\n", debug1);
}