#include "mainFunctions.h"
#include "ErrorHandler.h"

void pointsFormLine(point * topLeft, point * bottomRight, char * line)
{
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

    //powiekszanie space size zeby miec pewnosc ze elementy nie beda "wystwa�y", nie wplywa to na wydajnosc, a moze pomoc
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

int getDistanceRomTwoPoints(point p1, point p2)
{
    return (int)sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}

Rect RandomWalk(Rect R, QuadTree* mainTree)
{        
    ErrorHandler::getInstance() >> "Starting: " >> R >> "\n";

    REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE], intg[NSAMPLE + 1];
    UINT32_t Nsample = NSAMPLE;

    precompute_unit_square_green(g, dgdx, dgdy, intg, Nsample);//wyliczanie funkcji greena

    rng_init(3);//inicjalizacja genaeratora

    point p;
    double r;
    int index;
    bool isCollison;
    double adjustedFactor;
    Rect output, square = mainTree->creatGaussianSurfFrom(R, 1.5, adjustedFactor);
    ErrorHandler::getInstance() >> adjustedFactor >> "\n";

    bool broken = false;

    do
    {
        r = myrand() / (double)(MY_RAND_MAX);
        index = getIndex(intg, r);
        p = square.getPointFromNindex(index, NSAMPLE);
        ErrorHandler::getInstance() << p.x << "," << p.y << "\n";

        if(false == mainTree->isInBounds(p) || R.rectContains(p))
        {
            broken = SPECIAL_VALUE_BOOLEAN;
            SPECIAL_ACTION;
        }
        square = mainTree->drawBiggestSquareAtPoint(p);
        isCollison = mainTree->checkCollisons(p, output);
    }
    while (false == isCollison);

    if (false == broken)
        ErrorHandler::getInstance() >> "Ending: " >> output >> "\n";
    else
        ErrorHandler::getInstance() >> "Random walk is out of the bounds!" >> "\n";

    return output;
}

void printList(std::list<Rect> input){
    int i=0;
    for(std::list<Rect>::iterator iter = input.begin(); iter != input.end(); ++iter){
        i++;
        std::cout<<i<<" "<< iter->topLeft.x<<" "<<iter->topLeft.y<<" "<<iter->bottomRight.x<<" "<<iter->bottomRight.y<<std::endl;
     }
}