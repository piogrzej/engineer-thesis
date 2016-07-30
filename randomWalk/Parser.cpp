#include "Parser.h"
#include "ErrorHandler.h"

#include <sstream>

Parser::Parser(char * fileName, char * delimiter, int layerNum)
{
	if (nullptr == fileName && nullptr == delimiter)
		ErrorHandler::getInstance() >> "Podaj sciezke i separator!";

	ifstream file(fileName, std::ios::in);
	if (file.is_open())
	{
		char line[MAX_LINE_SIZE];
		int layerIt = 0;
		Layer layer;
		bool isFirst = true;
		if (layerNum == 0)
			layerNum--; // w while nie ma znaczenia

		while (file.getline(line, MAX_LINE_SIZE) && layerNum)
		{
			if (nullptr != strstr(line, delimiter))
			{
				if (isFirst)
				{
					isFirst = false;
					continue;
				}
				layerNum--;
				layers.push_back(Layer(layer));
				layer.clear();
				continue;
			}
			if(nullptr != strstr(line, LINE_HEADER))
				layer.push_back(loadRectFromLine(line));
		}
	}
	else
		ErrorHandler::getInstance() >> "Nie mo�na otworzy� pliku!";
}

Parser::~Parser()
{
}

Rect Parser::getLayerSize(int layerIt)
{
    if (layerIt >= layers.size())
    {
        ErrorHandler::getInstance() >> "Nie ma takiej warstwy!\n";
        return Rect();
    }
	int leftX, rightX, topY, bottomY;
	Layer layer = layers[layerIt];
	bool start = true;
	for (const Rect& rect : layer)
	{
		if (start)
		{
			leftX = rect.topLeft.x;
			topY = rect.topLeft.y;
			rightX = rect.bottomRight.x;
			bottomY = rect.bottomRight.y;
			start = false;
			continue;
		}
		
		if (rect.topLeft.x < leftX)
			leftX = rect.topLeft.x;
		if (rect.topLeft.y < topY)
			topY = rect.topLeft.y;
		if (rect.bottomRight.x > rightX)
			rightX = rect.bottomRight.x;
		if (rect.bottomRight.y > bottomY)
			bottomY = rect.bottomRight.y;
	}
	return Rect(point(leftX - 10, topY - 10), point(rightX + 10, bottomY + 10));
}

Rect Parser::loadRectFromLine(char * line)
{
	if (nullptr == line)
		return Rect();

	Rect rect;
	std::string lineStr(line);
	std::stringstream stream(lineStr); // it is much more safe than atoi, thers exceptions
	int cord[4];
	int i = 0;
	std::string header;
	stream >> header;

	try
	{
		while (stream >> cord[i]) 
		{
			i++;
		}
	}
	catch (std::exception const & e)
	{
		ErrorHandler::getInstance() << e.what();
	}

	point topLeft, rightBottom;

	topLeft.x = cord[0];
	topLeft.y = cord[1];
	rightBottom.x = cord[2];
	rightBottom.y = cord[3];

	rect.bottomRight = rightBottom;
	rect.topLeft = topLeft;

	return rect;
}