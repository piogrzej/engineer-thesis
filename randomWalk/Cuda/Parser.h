#pragma once

#include <string>
#include <cstring>
#include <iostream>
#include <vector>

#include "d_Rect.h"

#define MAX_LINE_SIZE 50
#define LINE_HEADER "rect"
#define BOUNDS_MUL_FACTOR 0.01

typedef std::vector<d_Rect> Layer;
typedef std::vector<Layer> Layers;

class Parser
{
public:
	// if 0 , all layers will be loaded.
	Parser(char * fileName, char * delimiter, int layerNum = 0); 
	~Parser();

	d_Rect			getLayerSize(int layerIt);
	Layer			getLayerAt(int i) { return layers[i]; }
	Layers			getAllLayers() { return layers; }

private:
	Layers			layers;

	d_Rect			loadRectFromLine(char * linek);

};

