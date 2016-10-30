#pragma once

#include <string>
#include <cstring>
#include <iostream>
#include <vector>

#include "d_Rect.h"

#define MAX_LINE_SIZE 50
#define LINE_HEADER "rect"
#define BOUNDS_MUL_FACTOR 0.01

typedef std::vector<d_Rect> d_Layer;
typedef std::vector<d_Layer> d_Layers;

class d_Parser
{
public:
	// if 0 , all layers will be loaded.
	d_Parser(char * fileName, char * delimiter, int layerNum = 0);
	~d_Parser();

	d_Rect			getLayerSize(int layerIt);
	d_Layer			getLayerAt(int i) { return layers[i]; }
	d_Layers		getAllLayers() { return layers; }

private:
	d_Layers		layers;

	d_Rect			loadRectFromLine(char * linek);

};

