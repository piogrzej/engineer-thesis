#pragma once

#include "quadTree.h"

#include <string>
#include <cstring>
#include <iostream>
#include <vector>
#define MAX_LINE_SIZE 50
#define LINE_HEADER "rect"

typedef std::vector<RectHost> Layer;
typedef std::vector<Layer> Layers;

class Parser
{
public:
	// if 0 , all layers will be loaded.
	Parser(char * fileName, char * delimiter, int layerNum = 0); 
	~Parser();

	RectHost			getLayerSize(int layerIt);
	Layer			getLayerAt(int i) { return layers[i]; }
	Layers			getAllLayers() { return layers; }

private:
	Layers			layers;

	RectHost			loadRectFromLine(char * linek);

};

