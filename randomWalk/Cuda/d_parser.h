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
	d_Parser(char * delimiter);
	~d_Parser();
	    void        parse(std::string const& fileName);
	d_Rect			getLayerSize(int layerIt);
	d_Layer			getLayerAt(int i) { return layers[i]; }
	d_Layers		getAllLayers() { return layers; }
	    int         getLayerCount() { return layers.size(); }

private:
	d_Layers		layers;
	char*           delimiter;
	d_Rect			loadRectFromLine(char*  linek);

};

