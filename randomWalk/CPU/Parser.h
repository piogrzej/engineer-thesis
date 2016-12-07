#pragma once

#include "quadTree.h"

#include <string>
#include <cstring>
#include <iostream>
#include <vector>

typedef std::vector<RectHost> Layer;
typedef std::vector<Layer> Layers;

class Parser
{
public:
	Parser(char * delimiter);
	~Parser();
	     void       parse(std::string const&  fileName);
	RectHost		getLayerSize(int layerIt);
	Layer			getLayerAt(int i) { return layers[i]; }
	Layers			getAllLayers() { return layers; }
    int             getLayerCount() { return layers.size(); }

private:
	Layers			layers;
	char*           delimiter;

	RectHost	    loadRectFromLine(char* linek);

};

