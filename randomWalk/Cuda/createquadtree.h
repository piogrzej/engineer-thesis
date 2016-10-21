/*
 * createquadtree.h
 *
 *  Created on: 21 paź 2016
 *      Author: mknap
 */

#ifndef CREATEQUADTREE_H_
#define CREATEQUADTREE_H_

#include "d_quadtree.h"
#include "d_vector.h"
#include <vector>

QuadTreeManager createQuadTree(const std::vector<RectHost>& layer,RectHost const& spaceSize,bool doCheck);


#endif /* CREATEQUADTREE_H_ */
