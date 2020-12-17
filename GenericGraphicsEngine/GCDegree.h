/*
 *  GCDegree.h
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/1/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#ifndef _GCDegree_h_
#define _GCDegree_h_

#include "Constants.h"
#include <iostream>

struct GCDegree;
struct GCRadian;

std::ostream& operator<<(std::ostream&, const GCDegree&);

struct GCDegree
{
	double degree;
	void limit();//Degree is within range from 0 ~ <360
	GCDegree():degree(0){}
	GCDegree(double d):degree(d){limit();}
	//Assignment operator
	GCDegree& operator=(const GCDegree&);
	GCDegree& operator=(const GCRadian&);
	//Copy constructor
	GCDegree(const GCDegree&);
	GCDegree(const GCRadian&);
	
	
};

#endif