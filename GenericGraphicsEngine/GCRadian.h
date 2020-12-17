/*
 *  GCRadian.h
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/1/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */
#ifndef _GCRadian_h_
#define _GCRadian_h_

#include "Constants.h"
#include <iostream>

struct GCRadian;
struct GCDegree;

std::ostream& operator<<(std::ostream&, const GCRadian&);

struct GCRadian
{
	double radian;

	GCRadian():radian(0){}
	GCRadian(double r):radian(r){limit();}
	//Assignment operator
	GCRadian& operator=(const GCRadian&);
	GCRadian& operator=(const GCDegree&);
	//Copy constructor
	GCRadian(const GCRadian&);
	GCRadian(const GCDegree&);
	void limit(); //Radian is within range from 0 ~ <2PI
};


#endif