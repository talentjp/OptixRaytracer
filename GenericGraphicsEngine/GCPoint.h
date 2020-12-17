/*
 *  GCPoint.h
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/1/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#ifndef _GCPoint_h_
#define _GCPoint_h_

#include "GCVector.h"
#include <math.h>
#include <iostream>

struct GCPoint{
	double x;
	double y;
	double z;
	GCPoint(double p1, double p2, double p3) : x(p1), y(p2), z(p3){}
	GCPoint() : x(0), y(0), z(0) {}

	inline GCPoint operator+(const GCVector& rhs) const{return GCPoint(this->x + rhs.x, this->y + rhs.y, this->z + rhs.z);}
	inline double distance(const GCPoint& rhs) const{return sqrt( (this->x - rhs.x) * (this->x - rhs.x) + (this->y - rhs.y) * (this->y - rhs.y) + (this->z - rhs.z)*(this->z - rhs.z) );}
	inline GCVector vectorTo(const GCPoint& rhs) const{return GCVector(rhs.x - this->x, rhs.y - this->y, rhs.z - this->z);}
	inline GCVector vectorFrom(const GCPoint& rhs) const{return GCVector(this->x - rhs.x, this->y - rhs.y, this->z - rhs.z);}
};

std::ostream& operator<<(std::ostream&, const GCPoint&);

#endif