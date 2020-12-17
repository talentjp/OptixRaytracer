/*
 *  GCVector.h
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/1/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#ifndef _GCVector_h_
#define _GCVector_h_

#include "GCRadian.h"
#include <iostream>
#include <math.h>
#include <stdexcept>


struct GCVector;

std::ostream& operator<<(std::ostream&, const GCVector&);
GCVector operator*(double, const GCVector&);
GCVector operator*(const GCVector&, double);

struct GCVector{
	double x;
	double y;
	double z;
	
	GCVector(double p1, double p2, double p3) : x(p1), y(p2), z(p3){}
	GCVector():x(0),y(0),z(0){}
	inline GCVector operator+(const GCVector& rhs) const{return GCVector(this->x + rhs.x, this->y + rhs.y, this->z + rhs.z);}
	inline GCVector operator-(const GCVector& rhs) const{return GCVector(this->x - rhs.x, this->y - rhs.y, this->z - rhs.z);}
	inline GCVector operator-() const{return GCVector(-this->x, -this->y, -this->z);}
	inline double operator*(const GCVector& rhs) const{return this->x * rhs.x + this->y * rhs.y + this->z * rhs.z;}
	inline GCVector pieceWiseMultiply(const GCVector& rhs) const{return GCVector(this->x * rhs.x, this->y * rhs.y, this->z * rhs.z);}
	inline GCVector cross(const GCVector& rhs) const{return GCVector(this->y * rhs.z - this->z * rhs.y, this->z * rhs.x - this->x * rhs.z, this->x * rhs.y - this->y * rhs.x);}
	GCVector normalize();
	inline double length() const{return sqrt(this->x * this->x + this->y * this->y + this->z * this->z);}
	GCVector projectionOn(const GCVector& rhs) const;
	GCVector operator/(double scalar) const;
	GCRadian angleBetween(const GCVector& rhs) const;
	bool operator==(const GCVector& rhs) const
	{
		return x == rhs.x && y == rhs.y && z == rhs.z;
	}	
	bool operator!=(const GCVector& rhs) const
	{
		return !((*this) == rhs);
	}
};

#endif