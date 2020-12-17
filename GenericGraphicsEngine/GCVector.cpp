/*
 *  GCVector.cpp
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/1/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#include "GCVector.h"
#include <iostream>

std::ostream& operator<<(std::ostream& output, const GCVector& rhs)
{
	output<<"("<<rhs.x<<", "<<rhs.y<<", "<<rhs.z<<")";
	return output;
}

GCVector operator*(double scalar, const GCVector& vector)
{
	return GCVector(vector.x * scalar, vector.y * scalar, vector.z * scalar);
}
GCVector operator*(const GCVector& vector, double scalar)
{
	return GCVector(vector.x * scalar, vector.y * scalar, vector.z * scalar);
}

GCVector GCVector::normalize()
{
	double len = length();
	x /= len;
	y /= len;
	z /= len;
	return (*this);
}

GCVector GCVector::projectionOn(const GCVector& rhs) const
{
	GCVector thisVector = (*this);
	return thisVector * rhs * rhs / (rhs.length() * rhs.length());
}
GCVector GCVector::operator/(double scalar) const
{
	if(scalar != 0)
		return GCVector(this->x / scalar, this->y / scalar, this->z / scalar);
	else {
		throw std::runtime_error("Division by Zero");
	}
}
GCRadian GCVector::angleBetween(const GCVector& rhs) const
{
	GCRadian angle;
	angle.radian = acos( (*this) * rhs / (this->length() * rhs.length()) );
	return angle;
}