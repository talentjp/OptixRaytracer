/*
 *  GCDegree.cpp
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/1/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#include "GCDegree.h"
#include "GCRadian.h"

std::ostream& operator<<(std::ostream& output, const GCDegree& rhs)
{
	output<<rhs.degree<<" DEGREES";
	return output;
}

GCDegree& GCDegree::operator=(const GCDegree& rhs)
{
	degree = rhs.degree;
	return *this;
}

GCDegree& GCDegree::operator=(const GCRadian& rhs)
{
	degree = rhs.radian / PI * 180;
	limit();
	return *this;
}

GCDegree::GCDegree(const GCDegree& rhs)
{
	degree = rhs.degree;
}

GCDegree::GCDegree(const GCRadian& rhs)
{
	degree = rhs.radian / PI * 180;
	limit();
}

void GCDegree::limit()
{
	while(degree >= 360)
	{
		degree -= 360;
	}
	while(degree < 0)
	{
		degree += 360;
	}
}


