/*
 *  GCRadian.cpp
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/1/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#include "GCRadian.h"
#include "GCDegree.h"

std::ostream& operator<<(std::ostream& output, const GCRadian& rhs)
{
	output<<rhs.radian<<" RAD";
	return output;
}

GCRadian& GCRadian::operator=(const GCRadian& rhs)
{
	radian = rhs.radian;
	return *this;
}

GCRadian& GCRadian::operator=(const GCDegree& rhs)
{
	radian = rhs.degree / 180 * PI;
	limit();
	return *this;
}

GCRadian::GCRadian(const GCRadian& rhs)
{
	radian = rhs.radian;
}

GCRadian::GCRadian(const GCDegree& rhs)
{
	radian = rhs.degree / 180 * PI;
	limit();
}

void GCRadian::limit()
{
	while(radian >= 2 * PI)
	{
		radian -= 2 * PI;
	}
	while(radian < 0)
	{
		radian += 2 * PI;
	}
}
