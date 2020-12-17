/*
 *  GCPoint.cpp
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/1/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#include "GCPoint.h"
#include <iostream>

std::ostream& operator<<(std::ostream& output, const GCPoint& rhs)
{
	output<<"["<<rhs.x<<", "<<rhs.y<<", "<<rhs.z<<"]";
	return output;
}