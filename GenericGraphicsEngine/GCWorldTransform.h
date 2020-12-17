/*
 *  GCWorldTransform.h
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/4/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#ifndef _GCWorldTransform_h_
#define _GCWorldTransform_h_

#include "GCMatrix.h"

class GCWorldTransform
{
private:
	GCMatrix _transformMatrix;
	
public:
	GCWorldTransform();
	~GCWorldTransform();
	void rotate(const GCRadian&, const GCRadian&, const GCRadian&);
	void translate(double, double, double);
	void scale(double);
	GCMatrix getMatrix() const;
};

#endif