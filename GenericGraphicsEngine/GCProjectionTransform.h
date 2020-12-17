/*
 *  GCProjectionTransform.h
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/4/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#ifndef _GCProjectionTransform_h_
#define _GCProjectionTransform_h_

#include "GCMatrix.h"
#include "GCDegree.h"
#include "GCRadian.h"

class GCProjectionTransform
{
private:
	GCMatrix _transformMatrix;
	double _nearPlane;
	double _farPlane;
	GCRadian _fieldOfView;
	double _aspectRatio;
	
public:
	GCProjectionTransform();
	GCProjectionTransform(double, double, double, const GCRadian&);
	void setAspectRatio(double);
	void setNearPlane(double);
	void setFarPlane(double);
	void setFieldOfView(const GCRadian&);
	void updateViewFrustum();
	GCMatrix getMatrix();
};


#endif