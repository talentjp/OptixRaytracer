/*
 *  GCProjectionTransform.cpp
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/4/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#include "GCProjectionTransform.h"


GCProjectionTransform::GCProjectionTransform()
{
	_aspectRatio = 16.0/9.0;
	_nearPlane = 5;
	_farPlane = 1000;
	_fieldOfView = GCDegree(20);
	updateViewFrustum();
}

GCProjectionTransform::GCProjectionTransform(double ratio, double near, double far, const GCRadian& field):
											_aspectRatio(ratio), _nearPlane(near), _farPlane(far), _fieldOfView(field)
{
	updateViewFrustum();
}

void GCProjectionTransform::setAspectRatio(double ratio)
{
	_aspectRatio = ratio;
}
void GCProjectionTransform::setNearPlane(double near)
{
	_nearPlane = near;
}
void GCProjectionTransform::setFarPlane(double far)
{
	_farPlane = far;
}
void GCProjectionTransform::setFieldOfView(const GCRadian& field)
{
	_fieldOfView = field;
}

void GCProjectionTransform::updateViewFrustum()
{
	double half_width = _nearPlane * sin(_fieldOfView.radian) / cos(_fieldOfView.radian);
	double half_height = half_width / _aspectRatio;
	//R = -1 ~ +1
	_transformMatrix.elements[0][0] = _nearPlane/half_width;
	//U = -1 ~ +1
	_transformMatrix.elements[1][1] = _nearPlane/half_height;
	//D = 0 ~ 1
	_transformMatrix.elements[2][2] = -_farPlane/(_farPlane - _nearPlane);
	_transformMatrix.elements[2][3] =  -_nearPlane * _farPlane / (_farPlane - _nearPlane);
	_transformMatrix.elements[3][2] = -1;
	_transformMatrix.elements[3][3] = 0;
}

GCMatrix GCProjectionTransform::getMatrix()
{
	return _transformMatrix;
}