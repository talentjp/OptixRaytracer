/*
 *  GCViewTransform.cpp
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/4/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#include "GCViewTransform.h"
#include "GCPoint.h"
#include "GCVector.h"
#include "GCMatrix.h"

void GCViewTransform::lookAt(const GCVector& eye, const GCVector& at, const GCVector& up)
{
	_eyePt = eye;
	//Calculates the view transformation matrix here
	_outVec = eye - at;
	_outVec = _outVec.normalize();
	_rightVec = up.cross(_outVec);
	_rightVec = _rightVec.normalize();
	//out and right vecs are both normalized, no need to normalize again
	_upVec = _outVec.cross(_rightVec);
	_transformMatrix.elements[0][0] = _rightVec.x;
	_transformMatrix.elements[0][1] = _rightVec.y;
	_transformMatrix.elements[0][2] = _rightVec.z;
	_transformMatrix.elements[0][3] =  -(_rightVec * eye);
	_transformMatrix.elements[1][0] = _upVec.x;
	_transformMatrix.elements[1][1] = _upVec.y;
	_transformMatrix.elements[1][2] = _upVec.z;
	_transformMatrix.elements[1][3] =  -(_upVec * eye);
	_transformMatrix.elements[2][0] = _outVec.x;
	_transformMatrix.elements[2][1] = _outVec.y;
	_transformMatrix.elements[2][2] = _outVec.z;
	_transformMatrix.elements[2][3] =  -(_outVec * eye);
	_transformMatrix.elements[3][0] = 0;
	_transformMatrix.elements[3][1] = 0;
	_transformMatrix.elements[3][2] = 0;
	_transformMatrix.elements[3][3] = 1;
}

GCMatrix GCViewTransform::getMatrix() const
{
	return _transformMatrix;
}
