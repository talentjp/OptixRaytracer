/*
 *  GCWorldTransform.cpp
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/4/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */
#include "GCWorldTransform.h"

GCWorldTransform::GCWorldTransform()
{	
}

GCWorldTransform::~GCWorldTransform()
{	
}

void GCWorldTransform::rotate(const GCRadian& x, const GCRadian& y, const GCRadian& z)
{
	_transformMatrix = GCMatrix::createRotation(x, y, z) * _transformMatrix;
}
void GCWorldTransform::translate(double x, double y, double z)
{
	_transformMatrix = GCMatrix::createTranslation(x, y, z) * _transformMatrix;
}
void GCWorldTransform::scale(double scalar)
{
	_transformMatrix = GCMatrix::createScale(scalar, scalar, scalar) * _transformMatrix;
}


GCMatrix GCWorldTransform::getMatrix() const
{
	return _transformMatrix;
}