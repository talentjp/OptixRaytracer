/*
 *  GCViewTransform.h
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/4/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#ifndef _GCViewTransform_h_
#define _GCViewTransform_h_

#include "GCVector.h"
#include "GCMatrix.h"

class GCViewTransform
{
private:
	GCVector _eyePt;
	GCVector _rightVec;
	GCVector _upVec;
	GCVector _outVec;
	GCMatrix _transformMatrix;
	
public:
	void lookAt(const GCVector&, const GCVector&, const GCVector&);
	GCMatrix getMatrix() const;
};


#endif