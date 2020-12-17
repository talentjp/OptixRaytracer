/*
 *  GCMatrix.h
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/2/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#ifndef _GCMatrix_h_
#define _GCMatrix_h_

#include "GCVector.h"
#include "GCPoint.h"
#include <assert.h>

struct GCRadian;
struct GCMatrix;
std::ostream& operator<<(std::ostream&, const GCMatrix&);
GCMatrix operator*(double, const GCMatrix&);
GCMatrix operator*(const GCMatrix&, double);
GCMatrix operator+(const GCMatrix&, const GCMatrix&);

struct GCMatrix
{
	double elements[4][4];
	int size;

	GCMatrix(double m11,double m12,double m13,double m14,
			 double m21,double m22,double m23,double m24,
			 double m31,double m32,double m33,double m34,
			 double m41,double m42,double m43,double m44)
	{
		size = 4;
		elements[0][0] = m11; elements[0][1] = m12; elements[0][2] = m13; elements[0][3] = m14;
		elements[1][0] = m21; elements[1][1] = m22; elements[1][2] = m23; elements[1][3] = m24;
		elements[2][0] = m31; elements[2][1] = m32; elements[2][2] = m33; elements[2][3] = m34;
		elements[3][0] = m41; elements[3][1] = m42; elements[3][2] = m43; elements[3][3] = m44;
	}

	GCMatrix(double m11,double m12,double m13,
			 double m21,double m22,double m23,
			 double m31,double m32,double m33)
	{
		size = 3;
		elements[0][0] = m11; elements[0][1] = m12; elements[0][2] = m13;
		elements[1][0] = m21; elements[1][1] = m22; elements[1][2] = m23;
		elements[2][0] = m31; elements[2][1] = m32; elements[2][2] = m33;
	}
	//mutable version
	inline double& operator()(int row, int col) {return elements[row - 1][col - 1];}
	//inmutable version
	inline double  operator()(int row, int col) const { return elements[row - 1][col - 1]; }
	
	//4x4 Identity Matrix by default
	GCMatrix(){InitializeMatrix(4);}
	GCMatrix(int s){InitializeMatrix(s);}
	void InitializeMatrix(int s);

	GCMatrix transpose() const;
	GCMatrix inverse() const;
	GCMatrix operator/(double divider);
	GCMatrix operator*(const GCMatrix& rhs) const;
	GCPoint  operator*(const GCPoint& rhs) const;
	GCVector operator*(const GCVector& rhs) const;

	static GCMatrix createRotation(const GCRadian&, const GCRadian&, const GCRadian&);
	static GCMatrix createRotation(const GCDegree&, const GCVector&);
	static GCMatrix createTranslation(double, double, double);
	static GCMatrix createScale(double, double, double);
};
							  
#endif

