/*
 *  GCMatrix.cpp
 *  GenericGraphicsEngine
 *
 *  Created by Talent on 11/2/10.
 *  Copyright 2010 NYU. All rights reserved.
 *
 */

#include "GCMatrix.h"
#include <stdexcept>
#include <math.h>

std::ostream& operator<<(std::ostream& output, const GCMatrix& rhs)
{
	output<<"Matrix:{";
	for(int row = 0 ; row < rhs.size; row++)
	{
		output<<"[";
		for(int col = 0 ; col < rhs.size; col++)
		{
			output<<rhs.elements[row][col];
			if(col != rhs.size - 1) output<<", ";
		}
		output<<"]";
		if(row != rhs.size - 1) output<<"\n       ,";
		else output<<"}\n";
	}
	return output;
}


GCMatrix operator*(double scalar, const GCMatrix& rhs)
{
	GCMatrix newMat(rhs.size);
	for(int i = 0; i < rhs.size; i++){
		for(int j = 0; j < rhs.size; j++){
			newMat.elements[i][j] = rhs.elements[i][j] * scalar;
		}
	}
	return newMat;
}

GCMatrix operator*(const GCMatrix& lhs, double scalar)
{
	GCMatrix newMat(lhs.size);
	for(int i = 0; i < lhs.size; i++){
		for(int j = 0; j < lhs.size; j++){
			newMat.elements[i][j] = lhs.elements[i][j] * scalar;
		}
	}
	return newMat;
}

GCMatrix operator+(const GCMatrix& lhs, const GCMatrix& rhs)
{
	GCMatrix newMat(lhs.size);
	if(lhs.size == rhs.size)
	{
		for(int i = 0; i < lhs.size; i++){
			for(int j = 0; j < lhs.size; j++){
				newMat.elements[i][j] = lhs.elements[i][j] + rhs.elements[i][j];
			}
		}
	}
	else
	{
		std::cout<<"ERROR : Matrices have different sizes and cannot be added together!"<<std::endl;
		assert(0);
	}
	return newMat;
}

void GCMatrix::InitializeMatrix(int s)
{
	size = s;
	for(int i = 0; i < size; i++)
		for(int j = 0; j < size; j++)
			elements[i][j] = 0;
	for(int i = 0; i < size; i++)
		elements[i][i] = 1;
}

GCMatrix GCMatrix::transpose() const
{
	GCMatrix newMat(this->size);
	for(int i = 0 ; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			newMat.elements[j][i] = this->elements[i][j];
		}
	}
	return newMat;
}

GCMatrix GCMatrix::operator/(double divider)
{
	GCMatrix newMat(size);
	for(int row = 0; row < size; row++){
		for(int col = 0 ; col < size; col++){
			newMat.elements[row][col] = elements[row][col] / divider;
		}
	}
	return newMat;
}

GCMatrix GCMatrix::operator*(const GCMatrix& rhs) const
{
	GCMatrix newMat(size);
	if(size == rhs.size)
	{
		for(int row = 0; row < size; row++){
			for(int col = 0 ; col < size; col++){
				double sum = 0;
				for(int i = 0; i < size; i++){
					sum += elements[row][i] * rhs.elements[i][col];
				}
				newMat.elements[row][col] = sum;
			}
		}
		return newMat;
	}
	else
	{
		std::cout<<"ERROR: Matrices are of different sizes and cannot not get multiplied together!"<<std::endl;
		assert(0);
	}
	return newMat;
}

GCPoint GCMatrix::operator*(const GCPoint& rhs) const
{
	//Software matrix-vector multiplication
	GCPoint newPt;
	if(size == 4)
	{
		double w =  elements[3][0] * rhs.x + elements[3][1] * rhs.y + elements[3][2] * rhs.z + elements[3][3];
		if(w != 0)
		{
			newPt.x = (elements[0][0] * rhs.x + elements[0][1] * rhs.y + elements[0][2] * rhs.z + elements[0][3])/w;
			newPt.y = (elements[1][0] * rhs.x + elements[1][1] * rhs.y + elements[1][2] * rhs.z + elements[1][3])/w;
			newPt.z = (elements[2][0] * rhs.x + elements[2][1] * rhs.y + elements[2][2] * rhs.z + elements[2][3])/w;
		}
		else {
			newPt.z = newPt.y = newPt.x = std::numeric_limits<double>::infinity();
		}
	}
	else
	{
		std::cout<<"ERROR: Matrix size is less than 4 and cannot be applied on a Point!"<<std::endl;
		assert(0);
	}
	return newPt;
}

GCVector GCMatrix::operator*(const GCVector& rhs) const
{
	GCVector newVec;
	if(size >= 3)
	{
		newVec.x = elements[0][0] * rhs.x + elements[0][1] * rhs.y + elements[0][2] * rhs.z;
		newVec.y = elements[1][0] * rhs.x + elements[1][1] * rhs.y + elements[1][2] * rhs.z;
		newVec.z = elements[2][0] * rhs.x + elements[2][1] * rhs.y + elements[2][2] * rhs.z;
	}
	else
	{
		std::cout<<"ERROR: Matrix size is less than 3 and cannot be applied on a Vector!"<<std::endl;
		assert(0);
	}
	return newVec;
}

GCMatrix GCMatrix::createRotation(const GCRadian& x, const GCRadian& y, const GCRadian& z)
{
	GCMatrix rotMatX;
	rotMatX.elements[1][1] =  cos(x.radian);
	rotMatX.elements[1][2] = -sin(x.radian);
	rotMatX.elements[2][1] =  sin(x.radian);
	rotMatX.elements[2][2] =  cos(x.radian);
	//Inverted to conform to the handedness
	GCMatrix rotMatY;
	rotMatY.elements[0][0] =  cos(-y.radian);
	rotMatY.elements[0][2] =  sin(-y.radian);
	rotMatY.elements[2][0] = -sin(-y.radian);
	rotMatY.elements[2][2] =  cos(-y.radian);
	GCMatrix rotMatZ;
	rotMatZ.elements[0][0] =  cos(z.radian);
	rotMatZ.elements[0][1] = -sin(z.radian);
	rotMatZ.elements[1][0] =  sin(z.radian);
	rotMatZ.elements[1][1] =  cos(z.radian);
	return rotMatX * rotMatY * rotMatZ;
}

GCMatrix GCMatrix::createRotation(const GCDegree& theta_degree, const GCVector& axis)
{
	//Rodrigues' rotation formula
    GCVector normalized_rotation_axis = axis;
	normalized_rotation_axis.normalize();
	GCRadian theta_radian(theta_degree);

	GCMatrix newMat3 = cos(theta_radian.radian) * GCMatrix(1,0,0,0,1,0,0,0,1) +
		(1.0 - cos(theta_radian.radian)) * 
		GCMatrix(normalized_rotation_axis.x * normalized_rotation_axis.x, normalized_rotation_axis.x * normalized_rotation_axis.y, normalized_rotation_axis.x * normalized_rotation_axis.z,
		         normalized_rotation_axis.x * normalized_rotation_axis.y, normalized_rotation_axis.y * normalized_rotation_axis.y, normalized_rotation_axis.y * normalized_rotation_axis.z,
				 normalized_rotation_axis.x * normalized_rotation_axis.z, normalized_rotation_axis.y * normalized_rotation_axis.z, normalized_rotation_axis.z * normalized_rotation_axis.z) +
				 sin(theta_radian.radian) * GCMatrix(0, -normalized_rotation_axis.z, normalized_rotation_axis.y,
				                                     normalized_rotation_axis.z, 0, -normalized_rotation_axis.x,
				                                     -normalized_rotation_axis.y, normalized_rotation_axis.x, 0);

	GCMatrix newMat4(newMat3.elements[0][0], newMat3.elements[0][1], newMat3.elements[0][2], 0,
		             newMat3.elements[1][0], newMat3.elements[1][1], newMat3.elements[1][2], 0,
					 newMat3.elements[2][0], newMat3.elements[2][1], newMat3.elements[2][2], 0,
					 0,0,0,1);

	return newMat4;
}

GCMatrix GCMatrix::createTranslation(double x, double y, double z)
{
	GCMatrix newMat;
	newMat.elements[0][3] = x;
	newMat.elements[1][3] = y;
	newMat.elements[2][3] = z;
	return newMat;
}

GCMatrix GCMatrix::createScale(double x, double y, double z)
{
	GCMatrix newMat;
	newMat.elements[0][0] = x;
	newMat.elements[1][1] = y;
	newMat.elements[2][2] = z;
	return newMat;
}

GCMatrix GCMatrix::inverse() const
{
	const GCMatrix& mat = (*this);
	double det =   mat(1,1)*mat(2,2)*mat(3,3)*mat(4,4) + mat(1,1)*mat(2,3)*mat(3,4)*mat(4,2) + mat(1,1)*mat(2,4)*mat(3,2)*mat(4,3) 
		         + mat(1,2)*mat(2,1)*mat(3,4)*mat(4,3) + mat(1,2)*mat(2,3)*mat(3,1)*mat(4,4) + mat(1,2)*mat(2,4)*mat(3,3)*mat(4,1)
				 + mat(1,3)*mat(2,1)*mat(3,2)*mat(4,4) + mat(1,3)*mat(2,2)*mat(3,4)*mat(4,1) + mat(1,3)*mat(2,4)*mat(3,1)*mat(4,2)
				 + mat(1,4)*mat(2,1)*mat(3,3)*mat(4,2) + mat(1,4)*mat(2,2)*mat(3,1)*mat(4,3) + mat(1,4)*mat(2,3)*mat(3,2)*mat(4,1) 
				 - mat(1,1)*mat(2,2)*mat(3,4)*mat(4,3) - mat(1,1)*mat(2,3)*mat(3,2)*mat(4,4) - mat(1,1)*mat(2,4)*mat(3,3)*mat(4,2) 
				 - mat(1,2)*mat(2,1)*mat(3,3)*mat(4,4) - mat(1,2)*mat(2,3)*mat(3,4)*mat(4,1) - mat(1,2)*mat(2,4)*mat(3,1)*mat(4,3) 
				 - mat(1,3)*mat(2,1)*mat(3,4)*mat(4,2) - mat(1,3)*mat(2,2)*mat(3,1)*mat(4,4) - mat(1,3)*mat(2,4)*mat(3,2)*mat(4,1) 
				 - mat(1,4)*mat(2,1)*mat(3,2)*mat(4,3) - mat(1,4)*mat(2,2)*mat(3,3)*mat(4,1) - mat(1,4)*mat(2,3)*mat(3,1)*mat(4,2);

	if(det == 0)
	{
		std::cout<<"Det of matrix is 0, there is no inverse matrix!"<<std::endl;
		assert(0);
	}

	double b11 = mat(2,2)*mat(3,3)*mat(4,4) + mat(2,3)*mat(3,4)*mat(4,2) + mat(2,4)*mat(3,2)*mat(4,3) 
		         - mat(2,2)*mat(3,4)*mat(4,3) - mat(2,3)*mat(3,2)*mat(4,4) - mat(2,4)*mat(3,3)*mat(4,2);
	double b12 = mat(1,2)*mat(3,4)*mat(4,3) + mat(1,3)*mat(3,2)*mat(4,4) + mat(1,4)*mat(3,3)*mat(4,2) 
		         - mat(1,2)*mat(3,3)*mat(4,4) - mat(1,3)*mat(3,4)*mat(4,2) - mat(1,4)*mat(3,2)*mat(4,3);
	double b13 = mat(1,2)*mat(2,3)*mat(4,4) + mat(1,3)*mat(2,4)*mat(4,2) + mat(1,4)*mat(2,2)*mat(4,3) 
		         - mat(1,2)*mat(2,4)*mat(4,3) - mat(1,3)*mat(2,2)*mat(4,4) - mat(1,4)*mat(2,3)*mat(4,2);
	double b14 = mat(1,2)*mat(2,4)*mat(3,3) + mat(1,3)*mat(2,2)*mat(3,4) + mat(1,4)*mat(2,3)*mat(3,2) 
		         - mat(1,2)*mat(2,3)*mat(3,4) - mat(1,3)*mat(2,4)*mat(3,2) - mat(1,4)*mat(2,2)*mat(3,3);
	double b21 = mat(2,1)*mat(3,4)*mat(4,3) + mat(2,3)*mat(3,1)*mat(4,4) + mat(2,4)*mat(3,3)*mat(4,1) 
		         - mat(2,1)*mat(3,3)*mat(4,4) - mat(2,3)*mat(3,4)*mat(4,1) - mat(2,4)*mat(3,1)*mat(4,3);
	double b22 = mat(1,1)*mat(3,3)*mat(4,4) + mat(1,3)*mat(3,4)*mat(4,1) + mat(1,4)*mat(3,1)*mat(4,3) 
		         - mat(1,1)*mat(3,4)*mat(4,3) - mat(1,3)*mat(3,1)*mat(4,4) - mat(1,4)*mat(3,3)*mat(4,1);
	double b23 = mat(1,1)*mat(2,4)*mat(4,3) + mat(1,3)*mat(2,1)*mat(4,4) + mat(1,4)*mat(2,3)*mat(4,1) 
		         - mat(1,1)*mat(2,3)*mat(4,4) - mat(1,3)*mat(2,4)*mat(4,1) - mat(1,4)*mat(2,1)*mat(4,3);
	double b24 = mat(1,1)*mat(2,3)*mat(3,4) + mat(1,3)*mat(2,4)*mat(3,1) + mat(1,4)*mat(2,1)*mat(3,3) 
		         - mat(1,1)*mat(2,4)*mat(3,3) - mat(1,3)*mat(2,1)*mat(3,4) - mat(1,4)*mat(2,3)*mat(3,1);
	double b31 = mat(2,1)*mat(3,2)*mat(4,4) + mat(2,2)*mat(3,4)*mat(4,1) + mat(2,4)*mat(3,1)*mat(4,2) 
		         - mat(2,1)*mat(3,4)*mat(4,2) - mat(2,2)*mat(3,1)*mat(4,4) - mat(2,4)*mat(3,2)*mat(4,1);
	double b32 = mat(1,1)*mat(3,4)*mat(4,2) + mat(1,2)*mat(3,1)*mat(4,4) + mat(1,4)*mat(3,2)*mat(4,1) 
		         - mat(1,1)*mat(3,2)*mat(4,4) - mat(1,2)*mat(3,4)*mat(4,1) - mat(1,4)*mat(3,1)*mat(4,2);
	double b33 = mat(1,1)*mat(2,2)*mat(4,4) + mat(1,2)*mat(2,4)*mat(4,1) + mat(1,4)*mat(2,1)*mat(4,2) 
		         - mat(1,1)*mat(2,4)*mat(4,2) - mat(1,2)*mat(2,1)*mat(4,4) - mat(1,4)*mat(2,2)*mat(4,1);
	double b34 = mat(1,1)*mat(2,4)*mat(3,2) + mat(1,2)*mat(2,1)*mat(3,4) + mat(1,4)*mat(2,2)*mat(3,1) 
		         - mat(1,1)*mat(2,2)*mat(3,4) - mat(1,2)*mat(2,4)*mat(3,1) - mat(1,4)*mat(2,1)*mat(3,2);
	double b41 = mat(2,1)*mat(3,3)*mat(4,2) + mat(2,2)*mat(3,1)*mat(4,3) + mat(2,3)*mat(3,2)*mat(4,1) 
		         - mat(2,1)*mat(3,2)*mat(4,3) - mat(2,2)*mat(3,3)*mat(4,1) - mat(2,3)*mat(3,1)*mat(4,2);
	double b42 = mat(1,1)*mat(3,2)*mat(4,3) + mat(1,2)*mat(3,3)*mat(4,1) + mat(1,3)*mat(3,1)*mat(4,2) 
		         - mat(1,1)*mat(3,3)*mat(4,2) - mat(1,2)*mat(3,1)*mat(4,3) - mat(1,3)*mat(3,2)*mat(4,1);
	double b43 = mat(1,1)*mat(2,3)*mat(4,2) + mat(1,2)*mat(2,1)*mat(4,3) + mat(1,3)*mat(2,2)*mat(4,1) 
		         - mat(1,1)*mat(2,2)*mat(4,3) - mat(1,2)*mat(2,3)*mat(4,1) - mat(1,3)*mat(2,1)*mat(4,2);
	double b44 = mat(1,1)*mat(2,2)*mat(3,3) + mat(1,2)*mat(2,3)*mat(3,1) + mat(1,3)*mat(2,1)*mat(3,2) 
		         - mat(1,1)*mat(2,3)*mat(3,2) - mat(1,2)*mat(2,1)*mat(3,3) - mat(1,3)*mat(2,2)*mat(3,1);

	return GCMatrix(b11,b12,b13,b14,b21,b22,b23,b24,b31,b32,b33,b34,b41,b42,b43,b44) / det;
}