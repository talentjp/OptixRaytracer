#include "LinearAlgebraTool.h"
#include "..\GenericGraphicsEngine\Constants.h"
#include "..\GenericGraphicsEngine\GCMatrix.h"

TySolution CLinearAlgebraTool::SolveEquation(TyEquation equationA, TyEquation equationB)
{
	double a00, a01, b0;
	double a10, a11, b1;
	//Unpack the coefficients
	std::tie(a00,a01,b0) = equationA;
	//std::cout<<"Equation A:"<<a00<<"X+"<<a01<<"Y="<<b0<<std::endl;
	std::tie(a10,a11,b1) = equationB;
	//std::cout<<"Equation B:"<<a10<<"X+"<<a11<<"Y="<<b1<<std::endl;
	double x,y;

	if(a00 == 0 && a01 != 0)
	{
		y = b0 / a01;
		if(a10 == 0)
		{
			return TySolution(false,0,0);
		}
		else
		{
			x =  (b1 - a11 * y) / a10;
		}
	}
	else if(a10 == 0 && a11 != 0)
	{
		y = b1 / a11;
		if(a00 == 0)
		{
			return TySolution(false,0,0);
		}
		else
		{
			x = (b0 - a01 * y) / a00;
		}
	}
	else if(a00 != 0 && a01 == 0)
	{
		x = b0 / a00;
		if(a11 == 0)
		{
			return TySolution(false,0,0);
		}
		else
		{
			y = (b1 - a10 * x) / a11;
		}
	}
	else if(a10 != 0 && a11 == 0)
	{
		x = b1 / a10;
		if(a01 == 0)
		{
			return TySolution(false,0,0);
		}
		else
		{
			y = (b0 - a00 * x) / a01;
		}
	}

	else
	{
		if((a01 * a10  -  a11 * a00) == 0){
			return TySolution(false,0,0);
		}
		//Solve for Y
		y = (b0 * a10  - b1 * a00) / (a01 * a10  -  a11 * a00);
		//Solve for X
		if(a00 == 0){
			return TySolution(false,0,0);
		}
		x = (b0 - a01 * y) / a00;
	}

	if(x < -EQUATION_EPSILON || y < -EQUATION_EPSILON || x + y > 1.0 + EQUATION_EPSILON)
	{
		return TySolution(false,0,0);
	}
	return TySolution(true, x, y);
}


GCMatrix CLinearAlgebraTool::LookAt(const GCPoint& eye, const GCPoint& center, const GCVector& up)
{
	GCVector w = eye.vectorFrom(center).normalize();
	GCVector u = up.cross(w).normalize();
	GCVector v = w.cross(u);
	GCMatrix retMat = GCMatrix(u.x, u.y, u.z, 0,
			                    v.x, v.y, v.z, 0,
								w.x, w.y, w.z, 0,
								0,     0,   0, 1) *
                        GCMatrix(1, 0, 0, -eye.x,
						        0, 1, 0, -eye.y,
								0, 0, 1, -eye.z,
								0, 0, 0,      1);
	return retMat;
}