#ifndef _LinearAlgebraTool_h_
#define _LinearAlgebraTool_h_

#include <tuple>

typedef std::tuple<bool, double, double> TySolution;
typedef std::tuple<double, double, double> TyEquation;

struct GCPoint;
struct GCMatrix;
struct GCVector;

class CLinearAlgebraTool
{
public:
	static TySolution SolveEquation(TyEquation equationA, TyEquation equationB);
	static GCMatrix LookAt(const GCPoint& eye, const GCPoint& center, const GCVector& up);
};

#endif