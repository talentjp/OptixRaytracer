#ifndef _Structs_h_
#define _Structs_h_

#include "enumdefs.h"
#include "..\GenericGraphicsEngine\GCPoint.h"
#include "..\GenericGraphicsEngine\GCVector.h"

struct Ray
{
	GCPoint  origin;
	GCVector direction;
	Ray(){}
	Ray(const GCPoint& pt, const GCVector& dir):origin(pt), direction(dir){}
};

struct Material
{
	GCVector emission;
	GCVector ambient;
	GCVector specular;
	GCVector diffuse;
	double   shininess = 10.0;
	double   roughness = 0.25;
	BRDFType brdfType = BRDFType::Phong;
	int      lightNum = -1;

	bool operator==(const Material& other) const
	{
		return emission == other.emission && ambient == other.ambient && 
			   specular == other.specular && diffuse == other.diffuse && 
			   shininess == other.shininess && lightNum == other.lightNum &&
			   roughness == other.roughness && brdfType == other.brdfType;
	}
};

struct IntersectInfo
{
	bool     ifIntersect;
	GCPoint  intersectPt;
	GCVector normal;
	double   depth;
};


#endif // !_Structs_h_
