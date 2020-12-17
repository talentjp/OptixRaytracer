#ifndef _Lights_h_
#define _Lights_h_

#include "Structs.h"
#include <algorithm>

//Pure interface
class ILight
{
public:
	virtual Ray GetRay(const GCPoint&) const = 0; //Returns the ray from the observing point
	virtual GCVector GetColor() const = 0;
	virtual double GetDistance(const GCPoint&) const = 0;
	virtual GCVector ComputeColor(const GCVector&, const GCVector&, const GCPoint&, const Material&) const = 0;
};


class CPointLite : public ILight
{
private:
	GCPoint  m_litePos;
	GCVector m_liteColor;
	double   m_attenuation[3];

public:
	CPointLite(const GCPoint& pos, const GCVector& color): m_litePos(pos), m_liteColor(color){m_attenuation[0] = 1.0; m_attenuation[1] = m_attenuation[2] = 0;}

	Ray GetRay(const GCPoint& pos) const {return Ray(pos, m_litePos.vectorFrom(pos).normalize());}

	GCVector GetColor() const {return m_liteColor;}

	double GetDistance(const GCPoint& pos) const {return m_litePos.vectorFrom(pos).length();}

	GCVector ComputeColor(const GCVector& rayDir, const GCVector& normal, 
		                  const GCPoint& pos, const Material& material) const;

	void SetAttenuation(double constant, double linear, double quadratic);

	GCPoint GetPosition() const {
		return m_litePos;
	}

	GCPoint GetAttenuation() const {
		return GCPoint{m_attenuation[0], m_attenuation[1], m_attenuation[2]};
	}
};

class CPolygonalLite : public ILight
{
private:
	GCPoint  m_corner;
	GCVector m_v1, m_v2;
	GCVector m_emission;

public:
	CPolygonalLite(const GCPoint& corner, const GCVector& v1, const GCVector& v2, const GCVector& emission) :
		m_corner(corner), m_v1(v1), m_v2(v2), m_emission(emission)
	{}

	//This is not quite correct as we are dealing with area light source
	Ray GetRay(const GCPoint& pos) const override { 
		return Ray(pos, m_corner.vectorFrom(pos).normalize()); 
	}

	GCVector GetColor() const override 
	{ 
		return m_emission; 
	}

	double GetDistance(const GCPoint& pos) const override 
	{ 
		return m_corner.vectorFrom(pos).length(); 
	}

	GCVector ComputeColor(const GCVector& rayDir, const GCVector& normal,
		const GCPoint& pos, const Material& material) const override {
		//TODO
		return GCVector();
	}

	GCPoint GetCorner() const {
		return m_corner;
	}

	GCVector GetVectorAB() const {
		return m_v1;
	}

	GCVector GetVectorAC() const {
		return m_v2;
	}
};

class CDirectionalLite : public ILight
{
private:
	GCVector m_liteDirection;
	GCVector m_liteColor;

public:
	CDirectionalLite(const GCVector& direction, const GCVector& color);

	Ray GetRay(const GCPoint& pos) const{return Ray(pos, m_liteDirection);}

	GCVector GetColor() const{return m_liteColor;}

	double GetDistance(const GCPoint& pos) const //{return std::numeric_limits<double>::max();} //Directional light has no distance property
	{
		return std::numeric_limits<double>::max();
		//return -1;
	} 

	GCVector ComputeColor(const GCVector& rayDir, const GCVector& normal, 
		                  const GCPoint& pos, const Material& material) const;
};



#endif // !_Lights_h_
