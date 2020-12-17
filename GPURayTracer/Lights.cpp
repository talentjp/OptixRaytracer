#include "Lights.h"

GCVector CPointLite::ComputeColor(const GCVector& rayDir, const GCVector& normal, 
		                const GCPoint& pos, const Material& material) const {
	double dist = m_litePos.vectorFrom(pos).length();
	GCVector eyeVec = rayDir;
	eyeVec = -eyeVec.normalize();	
	GCVector lightVec = m_litePos.vectorFrom(pos) / dist;	
	GCVector halfVec = (eyeVec + lightVec).normalize();
	//Real light, with attenuation
	GCVector color = ( 1.0 /  (m_attenuation[0] + m_attenuation[1] * dist + m_attenuation[2]*(dist * dist)) ) * 
		             m_liteColor.pieceWiseMultiply(material.diffuse * std::max(0.0, normal * lightVec) + material.specular * pow(std::max(0.0, normal * halfVec), material.shininess));
	return color;
}


void CPointLite::SetAttenuation(double constant, double linear, double quadratic)
{
	m_attenuation[0] = constant;
	m_attenuation[1] = linear;
	m_attenuation[2] = quadratic;
}


CDirectionalLite::CDirectionalLite(const GCVector& direction, const GCVector& color)
{
	m_liteDirection = direction;
	m_liteDirection.normalize();
	m_liteColor = color;
}

GCVector CDirectionalLite::ComputeColor(const GCVector& rayDir, const GCVector& normal, 
		                const GCPoint& pos, const Material& material) const {
	GCVector eyeVec = rayDir;
	eyeVec = -eyeVec.normalize();
	//directional light direction is direction to the light source (weird convention)
	GCVector lightVec = m_liteDirection;
	GCVector halfVec = (eyeVec + lightVec).normalize();
	GCVector color = m_liteColor.pieceWiseMultiply(material.diffuse * std::max(0.0, normal * lightVec) + material.specular * pow(std::max(0.0, normal * halfVec), material.shininess));
	return color;
}