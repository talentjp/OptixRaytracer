#include "Shapes.h"
#include "LinearAlgebraTool.h"
#include <algorithm>


void CTriangle::ApplyTransform(const GCMatrix& mat)
{
	m_pointA = mat * m_pointA;
	m_pointB = mat * m_pointB;
	m_pointC = mat * m_pointC;
}

void CTriangle::SetMaterialCache(std::vector<Material>* cache)
{
	m_materialCache = cache;
}

Material CTriangle::GetMaterial() const {
	if (m_materialIndex == -1 || !m_materialCache)
	{
		return Material();
	}

	return (*m_materialCache)[m_materialIndex];
}

void CTriangle::SetMaterial(const Material& m) {
	for (int i = 0; i < (*m_materialCache).size(); ++i)
	{
		if ((*m_materialCache)[i] == m)
		{
			m_materialIndex = i;
			return;
		}
	}
	//Not found, need to create a new material in the cache
	m_materialCache->push_back(m);
	m_materialIndex = m_materialCache->size() - 1;
}

std::tuple<GCPoint, GCPoint, GCPoint> CTriangle::GetVertices() {
	return std::make_tuple(m_pointA, m_pointB, m_pointC);
}

IntersectInfo CTriangle::IntersectWith(const Ray& ray, bool backFaceCulling)
{
	IntersectInfo info = IntersectWith_impl_new(ray, backFaceCulling);
	return info;
}

//to prevent colliding with the same object
const double CULL_MIN_DISTANCE = 0.00001;

IntersectInfo CTriangle::IntersectWith_impl_old(const Ray& ray)
{
	IntersectInfo info;
	info.ifIntersect = false;

	GCVector vector1 = m_pointB.vectorFrom(m_pointA);
	GCVector vector2 = m_pointC.vectorFrom(m_pointA);
	info.normal = vector1.cross(vector2).normalize();
	//the triangle is back-facing
	if (ray.direction * info.normal >= 0) {
		return info;
	}
	//depth is distance from the ray origin to the plane the triangle is on
	info.depth = (m_pointA.vectorFrom(ray.origin) * info.normal) / (ray.direction * info.normal);
	if (info.depth <= CULL_MIN_DISTANCE) {
		return info;
	}

	info.intersectPt = ray.origin + ray.direction * info.depth;

	TySolution solutionA = CLinearAlgebraTool::SolveEquation(TyEquation(vector1.x, vector2.x, info.intersectPt.vectorFrom(m_pointA).x),
		TyEquation(vector1.y, vector2.y, info.intersectPt.vectorFrom(m_pointA).y));
	TySolution solutionB = CLinearAlgebraTool::SolveEquation(TyEquation(vector1.y, vector2.y, info.intersectPt.vectorFrom(m_pointA).y),
		TyEquation(vector1.z, vector2.z, info.intersectPt.vectorFrom(m_pointA).z));
	TySolution solutionC = CLinearAlgebraTool::SolveEquation(TyEquation(vector1.z, vector2.z, info.intersectPt.vectorFrom(m_pointA).z),
		TyEquation(vector1.x, vector2.x, info.intersectPt.vectorFrom(m_pointA).x));

	//If we got any solution from the 3 formulas then there is an intersection
	if (std::get<0>(solutionA) || std::get<0>(solutionB) || std::get<0>(solutionC)) {
		info.ifIntersect = true;
	}
	return info;
}

IntersectInfo CTriangle::IntersectWith_impl_new(const Ray& ray, bool backFaceCulling)
{
	IntersectInfo info;
	info.ifIntersect = false;

	////beginning of the new algorithm
	auto v0v1 = m_pointB.vectorFrom(m_pointA);
	auto v0v2 = m_pointC.vectorFrom(m_pointA);
	auto pvec = ray.direction.cross(v0v2);
	auto det = v0v1 * pvec;
	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle	
	if (backFaceCulling && det < 0.000001) return info;

	// ray and triangle are parallel if det is close to 0
	if (fabs(det) < 0.000001) return info;

	auto invDet = 1 / det;
	auto tvec = ray.origin.vectorFrom(m_pointA);
	auto u = (tvec * pvec) * invDet;
	if (u < 0 || u > 1) return info;

	auto qvec = tvec.cross(v0v1);
	auto v = (ray.direction * qvec) * invDet;
	if (v < 0 || u + v > 1) return info;

	info.depth = (v0v2 * qvec) * invDet;
	if (info.depth <= CULL_MIN_DISTANCE)
	{
		return info;
	}
	info.ifIntersect = true;
	info.intersectPt = ray.origin + ray.direction * info.depth;
	info.normal = v0v1.cross(v0v2).normalize();
	return info;
}

IntersectInfo CSphere::IntersectWith(const Ray& ray, bool backFaceCulling)
{
	IntersectInfo info;
	info.ifIntersect = false;
	GCPoint newOrigin = m_transformMat.inverse() * ray.origin;
	GCVector newDirection = m_transformMat.inverse() * ray.direction;
	newDirection.normalize();
	Ray newRay(newOrigin, newDirection);
	//v = origin of ray - center of sphere
	GCVector v = newRay.origin.vectorFrom(m_centerPt);
	//The ray origin is inside the sphere
	if (backFaceCulling && v * v <= m_radius * m_radius)
	{
		return info;
	}
	//solution has to be real number : sqrt(b^2 - 4ac) > 0
	double a = newRay.direction * newRay.direction;
	double b = 2.0 * (v * newRay.direction);
	double c = (v * v) - m_radius * m_radius;
	if( b * b >= 4.0 * a * c)
	{
		double t1 = (-b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
		double t2 = (-b - sqrt(b * b - 4.0 * a * c)) / (2.0 * a);

		if (t2 < 0)
		{
			t2 = t1;
			if (t2 < 0)
			{
				return info;
			}
		}

		if(t2 > CULL_MIN_DISTANCE)
		{
			info.ifIntersect = true;
			info.intersectPt = newRay.origin + newRay.direction * t2;
			info.normal = info.intersectPt.vectorFrom(m_centerPt).normalize();
			//Have to transform the inverted point and normal using the same matrix
			info.intersectPt = m_transformMat * info.intersectPt;
			info.normal = m_transformMat.inverse().transpose() * info.normal;
			info.normal.normalize();
			info.depth = ray.origin.vectorTo(info.intersectPt).length();
		}
	}
	return info;
}

Material CSphere::GetMaterial() const
{
	////same as triangle
	//if (m_materialIndex == -1 || !m_materialCache)
	//{
	//	return Material();
	//}

	//return (*m_materialCache)[m_materialIndex];
	return m_material;
}

void CSphere::SetMaterial(const Material& m)
{
	////same as triangle
	//for (int i = 0; i < (*m_materialCache).size(); ++i)
	//{
	//	if ((*m_materialCache)[i] == m)
	//	{
	//		m_materialIndex = i;
	//		return;
	//	}
	//}
	////Not found, need to create a new material in the cache
	//m_materialCache->push_back(m);
	//m_materialIndex = m_materialCache->size() - 1;
	m_material = m;
}

void CSphere::SetMaterialCache(std::vector<Material>* cache)
{
	//same as triangle
	//m_materialCache = cache;
}
