#ifndef _Shapes_h_
#define _Shapes_h_

#include "..\GenericGraphicsEngine\GCVector.h"
#include "..\GenericGraphicsEngine\GCPoint.h"
#include "..\GenericGraphicsEngine\GCMatrix.h"
#include <tuple>
#include <vector>
#include "optix_types.h"
#include "Structs.h"


//Pure interface
class IShape
{
public:
	virtual IntersectInfo IntersectWith(const Ray&, bool) = 0;
	virtual Material GetMaterial() const = 0;
	virtual void SetMaterialCache(std::vector<Material>* cache) = 0;
	virtual void SetMaterial(const Material&) = 0;
	virtual void PrintMe() = 0;
	virtual int32_t GetMaterialIndex() const = 0;
};

class CTriangle : public IShape
{
private:
	GCPoint m_pointA;
	GCPoint m_pointB;
	GCPoint m_pointC;
	std::vector<Material>* m_materialCache = nullptr;
	int32_t m_materialIndex = -1;

public:
	CTriangle(const GCPoint& pointA, const GCPoint& pointB, const GCPoint& pointC):m_pointA(pointA), m_pointB(pointB), m_pointC(pointC){}
	IntersectInfo IntersectWith(const Ray&, bool backFaceCulling);
	IntersectInfo IntersectWith_impl_old(const Ray&);
	IntersectInfo IntersectWith_impl_new(const Ray&, bool backFaceCulling);
	void ApplyTransform(const GCMatrix& mat);
	void SetMaterialCache(std::vector<Material>* cache) override;
	Material GetMaterial() const override;
	void SetMaterial(const Material& m) override;
	void PrintMe(){std::cout<<"Triangle: PtA - "<<m_pointA<<", PtB - "<<m_pointB<<", PtC - "<<m_pointC<<std::endl;}
	bool newAlgo = true;	
	std::tuple<GCPoint, GCPoint, GCPoint> GetVertices();
	int32_t GetMaterialIndex() const override
	{
		return m_materialIndex;
	}
};

class CSphere : public IShape
{
private:
	GCPoint m_centerPt;
	double m_radius;
	GCMatrix m_transformMat;
	std::vector<Material>* m_materialCache = nullptr;
	int32_t m_materialIndex = -1;
	Material m_material;

public:
	//Optix variables	
	OptixTraversableHandle   sphere_gas_handle = 0;  // Traversable handle for sphere AS
	CUdeviceptr              d_sphere_gas_output_buffer = 0;  // Sphere AS memory


	CSphere(const GCPoint& centerPt, double radius):m_centerPt(centerPt), m_radius(radius){}
	void SetTransform(const GCMatrix& mat){m_transformMat = mat;}
	GCMatrix GetTransform() const { return m_transformMat; }
	IntersectInfo IntersectWith(const Ray&, bool backFaceCulling);
	Material GetMaterial() const override;
	void SetMaterial(const Material& m) override;
	void PrintMe(){std::cout<<"Sphere Center:"<<m_centerPt<<std::endl;}
	void SetMaterialCache(std::vector<Material>* cache) override;
	GCPoint GetCenter() const { return m_centerPt; }
	float GetRadius() const { return m_radius; }
	int32_t GetMaterialIndex() const override
	{
		//Not implemented yet
		return m_materialIndex;
	}
};




#endif // !_Shapes_h_
