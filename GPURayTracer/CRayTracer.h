#ifndef _CRayTracer_h_
#define _CRayTracer_h_

#include <string>
#include <memory>
#include <vector>
#include "..\GenericGraphicsEngine\GCMatrix.h"
#include "..\GenericGraphicsEngine\GCPoint.h"
#include "LinearAlgebraTool.h"
#include "Shapes.h"
#include "Lights.h"
#include <assert.h>
#include "enumdefs.h"
//#include "..\FreeImage\FreeImage.h"

typedef std::shared_ptr<IShape> SmartShape;
typedef std::shared_ptr<ILight> SmartLight;

class OptixTracer_impl;
class CRayTracer{
friend class OptixTracer_impl;
private:
	std::string m_filename;
	int m_width;
	int m_height;
	double m_aspect;
	int m_maxSampleIdx;
	double m_fov;
	float m_gamma = 1.0f;
	GCMatrix m_cameraMat;
	GCPoint m_rayOrigin;
	int m_maxDepth;
	int m_numSamples = 1;
	int m_samplesPerPixel = 1;
	bool m_bStratified = false;
	NEEMode m_neeMode = NEEMode::Off;
	bool m_bRR = false;  //Russian Roulette
	std::vector< std::shared_ptr<IShape> > m_shapes;
	std::vector< std::shared_ptr<ILight> > m_lights;
	//Optix-related
	std::unique_ptr<OptixTracer_impl> m_optixTracer;
	IntegratorType m_integrator = IntegratorType::Raytracer;
	ImportanceSamplingType m_samplingType = ImportanceSamplingType::Hemisphere;
	std::vector<Material> m_materialCache;

public:
	CRayTracer();
	~CRayTracer();
	void SetFilename(std::string name){m_filename = name;}
	void SetSize(int width, int height);
	void SetMaxDepth(int depth){m_maxDepth = depth;}
	void SetIntegrator(IntegratorType integrator) { m_integrator = integrator; }
	void SetCamera(const GCPoint & eye, const GCPoint& lookAt, const GCVector& up, double fov);
	Ray GetSampleRayAtIdx(int idx);
	void AddShape(IShape* shape);
	void AddLight(ILight* light);
	int  GetLightCount() const;
	void EnableStratification(bool value);
	void SetNEE(NEEMode value);
	void EnableRR(bool value);
	void SetImportanceSampling(ImportanceSamplingType samplingType);
	void SetNumSamples(int nSamples);
	void SetSamplesPerPixel(int nSamples);
	void SetGamma(float gamma);
	void IterateRays();
	//Optix-related
	void LaunchOptix(IntegratorType integratorType);
	void Render();
	//CPU-based recursive color queries
	GCVector GetColorForRay_r(const Ray& ray, int depth);
	SmartShape GetNearestShape(const Ray& ray, bool backFaceCulling);
	int ConvertColorDoubleToInt(double colorDouble);
	std::vector<Material>* GetMaterialCachePtr();
};

#endif // !_CRayTracer_h_
