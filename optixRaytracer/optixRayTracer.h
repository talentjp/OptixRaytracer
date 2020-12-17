#ifndef _optixRayTracer_h_
#define _optixRayTracer_h_

//enum RayType
//{
//    RAY_TYPE_RADIANCE = 0,
//    RAY_TYPE_OCCLUSION = 1,
//    RAY_TYPE_COUNT
//};
//
//enum IntegratorType {
//    Raytracer,
//    AnalyticDirect,
//    Direct
//};

#include "enumdefs.h"
#include <sutil/vec_math.h>
#include <cuda/BufferView.h>

struct PointLight
{
    float3 position;
    float3 emission;
    float3 attenuation;
};

struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;    
    float3 emission;
};

struct Sphere
{
    float3 center;
    float  radius;    

    OptixAabb getAabb() const
    {
        float3 m_min = center - radius;
        float3 m_max = center + radius;

        OptixAabb aabb = { m_min.x, m_min.y, m_min.z, m_max.x, m_max.y, m_max.z };
        return aabb;
    }
};


struct Params
{
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int maxDepth;
    unsigned int samplesPerPixel;
    float3       eye;
    float3       U;
    float3       V;
    float3       W;
    float        fov;   
    float        gamma;
    PointLight   light;
    BufferView<ParallelogramLight> lights_polygonal;
    IntegratorType integratorType;
    ImportanceSamplingType samplingType;
    unsigned int sampleCount;
    bool         bStratified;
    NEEMode      neeMode;
    bool         bRR;
    OptixTraversableHandle handle;
};

struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};

//TODO: seperate mesh from sphere attributes
struct HitGroupData
{
    float3  diffuse_color;
    float3  ambient_color;
    float3  specular_color;
    float3  emission_color;
    float   shininess;
    float   roughness;
    BRDFType brdfType;
    int     lightNum;
    float4* vertices;
    Sphere  sphere;
};

#endif