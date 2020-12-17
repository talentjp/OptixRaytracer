#ifndef _ENUM_DEFS_
#define _ENUM_DEFS_
enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

enum IntegratorType {
    Raytracer,
    AnalyticDirect,
    Direct,
    Pathtracer
};

enum ImportanceSamplingType {
    Hemisphere,
    Cosine,
    BRDF
};

enum BRDFType {
    Phong,
    GGX
};

enum NEEMode {
    Off,
    On,
    MIS
};

#endif