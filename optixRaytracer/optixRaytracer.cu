#include <optix.h>

#include "optixRayTracer.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}



//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    float3       radiance;
    float3       attenuation;
    float3       origin;
    float3       direction;
    float3       normal;
    float3       emission_color;
    float3       diffuse_color;
    float3       specular_color;
    float        shininess;
    float        roughness;
    BRDFType     brdfType;
    int          lightNum;
    unsigned int seed;
    int          done;
    int          pad;
};


struct Onb
{
  __forceinline__ __device__ Onb(const float3& normal)
  {
    m_normal = normal;

    if( fabs(m_normal.x) > fabs(m_normal.z) )
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y =  m_normal.x;
      m_binormal.z =  0;
    }
    else
    {
      m_binormal.x =  0;
      m_binormal.y = -m_normal.z;
      m_binormal.z =  m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross( m_binormal, m_normal );
  }

  __forceinline__ __device__ void inverse_transform(float3& p) const
  {
    p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

struct NEEData
{    
    float3 P;
    float3 wo;
    float3 normal;
    //maximum number of light sources
    float3* samples;
    //so we don't have to shoot again
    bool* bOccluded;
    float3* radiances;
};

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
}

static __forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<unsigned int>( occluded ) );
}

//u1 and u2 have to be [0, 1)
static __forceinline__ __device__ float3 sample_hemisphere(const float u1, const float u2)
{
  // Uniformly sample disk.
  const float theta   = acosf( u1 );
  const float phi = 2.0f * M_PIf * u2;
  //hemisphere is centered around z axis (pointing out of monitor)
  float3 p;
  p.x = cosf(phi) * sinf(theta);
  p.y = sinf(phi) * sinf(theta);
  p.z = cosf(theta);
  return p;
}

static __forceinline__ __device__ float3 sample_cosine(const float u1, const float u2)
{
    const float theta = acosf(sqrtf(u1));
    const float phi = 2.0f * M_PIf * u2;

    float3 p;
    p.x = cosf(phi) * sinf(theta);
    p.y = sinf(phi) * sinf(theta);
    p.z = cosf(theta);
    return p;
}

static __forceinline__ __device__ float3 sample_phong_brdf(const float u0, const float u1, const float u2, const float t, const float s)
{
    const float phi = 2.0f * M_PIf * u2;
    float theta;

    if (u0 > t)
    {
        //diffuse
        theta = acosf(sqrtf(u1));
    }
    else
    {
        //specular
        theta = acosf(powf(u1, 1.0 / (s + 1.0)));
    } 

    float3 p;
    p.x = cosf(phi) * sinf(theta);
    p.y = sinf(phi) * sinf(theta);
    p.z = cosf(theta);
    return p;
}

static __forceinline__ __device__ float3 sample_ggx_brdf(const float u0, const float u1, const float u2, 
    const float t, const float roughness)
{
    const float phi = 2.0f * M_PIf * u2;
    float theta;

    if (u0 > t)
    {
        //diffuse
        theta = acosf(sqrtf(u1));
    }
    else
    {
        //specular
        theta = atanf(roughness * sqrtf(u1) / sqrtf(1.0f - u1));
    }

    float3 p;
    p.x = cosf(phi) * sinf(theta);
    p.y = sinf(phi) * sinf(theta);
    p.z = cosf(theta);
    return p;
}

static __forceinline__ __device__ float3 rotate_to_world(const float3& local, const float3& normal)
{
    const float3 w = normalize(normal);
    //normal shouldn't be too close to the up vector otherwise there will be numeric errors
    const bool upTest = abs(dot(make_float3(0.0, 1.0, 0.0), w)) > 0.8;
    float3 u, v;

    if (!upTest)
    {
        const float3 up = make_float3(0.0, 1.0, 0.0);
        //up x w = u
        u = normalize(cross(up, w));
        v = cross(w, u);
    }
    else
    {
        const float3 right = make_float3(1.0, 0.0, 0.0);
        //w x right = v
        v = normalize(cross(w, right));
        u = cross(v, w);
    }
    
    return local.x * u + local.y * v + local.z * w;
}

static __forceinline__ __device__ float3 brdf_ggx(float3 specular, float3 diffuse, float roughness, float3 wo, float3 wi, float3 normal)
{
    const float3 h = normalize(wo + wi);
    //artifact removal
    const float3 fresnel = specular + (1.0f - specular) * powf(1.0f - fminf(fmaxf(dot(wi, h), 0.0f), 1.0f), 5);
    const float theta_wi = acosf(dot(wi, normal));
    const float theta_wo = acosf(dot(wo, normal));
    float masking_wi = 2.0f /
        (1.0f + sqrtf(1.0f + roughness * roughness * tan(theta_wi) * tan(theta_wi)));
    float masking_wo = 2.0f /
        (1.0f + sqrtf(1.0f + roughness * roughness * tan(theta_wo) * tan(theta_wo)));
    //artifact removal
    masking_wi = fmaxf(fminf(1.0f, masking_wi), 0.0f);
    masking_wo = fmaxf(fminf(1.0f, masking_wo), 0.0f);
    const float masking = masking_wi * masking_wo;
    //Cap it to epsilon to remove the division-by-zero artifact
    const float theta_half = fmaxf(0.000001, acosf(dot(h, normal)));
    //Will be re-used for pdf    
    const float distribution = roughness * roughness / (M_PIf * powf(cos(theta_half), 4) *
        powf(roughness * roughness + tan(theta_half) * tan(theta_half), 2));
    return diffuse / M_PIf + fresnel * masking * distribution / (4.0f * dot(wi, normal) * dot(wo, normal));
}


//TODO : Optimize these 3 functions so they don't compute the same things more than once
static __forceinline__ __device__ void generate_nee_samples(unsigned int& seed, NEEData& data)
{
    for (int i = 0; i < params.lights_polygonal.count; ++i)
    {
        const ParallelogramLight& light = params.lights_polygonal[i];
        const float r1 = rnd(seed);
        const float r2 = rnd(seed);
        data.samples[i] = light.corner + r1 * light.v1 + r2 * light.v2;
    }
}

static __forceinline__ __device__ void eval_nee(const RadiancePRD& prd, NEEData& nee_data)
{
    for (int i = 0; i < params.lights_polygonal.count; ++i)
    {
        ParallelogramLight light = params.lights_polygonal[i];
        float3 weight = make_float3(0.0, 0.0, 0.0);
        const float3 sampledPoint = nee_data.samples[i];
        //get visibility
        const float  Ldist = length(sampledPoint - nee_data.P);
        const float3 L = normalize(sampledPoint - nee_data.P);
        //We don't want to count the back side, TODO : see if we need to adjust the pdf as well
        if (dot(L, cross(light.v1, light.v2)) >= 0  && !nee_data.bOccluded[i])
        {
            float3 brdf;
            if (prd.brdfType == BRDFType::Phong)
            {
                brdf = prd.diffuse_color / M_PIf +
                    prd.specular_color * (prd.shininess + 2.0) / (2.0 * M_PIf) *
                    powf(fmaxf(dot(reflect(-nee_data.wo, nee_data.normal), L), 0.0), prd.shininess);
            }
            else if (prd.brdfType == BRDFType::GGX)
            {
                if (dot(nee_data.wo, nee_data.normal) <= 0 || dot(L, nee_data.normal) <= 0)
                {
                    brdf = make_float3(0.0f);
                }
                else
                {
                    brdf = brdf_ggx(prd.specular_color, prd.diffuse_color, prd.roughness, nee_data.wo, L, nee_data.normal);
                }
            }

            const float3 lightNormal = normalize(cross(light.v1, light.v2));
            const float G = fmaxf(0.0, dot(L, nee_data.normal)) * abs(dot(L, lightNormal)) / Ldist / Ldist;
            weight += brdf * G;
        }
        //to work around some weird negative contribution issue (black pixels)
        weight = make_float3(fmaxf(weight.x, 0.0f), fmaxf(weight.y, 0.0f), fmaxf(weight.z, 0.0f));
        const float A = length(cross(light.v1, light.v2));
        nee_data.radiances[i] = light.emission * A / params.sampleCount * weight;
    }   
}

//may be useful in the future
static __forceinline__ __device__ bool is_point_on_quad(const float3& pt, const ParallelogramLight& light)
{
    //has to be on the same plane
    const float3 p = pt - light.corner;
    const float3 crossV1 = normalize(cross(light.v1, p));
    const float3 crossV2 = normalize(cross(p, light.v2));
    //TODO : find an optimal error for this
    if (length(crossV1 - crossV2) > 0.0001)
    {
        return false;
    }
    const float dotV1 = dot(light.v1, p) / dot(light.v1, light.v1);
    if (dotV1 < 0.0001 || dotV1 > 0.9999)
    {
        return false;
    }
    const float dotV2 = dot(light.v2, p) / dot(light.v2, light.v2);
    if (dotV2 < 0.0001 || dotV2 > 0.9999)
    {
        return false;
    }
    return true;
}

static __forceinline__ __device__ float compute_pdf_nee(int lightNum, const float3& samplePt, const float3& P)
{ 
    if (lightNum == -1)
    {
        return 0.0f;
    }

    const ParallelogramLight& light = params.lights_polygonal[lightNum];
    const float  Ldist = length(samplePt - P);
    const float3 wi = normalize(samplePt - P);
    const float3 lightNormal = normalize(cross(light.v1, light.v2));
    const float A = length(cross(light.v1, light.v2));
    const float pdf = Ldist * Ldist / (A * abs(dot(lightNormal, wi)));            
    //has to be divided by the number of samples taken
    return pdf / params.lights_polygonal.count;
}

static __forceinline__ __device__ float3 generate_indirect_sample(float r0, float r1, float r2, 
                                                                  const float3& r, const RadiancePRD& prd, const float3& wo, float t)
{   
    if (params.samplingType == ImportanceSamplingType::Hemisphere)
    {
        const float3 w_local = sample_hemisphere(r1, r2);
        return rotate_to_world(w_local, prd.normal);     
    }
    else if (params.samplingType == ImportanceSamplingType::Cosine)
    {
        const float3 w_local = sample_cosine(r1, r2);
        return rotate_to_world(w_local, prd.normal);
    }
    else if (params.samplingType == ImportanceSamplingType::BRDF)
    {
        if (prd.brdfType == BRDFType::Phong)
        {
            const float3 w_local = sample_phong_brdf(r0, r1, r2, t, prd.shininess);
            if (r0 > t)
            {
                return rotate_to_world(w_local, prd.normal);
            }
            else
            {
                return rotate_to_world(w_local, r);
            }
        }
        else if (prd.brdfType == BRDFType::GGX)
        {
            const float3 w_local = sample_ggx_brdf(r0, r1, r2, t, prd.roughness);
            if (r0 > t)
            {
                return rotate_to_world(w_local, prd.normal);
            }
            else
            {
                const float3 h = rotate_to_world(w_local, prd.normal);
                return reflect(-wo, h);
            }
        }
    }
}

static __forceinline__ __device__ float3 compute_brdf(const RadiancePRD& prd, const float3& wo, const float3& wi, const float3& r)
{
    float3 brdf;

    if (prd.brdfType == BRDFType::Phong)
    {
        brdf = prd.diffuse_color / M_PIf +
            prd.specular_color * (prd.shininess + 2.0) / (2.0 * M_PIf) *
            powf(fmaxf(dot(r, wi), 0.0), prd.shininess);
    }
    else if (prd.brdfType == BRDFType::GGX)
    {
        //if either is 0 the masking term will be 0
        if (dot(wi, prd.normal) <= 0 || dot(wo, prd.normal) <= 0)
        {
            brdf = make_float3(0.0f);
        }
        else
        {
            brdf = brdf_ggx(prd.specular_color, prd.diffuse_color, prd.roughness, wo, wi, prd.normal);
        }
    }

    return brdf;
}

static __forceinline__ __device__ float compute_pdf(const float3& wo, const float3& wi, const float3& r, const RadiancePRD& prd, float t)
{
    const float G = fmaxf(0.0, dot(wi, prd.normal));

    if (params.samplingType == ImportanceSamplingType::Hemisphere)
    {
        return 1.0f / (2.0f * M_PIf);
    }
    else if (params.samplingType == ImportanceSamplingType::Cosine)
    {
        return G / M_PIf;
    }
    else if (params.samplingType == ImportanceSamplingType::BRDF)
    {
        if (prd.brdfType == BRDFType::Phong)
        {
            return (1.0 - t) * G / M_PIf +
                t * (prd.shininess + 1.0) / (2.0 * M_PIf) * powf(fmaxf(dot(r, wi), 0.0), prd.shininess);
        }
        else if (prd.brdfType == BRDFType::GGX)
        {
            if (dot(wo, prd.normal) <= 0 || dot(wi, prd.normal) <= 0)
            {
                return 0;
            }
            else
            {
                const float3 h = normalize(wo + wi);
                //const float theta_half = acosf(dot(h, prd.normal));
                const float theta_half = fmaxf(0.000001, acosf(dot(h, prd.normal)));
                const float distribution = prd.roughness * prd.roughness / (M_PIf * powf(cos(theta_half), 4) *
                    powf(prd.roughness * prd.roughness + tan(theta_half) * tan(theta_half), 2));
                return (1.0f - t) * dot(prd.normal, wi) / M_PIf + t * distribution * dot(prd.normal, h) /
                    (4.0f * dot(h, wi));
            }
        }
    } 
}

static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        )
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1 );
}


static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    unsigned int occluded = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION,      // missSBTIndex
            occluded );
    return occluded;
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w    = params.width;
    const int    h    = params.height;
    const float3 eye  = params.eye;
    const float3 U    = params.U;
    const float3 V    = params.V;
    const float3 W    = params.W;
    const float  fov  = params.fov;
    const uint3  idx  = optixGetLaunchIndex();   
    unsigned int seed = tea<4>(idx.y, idx.x);

    float3 result = make_float3( 0.0f );

    float xVec = w / (float)h * (idx.x + 0.5 - w / 2.0) / (w / 2.0);
    float yVec = (idx.y + 0.5 - h / 2.0) / (h / 2.0);
    float zVec = -1.0 / tan(fov / 2.0 / 180.0 * M_PIf);
    float3 rayDir = normalize(make_float3(xVec, yVec, zVec));      
    float3 U_t = make_float3(U.x, V.x, W.x);
    float3 V_t = make_float3(U.y, V.y, W.y);
    float3 W_t = make_float3(U.z, V.z, W.z);
    //rotate (x,y,z) from local space to world space
    float3 ray_direction = normalize(make_float3(dot(U_t, rayDir), dot(V_t, rayDir), dot(W_t, rayDir)));
    float3 ray_origin    = eye;

    RadiancePRD prd;
    prd.radiance     = make_float3(0.f);
    prd.attenuation  = make_float3(1.f);
    prd.done         = false;
    prd.seed = seed;

    if (params.integratorType == IntegratorType::Raytracer)
    {
        int depth = 1;
        float3 prev_attenuation = prd.attenuation;
        for (;; )
        {
            traceRadiance(
                params.handle,
                ray_origin,
                ray_direction,
                0.01f,  // tmin       // TODO: smarter offset
                1e16f,  // tmax
                &prd);

            result += prd.radiance * prev_attenuation;
            prev_attenuation = prd.attenuation;

            if (prd.done || depth >= params.maxDepth)
                break;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }
    }
    else if(params.integratorType == IntegratorType::AnalyticDirect || params.integratorType == IntegratorType::Direct) //direct lighting doesn't need to go deeper
    {
        traceRadiance(
            params.handle,
            ray_origin,
            ray_direction,
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &prd);

        result += prd.radiance;
    }
    else if (params.integratorType == IntegratorType::Pathtracer)
    {
        //first ray is centered (like before)        
        for (int i = 0; i < params.samplesPerPixel; ++i) {
            //The first ray has to be centered, the other ones can be random
            if (i > 0)
            {
                xVec = w / (float)h * (idx.x + rnd(seed) - w / 2.0) / (w / 2.0);
                yVec = (idx.y + rnd(seed) - h / 2.0) / (h / 2.0);
                zVec = -1.0 / tan(fov / 2.0 / 180.0 * M_PIf);
                rayDir = normalize(make_float3(xVec, yVec, zVec));
                U_t = make_float3(U.x, V.x, W.x);
                V_t = make_float3(U.y, V.y, W.y);
                W_t = make_float3(U.z, V.z, W.z);
                //rotate (x,y,z) from local space to world space
                ray_direction = normalize(make_float3(dot(U_t, rayDir), dot(V_t, rayDir), dot(W_t, rayDir)));
                ray_origin = eye;
            }

            //re-initialize payload            
            prd.attenuation = make_float3(1.f);
            prd.done = false;
            prd.seed = seed;
            //reset depth
            int depth = 1;
            float3 attenuation = make_float3(1.0);
            float rrBoost = 1.0;
            float3 cur_ray_origin = ray_origin;
            float3 cur_ray_direction = ray_direction;
            float3 prev_ray_origin;
            float3 prev_ray_direction;
            float3 prev_brdf;
            float  prev_pdf;
            float  prev_G;
            float3 prev_T;
            //the contributions from the previous iteation
            float3 radiance_multiplier = make_float3(1.0f);
            float3 radiance_nee = make_float3(0.0f);
            float3 attenuation_factor = make_float3(1.0f);

            //TODO : break this into NEE on/off/mis
            for (;; )
            {
                prd.radiance = make_float3(0.f);
                prd.emission_color = make_float3(0.f);

                traceRadiance(
                    params.handle,
                    cur_ray_origin,
                    cur_ray_direction,
                    0.01f,  // tmin       // TODO: smarter offset
                    1e16f,  // tmax
                    &prd);

                if (params.neeMode == NEEMode::On)
                {
                    //Corner case - if the first ray hits the light it needs the emission
                    if (depth == 1)
                    {
                        result += prd.emission_color;
                    }

                    result += prd.radiance * attenuation * rrBoost;

                    //We hit the light, it's over
                    if (prd.done)
                    {
                        break;
                    }
                    //We got too deep , it's over
                    //TODO : make sure the definition of depth matches our implementation
                    if (params.maxDepth != -1 && depth >= params.maxDepth)
                    {
                        break;
                    }
                }
                else if (params.neeMode == NEEMode::Off)
                {
                    result += prd.radiance * attenuation * rrBoost;

                    //We hit the light, it's over
                    if (prd.done)
                    {
                        break;
                    }
                    //We got too deep , it's over
                    //TODO : make sure the definition of depth matches our implementation
                    if (params.maxDepth != -1 && depth > params.maxDepth)
                    {
                        break;
                    }

                }
                else if (params.neeMode == NEEMode::MIS)
                {
                    if (depth == 1)
                    {
                        result += prd.radiance;
                    }
                    else
                    {
                        float pdf_random_nee = compute_pdf_nee(prd.lightNum, prd.origin, cur_ray_origin);                        
                        //we don't want zero contributions
                        if (prev_T.x != 0 || prev_T.y != 0 || prev_T.z != 0)
                        {
                            const float random_weight = prev_pdf * prev_pdf / (prev_pdf * prev_pdf + pdf_random_nee * pdf_random_nee);
                            result += attenuation * rrBoost * random_weight * prd.radiance * prev_T;
                        } 

                        attenuation *= prev_T;
                    }                   

                    //We hit the light, it's over
                    if (prd.done)
                    {
                        break;
                    }
                    //We got too deep , it's over
                    //TODO : make sure the definition of depth matches our implementation
                    if (params.maxDepth != -1 && depth > params.maxDepth)
                    {
                        break;
                    }
                }

                const float r0 = rnd(seed);
                const float r1 = rnd(seed);
                const float r2 = rnd(seed);
                const float diffuse_avg = (prd.diffuse_color.x + prd.diffuse_color.y + prd.diffuse_color.z) / 3.0;
                const float specular_avg = (prd.specular_color.x + prd.specular_color.y + prd.specular_color.z) / 3.0;
                float t = specular_avg / (diffuse_avg + specular_avg);                
                //We want to sample fresnel even if specular is 0
                if (prd.brdfType == BRDFType::GGX)
                {
                    t = fmaxf(0.25f, t);
                    //if undefined, we still want to sample fresnel
                    if (diffuse_avg == 0 && specular_avg == 0)
                    {
                        t = 1.0f;
                    }
                }

                //reflection vector
                const float3 r = reflect(cur_ray_direction, prd.normal);
                const float3 w = generate_indirect_sample(r0, r1, r2, r, prd, -cur_ray_direction, t);
                //update attenuation to brdf of this surface and reflection ray
                const float3 brdf = compute_brdf(prd, -cur_ray_direction, w, r);                                          
                const float G = fmaxf(0.0, dot(w, prd.normal));
                const float pdf = compute_pdf(-cur_ray_direction, w, r, prd, t);
                float3 T;
                //division-by-zero guard
                if (pdf < 0.000001)
                {
                    T = make_float3(0.0, 0.0, 0.0);
                }
                else
                {
                    T = brdf * G / pdf;                    
                }

                if (params.neeMode == NEEMode::MIS)
                { 
                    float3 samples[32];
                    bool   occluded[32];
                    float3 radiances[32];
                    NEEData nee_data;
                    nee_data.samples = &samples[0];
                    nee_data.bOccluded = &occluded[0];
                    nee_data.radiances = &radiances[0];
                    nee_data.normal = prd.normal;
                    nee_data.P = prd.origin;
                    nee_data.wo = -cur_ray_direction;
                    generate_nee_samples(seed, nee_data);

                    for (int i = 0; i < params.lights_polygonal.count; ++i)
                    {
                        const float  Ldist = length(nee_data.samples[i] - nee_data.P);
                        const float3 L = normalize(nee_data.samples[i] - nee_data.P);

                        const bool occluded = traceOcclusion(
                            params.handle,
                            nee_data.P,
                            L,
                            0.01f,         // tmin
                            Ldist - 0.01f  // tmax
                        );
                        nee_data.bOccluded[i] = occluded;
                    }
                    //calculate contribution of all samples, result will be in radiances[i]
                    eval_nee(prd, nee_data);
                    //calculate weight for nee samples and accumualte radiance
                    float3 radiance_from_nee = make_float3(0.0f);

                    for (int i = 0; i < params.lights_polygonal.count; ++i)
                    {
                        if (nee_data.bOccluded[i])
                        {
                            continue;
                        }
                        //TODO:see if this works for non brdf sampling too
                        const float pdf_light_brdf = compute_pdf(-cur_ray_direction, normalize(nee_data.samples[i] - nee_data.P), r, prd, t);
                        float pdf_light_nee = compute_pdf_nee(i, nee_data.samples[i], nee_data.P);   
                        const float nee_weight = pdf_light_nee * pdf_light_nee /
                            (pdf_light_brdf * pdf_light_brdf + pdf_light_nee * pdf_light_nee);
                        radiance_from_nee += nee_weight * nee_data.radiances[i];
                    }

                    result += radiance_from_nee * attenuation * rrBoost;
                    prev_brdf = brdf;
                    prev_pdf = pdf;
                    prev_G = G;
                    prev_T = T;
                }
                //minimum of 0.2 to reduce noise
                const float p = fmaxf(0.2, fminf(1.0, fmaxf(fmaxf(T.x, T.y), T.z)));
                //Didn't pass the russian roulette, it's over
                if (params.bRR)
                {
                    if (rnd(seed) >= p)
                    {
                        break;
                    }
                    //to restore the bias
                    rrBoost /= p;
                }

                if (params.neeMode == NEEMode::Off || params.neeMode == NEEMode::On)
                {
                    attenuation *= T;
                }                                           

                cur_ray_origin = prd.origin;
                cur_ray_direction = w;
                ++depth;                
            }
        }

        result = result / params.samplesPerPixel;
    }
    //clamp the output to 0 ~ 1
    result = make_float3(fmaxf(fminf(result.x, 1.0f), 0.0f), 
                         fmaxf(fminf(result.y, 1.0f), 0.0f),
                         fmaxf(fminf(result.z, 1.0f), 0.0f));
    //gamma correction
    result = make_float3(powf(result.x, 1.0f / params.gamma), 
        powf(result.y, 1.0f / params.gamma), 
        powf(result.z, 1.0f / params.gamma));
    //quantize pixel and place in buffer
    const uint3    launch_index = optixGetLaunchIndex();

    //Debugging
    //if (launch_index.x == 427 && launch_index.y == 511 - 45)
    //{
    //    result = make_float3(1.0, 0, 0);
    //}

    const unsigned int image_index  = launch_index.y * params.width + launch_index.x;
    params.frame_buffer[image_index] = make_uchar4(result.x * 255, result.y * 255, result.z * 255, 255);
}


extern "C" __global__ void __miss__radiance()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD* prd = getPRD();

    prd->radiance = make_float3( rt_data->bg_color );    
    prd->done      = true;
    prd->lightNum = -1;
}


extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}


extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx * 3;
    float3 N;

    if (optixIsTriangleHit())
    {
        const float3 v0 = make_float3(rt_data->vertices[vert_idx_offset + 0]);
        const float3 v1 = make_float3(rt_data->vertices[vert_idx_offset + 1]);
        const float3 v2 = make_float3(rt_data->vertices[vert_idx_offset + 2]);
        const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));
        //the normal that faces the ray, either N_0 or -N_0
        N = faceforward(N_0, -ray_dir, N_0);
    }
    else
    {
        //is sphere
        N = make_float3(int_as_float(optixGetAttribute_0()),
                        int_as_float(optixGetAttribute_1()),
                        int_as_float(optixGetAttribute_2()));
    }

    //intersecting point
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    RadiancePRD* prd = getPRD();
    {       
        prd->direction = ray_dir - 2 * (ray_dir * N) * N;
        prd->origin    = P;    
    }

    if (params.integratorType == IntegratorType::Raytracer)
    {
        PointLight   light = params.light;
        const float  Ldist = length(light.position - P);
        const float3 L = normalize(light.position - P);

        prd->radiance = rt_data->emission_color + rt_data->ambient_color;
        //we only support one light atm
        const bool occluded = traceOcclusion(
            params.handle,
            P,
            L,
            0.01f,         // tmin
            Ldist - 0.01f  // tmax
        );

        if (!occluded)
        {
            float3 halfVec = normalize(-ray_dir + L);
            float3 color = light.emission * (rt_data->diffuse_color * max(0.0f, dot(N, L)) + rt_data->specular_color * pow(max(0.0f, dot(N, halfVec)), rt_data->shininess));
            float  weight = 1.0f / (light.attenuation.x + light.attenuation.y * Ldist + light.attenuation.z * Ldist * Ldist);
            prd->radiance += color * weight;
        }

        //every bounce reduces the contribution
        prd->attenuation *= rt_data->specular_color;
    }
    else if(params.integratorType == IntegratorType::AnalyticDirect)
    {
        prd->radiance = rt_data->emission_color;

        for (int i = 0; i < params.lights_polygonal.count; ++i)
        {
            ParallelogramLight light = params.lights_polygonal[i];
            float3 a = light.corner;
            float3 b = light.corner + light.v1;
            float3 c = light.corner + light.v2;
            float3 d = light.corner + light.v1 + light.v2;
            float thetaAB = acos(dot(normalize(a - P), normalize(b - P)));
            float3 crossAB = normalize(cross(a - P, b - P));
            float thetaBD = acos(dot(normalize(b - P), normalize(d - P)));
            float3 crossBD = normalize(cross(b - P, d - P));
            float thetaDC = acos(dot(normalize(d - P), normalize(c - P)));
            float3 crossDC = normalize(cross(d - P, c - P));
            float thetaCA = acos(dot(normalize(c - P), normalize(a - P)));
            float3 crossCA = normalize(cross(c - P, a - P));
            float3 projection = (thetaAB * crossAB + thetaBD * crossBD + thetaDC * crossDC + thetaCA * crossCA) / 2.0f;
            prd->radiance += rt_data->diffuse_color / M_PIf * light.emission * abs(dot(projection, N));
        }
    }
    else if (params.integratorType == IntegratorType::Direct)
    {
        prd->radiance = rt_data->emission_color;
        if (rt_data->lightNum != -1)
        {
            return;
        }

        for (int i = 0; i < params.lights_polygonal.count; ++i)
        {

            ParallelogramLight light = params.lights_polygonal[i];
            float3 weight = make_float3(0.0, 0.0, 0.0);

            for (int i = 0; i < params.sampleCount; ++i)
            {
                const float r1 = rnd(prd->seed);
                const float r2 = rnd(prd->seed);
                float3 sampledPoint;
                if (!params.bStratified)
                {
                    sampledPoint = light.corner + r1 * light.v1 + r2 * light.v2;
                }
                else
                {
                    const int grid_width = int(sqrtf(params.sampleCount));
                    int row = i / grid_width;
                    int col = i % grid_width;
                    sampledPoint = light.corner +
                        (row + r1) / (float)grid_width * light.v1 +
                        (col + r2) / (float)grid_width * light.v2;
                }
                //get visibility
                const float  Ldist = length(sampledPoint - P);
                const float3 L = normalize(sampledPoint - P);

                const bool occluded = traceOcclusion(
                    params.handle,
                    P,
                    L,
                    0.01f,         // tmin
                    Ldist - 0.01f  // tmax
                );

                if (!occluded)
                {
                    const float3 brdf = rt_data->diffuse_color / M_PIf +
                        rt_data->specular_color * (rt_data->shininess + 2.0) / (2.0 * M_PIf) *
                        powf(fmaxf(dot(reflect(ray_dir, N), L), 0.0), rt_data->shininess);
                    const float3 lightNormal = normalize(cross(light.v1, light.v2));
                    const float G = fmaxf(0.0, dot(L, N)) * abs(dot(L, lightNormal)) / Ldist / Ldist;
                    weight += brdf * G;
                }
            }
            const float A = length(cross(light.v1, light.v2));
            prd->radiance += light.emission * A / params.sampleCount * weight;
            prd->radiance = make_float3(fminf(prd->radiance.x, 1.0f), fminf(prd->radiance.y, 1.0f), fminf(prd->radiance.z, 1.0f));
        }
    }
    else if (params.integratorType == IntegratorType::Pathtracer)
    {
        prd->emission_color = rt_data->emission_color;
        prd->diffuse_color = rt_data->diffuse_color;
        prd->specular_color = rt_data->specular_color;
        prd->shininess = rt_data->shininess;
        prd->roughness = rt_data->roughness;
        prd->brdfType = rt_data->brdfType;
        prd->lightNum = rt_data->lightNum;
        prd->normal = N;

        //direct-lighting path for next event estimation only
        if (params.neeMode == NEEMode::On)
        {
            //terminate if it hits the light, and it doesn't need to report the radiance
            if (rt_data->lightNum != -1)
            {
                prd->radiance = make_float3(0, 0, 0);
                prd->done = true;
                return;
            }

            float3 samples[32];
            bool   occluded[32];
            float3 radiances[32];
            NEEData nee_data;
            nee_data.samples = &samples[0];
            nee_data.bOccluded = &occluded[0];
            nee_data.radiances = &radiances[0];
            nee_data.P = P;
            nee_data.normal = N;
            nee_data.wo = -ray_dir;
            generate_nee_samples(prd->seed, nee_data);

            for (int i = 0; i < params.lights_polygonal.count; ++i)
            {
                const float  Ldist = length(nee_data.samples[i] - P);
                const float3 L = normalize(nee_data.samples[i] - P);

                const bool occluded = traceOcclusion(
                    params.handle,
                    P,
                    L,
                    0.01f,         // tmin
                    Ldist - 0.01f  // tmax
                );
                nee_data.bOccluded[i] = occluded;
            }

            eval_nee(*prd, nee_data);    

            for (int i = 0; i < params.lights_polygonal.count; ++i)
            {
                prd->radiance += radiances[i];
            }
        }
        else if(params.neeMode == NEEMode::Off || params.neeMode == NEEMode::MIS)
        {
            //NEE is off, indirect lighting only
            prd->radiance = rt_data->emission_color;
            //terminate if it hits the light, also reset radiance if it hits the back side of the light
            if (rt_data->lightNum != -1)
            {
                const ParallelogramLight& light = params.lights_polygonal[rt_data->lightNum];                
                if (dot(ray_dir, cross(light.v1, light.v2)) < 0)
                {
                    prd->radiance = make_float3(0, 0, 0);
                }

                prd->done = true;            
            }
        }
    }
}


extern "C" __global__ void __intersection__sphere()
{
    HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const int     prim_idx = optixGetPrimitiveIndex();
    const Sphere  sphere = hg_data->sphere;
    const float3  ray_orig = optixGetObjectRayOrigin();
    const float3  ray_dir = optixGetObjectRayDirection();

    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float3 O = ray_orig - sphere.center;
    const float  l = 1.0f / length(ray_dir);
    const float3 D = ray_dir * l;
    const float  radius = sphere.radius;

    float b = dot(O, D);
    float c = dot(O, O) - radius * radius;
    float disc = b * b - c;

    if (disc > 0.0f)
    {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);
        float root11 = 0.0f;
        bool  check_second = true;

        const bool do_refine = fabsf(root1) > (10.0f * radius);

        if (do_refine)
        {
            // refine root10
            float3 O1 = O + root1 * D;
            b = dot(O1, D);
            c = dot(O1, O1) - radius * radius;
            disc = b * b - c;

            if (disc > 0.0f)
            {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        float  t;
        float3 normal;
        t = (root1 + root11) * l;

        if (t > ray_tmin && t < ray_tmax)
        {
            normal = (O + (root1 + root11) * D) / radius;           
            normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));

            unsigned int p0, p1, p2, p3;
            p0 = float_as_int(normal.x);
            p1 = float_as_int(normal.y);
            p2 = float_as_int(normal.z);
            p3 = float_as_int(radius);

            //closest hit will receive the normal as attributes
            if (optixReportIntersection(t, 0, p0, p1, p2, p3))
                check_second = false;
        }

        if (check_second)
        {
            float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
            t = root2 * l;
            normal = (O + root2 * D) / radius;
            if (t > ray_tmin && t < ray_tmax)
            {
                normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));
                unsigned int p0, p1, p2, p3;
                p0 = float_as_int(normal.x);
                p1 = float_as_int(normal.y);
                p2 = float_as_int(normal.z);
                p3 = float_as_int(radius);

                optixReportIntersection(t, 0, p0, p1, p2, p3);
            }
        }
    }
}