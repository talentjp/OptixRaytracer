#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

//#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>
#include <iomanip>

#include "optixRayTracer.h"
#include <array>
#include "../FreeImage/FreeImage.h"

#include "CRayTracer.h"

struct PathTracerState
{
	OptixDeviceContext context = 0;

	OptixTraversableHandle         gas_handle = 0;  // Traversable handle for triangle AS
	CUdeviceptr                    d_gas_output_buffer = 0;  // Triangle AS memory
	CUdeviceptr                    d_vertices = 0;

	OptixTraversableHandle         ias_handle = 0;  // Traversable handle for instance AS
	CUdeviceptr                    d_ias_output_buffer = 0;  // Instance AS memory

	OptixModule                    ptx_module = 0;
	OptixPipelineCompileOptions    pipeline_compile_options = {};
	OptixPipeline                  pipeline = 0;

	OptixProgramGroup              raygen_prog_group = 0;
	OptixProgramGroup              radiance_miss_group = 0;
	OptixProgramGroup              occlusion_miss_group = 0;
	OptixProgramGroup              radiance_hit_group = 0;
	OptixProgramGroup              occlusion_hit_group = 0;

	CUstream                       stream = 0;
	Params                         params;
	Params* d_params;

	OptixShaderBindingTable        sbt = {};
};

template <typename T>
struct Record
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RayGenRecord = Record<RayGenData>;
using MissRecord = Record<MissData>;
using HitGroupRecord = Record<HitGroupData>;

struct Vertex
{
	float x, y, z, pad;
};

struct Instance
{
	float transform[12];
};

class OptixTracer_impl
{
private:
	sutil::CUDAOutputBufferType output_buffer_type;
	PathTracerState m_state;
	std::vector<std::shared_ptr<CSphere>> m_spheres;

public:
	OptixTracer_impl() :output_buffer_type(sutil::CUDAOutputBufferType::GL_INTEROP)
	{

	}
	void CreateContext();
	void BuildMeshAccel(const std::vector<std::shared_ptr<IShape>>& shapes, const std::vector<Material>& materials);
	void BuildInstanceAccel(const std::vector<std::shared_ptr<IShape>>& shapes, const std::vector<Material>& materials);
	void CreateModule();
	void CreateProgramGroups();
	void CreatePipeline();
	void CreateSBT(const std::vector<Material> materials);
	void InitLaunchParams(IntegratorType integratorType, CRayTracer* raytracer);
	void RenderFrame(const std::string& outfile, const GCMatrix& mat, const GCPoint& eye, float fov);
	void CleanupState();
	void InitializeOptix(uint32_t width, uint32_t height);
};


CRayTracer::CRayTracer() : m_optixTracer(new OptixTracer_impl)
{
	SetMaxDepth(5);
	SetFilename("raytrace.png");
}

CRayTracer::~CRayTracer()
{

}

void CRayTracer::SetSize(int width, int height){
	m_width = width;
	m_height = height;
	m_aspect = width / (double)height;
	m_maxSampleIdx = width * height - 1;
}

void CRayTracer::SetCamera(const GCPoint& eye, const GCPoint& lookAt, const GCVector& up, double fov)
{
	m_fov = fov;
	m_cameraMat = CLinearAlgebraTool::LookAt(eye, lookAt, up);
	m_cameraMat.size = 3;
	m_rayOrigin = eye;
}

Ray CRayTracer::GetSampleRayAtIdx(int idx)
{
	if(idx <= m_maxSampleIdx)
	{
		int xIdx = idx % m_width;
		int yIdx = idx / m_width;
		double xVec = m_aspect * ((double)xIdx + 0.5 - m_width/2.0) / (m_width / 2.0);
		double yVec = ((double)yIdx + 0.5 - m_height / 2.0) / (m_height / 2.0);
		double zVec = -1.0 / tan(m_fov / 2.0 / 180.0 * PI);
		GCVector rayDir(xVec, yVec, zVec);
		rayDir.normalize();
		rayDir = m_cameraMat.transpose() * rayDir;
		Ray ray(m_rayOrigin,rayDir);
		return ray;
	}
	std::cout<<"Ray index is out of range!"<<std::endl;
	assert(0);

	return Ray();
}

void CRayTracer::AddShape(IShape* shape)
{
	//shape->PrintMe();
	SmartShape ptr(shape);
	m_shapes.push_back(ptr);
}

void CRayTracer::AddLight(ILight* light)
{
	SmartLight ptr(light);
	m_lights.push_back(ptr);
}

int CRayTracer::GetLightCount() const
{
	return m_lights.size();
}

void CRayTracer::EnableStratification(bool value)
{
	m_bStratified = value;
}

void CRayTracer::SetNEE(NEEMode value)
{
	m_neeMode = value;
}

void CRayTracer::EnableRR(bool value)
{
	m_bRR = value;
}

void CRayTracer::SetImportanceSampling(ImportanceSamplingType samplingType)
{
	m_samplingType = samplingType;
}

void CRayTracer::SetNumSamples(int nSamples)
{
	m_numSamples = nSamples;
}

void CRayTracer::SetSamplesPerPixel(int nSamples)
{
	m_samplesPerPixel = nSamples;
}

void CRayTracer::SetGamma(float gamma)
{
	m_gamma = gamma;
}

void CRayTracer::IterateRays()
{
	//Have to create a new image based on the dimensions
	BYTE* pixels = new BYTE[m_width * m_height * 3];
	memset(pixels, 0, m_width * m_height * 3);
	//Layout pixel[0,0] - BGR pixel[1,0] - BGR .....
	FreeImage_Initialise();

	for(int i = 0; i < m_width * m_height; i++)
	{
		Ray ray = GetSampleRayAtIdx(i);	
		GCVector color = GetColorForRay_r(ray, 0);
		//B
		pixels[i * 3]  = ConvertColorDoubleToInt(color.z);
		//G
		pixels[i * 3 + 1] = ConvertColorDoubleToInt(color.y);
		//R
		pixels[i * 3 + 2] = ConvertColorDoubleToInt(color.x);
	}

	FIBITMAP *img = FreeImage_ConvertFromRawBits(pixels, m_width, m_height, m_width * 3, 24, 0xFF0000, 0x00FF00, 0x0000FF, false);
	FreeImage_Save(FIF_PNG, img, m_filename.c_str(), 0);
	FreeImage_DeInitialise();
	delete[] pixels;
}


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void OptixTracer_impl::CreateContext()
{
	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	OptixDeviceContext context;
	CUcontext          cu_ctx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

	m_state.context = context;
}

void OptixTracer_impl::BuildMeshAccel(const std::vector<std::shared_ptr<IShape>>& shapes, const std::vector<Material>& materials)
{
	std::vector<Vertex> vertices;
	std::vector<uint32_t> triangle_mat_indices;	
	std::vector<Sphere> spheres;
	std::vector<uint32_t> sphere_mat_indices;

	for (auto shape : shapes)
	{
		auto triangle = std::dynamic_pointer_cast<CTriangle>(shape);
		if (triangle)
		{
			GCPoint a;
			GCPoint b;
			GCPoint c;
			std::tie(a, b, c) = triangle->GetVertices();
			vertices.push_back({ (float)a.x, (float)a.y, (float)a.z, 0 });
			vertices.push_back({ (float)b.x, (float)b.y, (float)b.z, 0 });
			vertices.push_back({ (float)c.x, (float)c.y, (float)c.z, 0 });
			triangle_mat_indices.push_back(triangle->GetMaterialIndex());
			continue;
		}
		auto sphere = std::dynamic_pointer_cast<CSphere>(shape);
		if (sphere)
		{
			m_spheres.push_back(sphere);
			spheres.push_back({ make_float3(sphere->GetCenter().x, sphere->GetCenter().y, sphere->GetCenter().z), sphere->GetRadius() });
			sphere_mat_indices.push_back(sphere->GetMaterialIndex());
		}
	}

	//Build triangle GAS
	if(!vertices.empty()){
		//
		// copy mesh data to device
		//
		const size_t vertices_size_in_bytes = vertices.size() * sizeof(Vertex);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.d_vertices), vertices_size_in_bytes));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(m_state.d_vertices),
			vertices.data(), vertices_size_in_bytes,
			cudaMemcpyHostToDevice
		));

		CUdeviceptr  d_mat_indices = 0;
		const size_t mat_indices_size_in_bytes = triangle_mat_indices.size() * sizeof(uint32_t);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_mat_indices),
			triangle_mat_indices.data(),
			mat_indices_size_in_bytes,
			cudaMemcpyHostToDevice
		));

		//
		// Build triangle GAS
		//
		std::vector<uint32_t> triangle_input_flags(materials.size(), OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
		OptixBuildInput triangle_input = {};
		triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
		triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
		triangle_input.triangleArray.vertexBuffers = &m_state.d_vertices;
		triangle_input.triangleArray.flags = triangle_input_flags.data();
		triangle_input.triangleArray.numSbtRecords = materials.size();
		triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
		triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
		triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			m_state.context,
			&accel_options,
			&triangle_input,
			1,  // num_build_inputs
			&gas_buffer_sizes
		));
		CUdeviceptr d_temp_buffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));
		// non-compacted output
		CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
		size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
			compactedSizeOffset + 8
		));

		OptixAccelEmitDesc emitProperty = {};
		emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

		OPTIX_CHECK(optixAccelBuild(
			m_state.context,
			0,                                  // CUDA stream
			&accel_options,
			&triangle_input,
			1,                                  // num build inputs
			d_temp_buffer,
			gas_buffer_sizes.tempSizeInBytes,
			d_buffer_temp_output_gas_and_compacted_size,
			gas_buffer_sizes.outputSizeInBytes,
			&m_state.gas_handle,
			&emitProperty,                      // emitted property list
			1                                   // num emitted properties
		));

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

		size_t compacted_gas_size;
		CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));
		//if compacted size is smaller we would try compacting
		if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
		{
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.d_gas_output_buffer), compacted_gas_size));

			// use handle as input and output
			OPTIX_CHECK(optixAccelCompact(m_state.context, 0, m_state.gas_handle, m_state.d_gas_output_buffer, compacted_gas_size, &m_state.gas_handle));

			CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
		}
		else
		{
			m_state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
		}
	}

	//
	// Build sphere GAS
	//	
	for (int i = 0; i < m_spheres.size(); ++i) {
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAabb aabb = spheres[i].getAabb();
		CUdeviceptr d_aabb_buffer;
		const size_t aabb_size_in_bytes = sizeof(OptixAabb);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), aabb_size_in_bytes));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabb_buffer), 
			&aabb,
			aabb_size_in_bytes,
			cudaMemcpyHostToDevice
		));

		uint32_t sphere_input_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
		OptixBuildInput sphere_input = {};
		sphere_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		sphere_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
		sphere_input.customPrimitiveArray.numPrimitives = spheres.size();
		sphere_input.customPrimitiveArray.flags = &sphere_input_flag;
		sphere_input.customPrimitiveArray.numSbtRecords = 1;
		sphere_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
		sphere_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
		sphere_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(m_state.context,
			&accel_options,
			&sphere_input,
			1,  // num_build_inputs
			&gas_buffer_sizes));

		CUdeviceptr d_temp_buffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

		// non-compacted output
		CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
		size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8));

		OptixAccelEmitDesc emitProperty = {};
		emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

		OPTIX_CHECK(optixAccelBuild(m_state.context,
			0,        // CUDA stream
			&accel_options,
			&sphere_input,
			1,        // num build inputs
			d_temp_buffer,
			gas_buffer_sizes.tempSizeInBytes,
			d_buffer_temp_output_gas_and_compacted_size,
			gas_buffer_sizes.outputSizeInBytes,
			&m_spheres[i]->sphere_gas_handle,
			&emitProperty,  // emitted property list
			1));          // num emitted properties

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aabb_buffer)));

		size_t compacted_gas_size;
		CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

		if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
		{
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_spheres[i]->d_sphere_gas_output_buffer), compacted_gas_size));

			// use handle as input and output
			OPTIX_CHECK(optixAccelCompact(m_state.context, 0, m_spheres[i]->sphere_gas_handle, m_spheres[i]->d_sphere_gas_output_buffer, compacted_gas_size, &m_spheres[i]->sphere_gas_handle));
			CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
		}
		else
		{
			m_spheres[i]->d_sphere_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
		}
	}	
}

void OptixTracer_impl::BuildInstanceAccel(const std::vector<std::shared_ptr<IShape>>& shapes, const std::vector<Material>& materials)
{
	CUdeviceptr d_instances;
	size_t      instance_size_in_bytes = sizeof(OptixInstance) * (1 + m_spheres.size());
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instance_size_in_bytes));

	OptixBuildInput instance_input = {};

	instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instance_input.instanceArray.instances = d_instances;
	instance_input.instanceArray.numInstances = 1 + m_spheres.size();

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes ias_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(m_state.context, &accel_options, &instance_input,
		1,  // num build inputs
		&ias_buffer_sizes));

	CUdeviceptr d_temp_buffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), ias_buffer_sizes.tempSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.d_ias_output_buffer), ias_buffer_sizes.outputSizeInBytes));

	// Use the identity matrix for the instance transform
	Instance instance = { { 1, 0, 0, 0, 
		                    0, 1, 0, 0, 
		                    0, 0, 1, 0 } };

	std::vector<OptixInstance> optix_instances(1 + m_spheres.size());
	memset(optix_instances.data(), 0, instance_size_in_bytes);

	optix_instances[0].traversableHandle = m_state.gas_handle;
	optix_instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
	optix_instances[0].instanceId = 0;
	optix_instances[0].sbtOffset = 0;
	optix_instances[0].visibilityMask = 1;
	memcpy(optix_instances[0].transform, instance.transform, sizeof(float) * 12);


	for (int i = 0; i < m_spheres.size(); ++i)
	{
		optix_instances[i + 1].traversableHandle = m_spheres[i]->sphere_gas_handle;
		optix_instances[i + 1].flags = OPTIX_INSTANCE_FLAG_NONE;
		optix_instances[i + 1].instanceId = i + 1;
		optix_instances[i + 1].sbtOffset = RAY_TYPE_COUNT * (materials.size() + i);
		optix_instances[i + 1].visibilityMask = 1;
		GCMatrix Mat = m_spheres[i]->GetTransform();
		Instance instance_sphere = { { Mat.elements[0][0], Mat.elements[0][1], Mat.elements[0][2], Mat.elements[0][3],
								 Mat.elements[1][0], Mat.elements[1][1], Mat.elements[1][2], Mat.elements[1][3],
								 Mat.elements[2][0], Mat.elements[2][1], Mat.elements[2][2], Mat.elements[2][3] } };
		memcpy(optix_instances[i + 1].transform, instance_sphere.transform, sizeof(float) * 12);
	}

	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instances), optix_instances.data(), instance_size_in_bytes,
		cudaMemcpyHostToDevice));

	OPTIX_CHECK(optixAccelBuild(m_state.context,
		0,  // CUDA stream
		&accel_options,
		&instance_input,
		1,  // num build inputs
		d_temp_buffer,
		ias_buffer_sizes.tempSizeInBytes,
		m_state.d_ias_output_buffer,
		ias_buffer_sizes.outputSizeInBytes,
		&m_state.ias_handle,
		nullptr,  // emitted property list
		0         // num emitted properties
	));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
}

void OptixTracer_impl::CreateModule()
{
	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

	m_state.pipeline_compile_options.usesMotionBlur = false;
	//m_state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	m_state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	m_state.pipeline_compile_options.numPayloadValues = 2;
	m_state.pipeline_compile_options.numAttributeValues = 4;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
	m_state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
	m_state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
	m_state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

	const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixRaytracer.cu");

	char   log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
		m_state.context,
		&module_compile_options,
		&m_state.pipeline_compile_options,
		ptx.c_str(),
		ptx.size(),
		log,
		&sizeof_log,
		&m_state.ptx_module
	));
}

void OptixTracer_impl::CreateProgramGroups()
{
	OptixProgramGroupOptions  program_group_options = {};

	char   log[2048];
	size_t sizeof_log = sizeof(log);

	{
		OptixProgramGroupDesc raygen_prog_group_desc = {};
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = m_state.ptx_module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			m_state.context, &raygen_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&m_state.raygen_prog_group
		));
	}

	{
		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = m_state.ptx_module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			m_state.context, &miss_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			log, &sizeof_log,
			&m_state.radiance_miss_group
		));

		memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = nullptr;  // NULL miss program for occlusion rays
		miss_prog_group_desc.miss.entryFunctionName = nullptr;
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			m_state.context, &miss_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&m_state.occlusion_miss_group
		));
	}

	{
		OptixProgramGroupDesc hit_prog_group_desc = {};
		hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hit_prog_group_desc.hitgroup.moduleCH = m_state.ptx_module;
		hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		hit_prog_group_desc.hitgroup.moduleIS = m_state.ptx_module;
		hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			m_state.context,
			&hit_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&m_state.radiance_hit_group
		));

		memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
		hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hit_prog_group_desc.hitgroup.moduleCH = m_state.ptx_module;
		hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
		hit_prog_group_desc.hitgroup.moduleIS = m_state.ptx_module;
		hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";

		sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			m_state.context,
			&hit_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&m_state.occlusion_hit_group
		));
	}
}

void OptixTracer_impl::CreatePipeline()
{
	OptixProgramGroup program_groups[] =
	{
		m_state.raygen_prog_group,
		m_state.radiance_miss_group,
		m_state.occlusion_miss_group,
		m_state.radiance_hit_group,
		m_state.occlusion_hit_group
	};

	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = 10;
	pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

	char   log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixPipelineCreate(
		m_state.context,
		&m_state.pipeline_compile_options,
		&pipeline_link_options,
		program_groups,
		sizeof(program_groups) / sizeof(program_groups[0]),
		log,
		&sizeof_log,
		&m_state.pipeline
	));

	// We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
	// parameters to optixPipelineSetStackSize.
	OptixStackSizes stack_sizes = {};
	OPTIX_CHECK(optixUtilAccumulateStackSizes(m_state.raygen_prog_group, &stack_sizes));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(m_state.radiance_miss_group, &stack_sizes));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(m_state.occlusion_miss_group, &stack_sizes));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(m_state.radiance_hit_group, &stack_sizes));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(m_state.occlusion_hit_group, &stack_sizes));

	uint32_t max_trace_depth = 10;
	uint32_t max_cc_depth = 0;
	uint32_t max_dc_depth = 0;
	uint32_t direct_callable_stack_size_from_traversal;
	uint32_t direct_callable_stack_size_from_state;
	uint32_t continuation_stack_size;
	OPTIX_CHECK(optixUtilComputeStackSizes(
		&stack_sizes,
		max_trace_depth,
		max_cc_depth,
		max_dc_depth,
		&direct_callable_stack_size_from_traversal,
		&direct_callable_stack_size_from_state,
		&continuation_stack_size
	));

	const uint32_t max_traversal_depth = 1;
	OPTIX_CHECK(optixPipelineSetStackSize(
		m_state.pipeline,
		direct_callable_stack_size_from_traversal,
		direct_callable_stack_size_from_state,
		continuation_stack_size,
		max_traversal_depth
	));
}

void OptixTracer_impl::CreateSBT(const std::vector<Material> materials)
{
	CUdeviceptr  d_raygen_record;
	const size_t raygen_record_size = sizeof(RayGenRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

	RayGenRecord rg_sbt = {};
	OPTIX_CHECK(optixSbtRecordPackHeader(m_state.raygen_prog_group, &rg_sbt));

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_raygen_record),
		&rg_sbt,
		raygen_record_size,
		cudaMemcpyHostToDevice
	));

	CUdeviceptr  d_miss_records;
	const size_t miss_record_size = sizeof(MissRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

	MissRecord ms_sbt[2];
	OPTIX_CHECK(optixSbtRecordPackHeader(m_state.radiance_miss_group, &ms_sbt[0]));
	ms_sbt[0].data.bg_color = make_float4(0.0f);
	OPTIX_CHECK(optixSbtRecordPackHeader(m_state.occlusion_miss_group, &ms_sbt[1]));
	ms_sbt[1].data.bg_color = make_float4(0.0f);

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_miss_records),
		ms_sbt,
		miss_record_size * RAY_TYPE_COUNT,
		cudaMemcpyHostToDevice
	));

	CUdeviceptr  d_hitgroup_records;
	const size_t hitgroup_record_size = sizeof(HitGroupRecord);

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&d_hitgroup_records),
		hitgroup_record_size * RAY_TYPE_COUNT * (materials.size() + m_spheres.size())
	));

	auto VecToColor = [](const GCVector& vec)
	{
		return make_float3(vec.x, vec.y, vec.z);
	};

	auto PtToFloat3 = [](const GCPoint& pt)
	{
		return make_float3(pt.x, pt.y, pt.z);
	};

	std::vector<HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT * (materials.size() + m_spheres.size()));
	for (int i = 0; i < materials.size(); ++i)
	{
		{
			const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

			OPTIX_CHECK(optixSbtRecordPackHeader(m_state.radiance_hit_group, &hitgroup_records[sbt_idx]));
			hitgroup_records[sbt_idx].data.emission_color = VecToColor(materials[i].emission);
			hitgroup_records[sbt_idx].data.diffuse_color = VecToColor(materials[i].diffuse);
			hitgroup_records[sbt_idx].data.ambient_color = VecToColor(materials[i].ambient);
			hitgroup_records[sbt_idx].data.specular_color = VecToColor(materials[i].specular);
			//For modified phong
			hitgroup_records[sbt_idx].data.shininess = materials[i].shininess;
			//For GGX
			hitgroup_records[sbt_idx].data.roughness = materials[i].roughness;
			hitgroup_records[sbt_idx].data.brdfType = materials[i].brdfType;
			hitgroup_records[sbt_idx].data.lightNum = materials[i].lightNum;
			hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float4*>(m_state.d_vertices);
		}

		{
			const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
			//zero initialize the values
			memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);
			OPTIX_CHECK(optixSbtRecordPackHeader(m_state.occlusion_hit_group, &hitgroup_records[sbt_idx]));
		}
	}

	//For spheres, one sphere has one material (TODO : merge materials)
	for (int i = 0; i < m_spheres.size(); ++i)
	{
		{
			const int sbt_idx = (i + materials.size()) * RAY_TYPE_COUNT + 0;
			OPTIX_CHECK(optixSbtRecordPackHeader(m_state.radiance_hit_group, &hitgroup_records[sbt_idx]));
			hitgroup_records[sbt_idx].data.emission_color = VecToColor(m_spheres[i]->GetMaterial().emission);
			hitgroup_records[sbt_idx].data.diffuse_color = VecToColor(m_spheres[i]->GetMaterial().diffuse);
			hitgroup_records[sbt_idx].data.ambient_color = VecToColor(m_spheres[i]->GetMaterial().ambient);
			hitgroup_records[sbt_idx].data.specular_color = VecToColor(m_spheres[i]->GetMaterial().specular);
			//For modified phong
			hitgroup_records[sbt_idx].data.shininess = m_spheres[i]->GetMaterial().shininess;
			//For GGX
			hitgroup_records[sbt_idx].data.roughness = m_spheres[i]->GetMaterial().roughness;
			hitgroup_records[sbt_idx].data.brdfType = m_spheres[i]->GetMaterial().brdfType;
			hitgroup_records[sbt_idx].data.sphere.center = PtToFloat3(m_spheres[i]->GetCenter());
			hitgroup_records[sbt_idx].data.sphere.radius = m_spheres[i]->GetRadius();
		}

		{
			const int sbt_idx = (i + materials.size()) * RAY_TYPE_COUNT + 1;
			memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);
			OPTIX_CHECK(optixSbtRecordPackHeader(m_state.occlusion_hit_group, &hitgroup_records[sbt_idx]));
			hitgroup_records[sbt_idx].data.sphere.center = PtToFloat3(m_spheres[i]->GetCenter());
			hitgroup_records[sbt_idx].data.sphere.radius = m_spheres[i]->GetRadius();
		}
	}

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_hitgroup_records),
		hitgroup_records.data(),
		hitgroup_record_size * RAY_TYPE_COUNT * (materials.size() + m_spheres.size()),
		cudaMemcpyHostToDevice
	));

	m_state.sbt.raygenRecord = d_raygen_record;
	m_state.sbt.missRecordBase = d_miss_records;
	m_state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
	m_state.sbt.missRecordCount = RAY_TYPE_COUNT;
	m_state.sbt.hitgroupRecordBase = d_hitgroup_records;
	m_state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
	m_state.sbt.hitgroupRecordCount = RAY_TYPE_COUNT * (materials.size() + m_spheres.size());
}

void OptixTracer_impl::InitLaunchParams(IntegratorType integratorType, CRayTracer* raytracer)
{
	m_state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

	std::vector<ParallelogramLight> quadLights;
	for (auto light : raytracer->m_lights)
	{
		auto pointLight = std::dynamic_pointer_cast<CPointLite>(light);
		if (pointLight)
		{
			//TODO : finish adding point light
			m_state.params.light.position = make_float3(pointLight->GetPosition().x, pointLight->GetPosition().y, pointLight->GetPosition().z);
			m_state.params.light.emission = make_float3(pointLight->GetColor().x, pointLight->GetColor().y, pointLight->GetColor().z);
			m_state.params.light.attenuation = make_float3(pointLight->GetAttenuation().x, pointLight->GetAttenuation().y, pointLight->GetAttenuation().z);
			continue;
		}
		auto quadLight = std::dynamic_pointer_cast<CPolygonalLite>(light);
		if (quadLight)
		{
			ParallelogramLight deviceLight;
			deviceLight.corner = make_float3(quadLight->GetCorner().x, quadLight->GetCorner().y, quadLight->GetCorner().z);
			deviceLight.v1 = make_float3(quadLight->GetVectorAB().x, quadLight->GetVectorAB().y, quadLight->GetVectorAB().z);
			deviceLight.v2 = make_float3(quadLight->GetVectorAC().x, quadLight->GetVectorAC().y, quadLight->GetVectorAC().z);
			deviceLight.emission = make_float3(quadLight->GetColor().x, quadLight->GetColor().y, quadLight->GetColor().z);
			quadLights.push_back(deviceLight);
		}
	}

	//TODO : extend to multiple point lights

	if (!quadLights.empty())
	{
		m_state.params.lights_polygonal.count = quadLights.size();
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_state.params.lights_polygonal.data ), quadLights.size() * sizeof(ParallelogramLight) ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( m_state.params.lights_polygonal.data ), quadLights.data(),
			                    quadLights.size() * sizeof(ParallelogramLight), cudaMemcpyHostToDevice ) );
	}
	//TODO : release quadlights
	m_state.params.integratorType = integratorType;
	m_state.params.samplingType = raytracer->m_samplingType;
	m_state.params.sampleCount = raytracer->m_numSamples;
	m_state.params.maxDepth = raytracer->m_maxDepth;
	m_state.params.samplesPerPixel = raytracer->m_samplesPerPixel;
	m_state.params.bStratified = raytracer->m_bStratified;
	m_state.params.neeMode = raytracer->m_neeMode;
	m_state.params.bRR = raytracer->m_bRR;	
	m_state.params.handle = m_state.ias_handle;
	m_state.params.gamma = raytracer->m_gamma;

	CUDA_CHECK(cudaStreamCreate(&m_state.stream));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.d_params), sizeof(Params)));
}

void OptixTracer_impl::RenderFrame(const std::string& outfile, const GCMatrix& mat, const GCPoint& eye, float fov)
{
	if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
	{
		sutil::initGLFW();  // For GL context
		sutil::initGL();
	}

	sutil::CUDAOutputBuffer<uchar4> output_buffer(
		output_buffer_type,
		m_state.params.width,
		m_state.params.height
	);
	//Update camera params
	m_state.params.eye = make_float3(eye.x, eye.y, eye.z);
	m_state.params.U = make_float3(mat(1, 1), mat(1, 2), mat(1, 3));
	m_state.params.V = make_float3(mat(2, 1), mat(2, 2), mat(2, 3));
	m_state.params.W = make_float3(mat(3, 1), mat(3, 2), mat(3, 3));
	m_state.params.fov = fov;

	//Resize the buffer
	output_buffer.resize(m_state.params.width, m_state.params.height);

	//Launch frame
	// Launch
	uchar4* result_buffer_data = output_buffer.map();
	m_state.params.frame_buffer = result_buffer_data;
	CUDA_CHECK(cudaMemcpyAsync(
		reinterpret_cast<void*>(m_state.d_params),
		&m_state.params, sizeof(Params),
		cudaMemcpyHostToDevice, m_state.stream
	));

	OPTIX_CHECK(optixLaunch(
		m_state.pipeline,
		m_state.stream,
		reinterpret_cast<CUdeviceptr>(m_state.d_params),
		sizeof(Params),
		&m_state.sbt,
		m_state.params.width,   // launch width
		m_state.params.height,  // launch height
		1                       // launch depth
	));
	output_buffer.unmap();
	CUDA_SYNC_CHECK();

	sutil::ImageBuffer buffer;
	buffer.data = output_buffer.getHostPointer();
	buffer.width = output_buffer.width();
	buffer.height = output_buffer.height();
	buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

	sutil::saveImage(outfile.c_str(), buffer, true);
	CleanupState();
}

void OptixTracer_impl::CleanupState()
{
	OPTIX_CHECK(optixPipelineDestroy(m_state.pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.raygen_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.radiance_miss_group));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.radiance_hit_group));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.occlusion_hit_group));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.occlusion_miss_group));
	OPTIX_CHECK(optixModuleDestroy(m_state.ptx_module));
	OPTIX_CHECK(optixDeviceContextDestroy(m_state.context));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.hitgroupRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_vertices)));

	for (auto sphere : m_spheres)
	{
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sphere->d_sphere_gas_output_buffer)));
	}

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_gas_output_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.params.lights_polygonal.data)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_params)));
}

void OptixTracer_impl::InitializeOptix(uint32_t width, uint32_t height)
{
	m_state.params.width = width;
	m_state.params.height = height;	
	CreateContext();
}

void CRayTracer::LaunchOptix(IntegratorType integratorType)
{
	m_optixTracer->InitializeOptix(m_width, m_height);
	m_optixTracer->BuildMeshAccel(m_shapes, m_materialCache);
	m_optixTracer->BuildInstanceAccel(m_shapes, m_materialCache);
	m_optixTracer->CreateModule();
	m_optixTracer->CreateProgramGroups();
	m_optixTracer->CreatePipeline();	
	m_optixTracer->CreateSBT(m_materialCache);
	m_optixTracer->InitLaunchParams(integratorType, this);
	m_optixTracer->RenderFrame(m_filename, m_cameraMat, m_rayOrigin, m_fov);
}

void CRayTracer::Render()
{
	LaunchOptix(m_integrator);
}

GCVector CRayTracer::GetColorForRay_r(const Ray& ray, int depth)
{
	GCVector color(0,0,0);
	if(depth < m_maxDepth)
	{
		//Color computation only works for the forward facing triangle
		SmartShape nearestShape = GetNearestShape(ray, true);
		//See if we hit anything
		if(nearestShape)
		{
			//A + E
			color = color + nearestShape->GetMaterial().ambient + nearestShape->GetMaterial().emission;
			IntersectInfo info = nearestShape->IntersectWith(ray, true);
			for(auto itr = m_lights.begin(); itr != m_lights.end(); itr++)
			{
				//ray from the intersection P to the light, should be any hit (occlusion detection => backface culling off) in Optix
				Ray rayToLite = (*itr)->GetRay(info.intersectPt);
				SmartShape blockingShape = GetNearestShape(rayToLite, false);				
				bool bLightOccluded = true;
				if(!blockingShape)
				{
					//nothing in the path between the object and the light
					bLightOccluded = false;
				}
				else
				{
					IntersectInfo blockingInfo = blockingShape->IntersectWith(rayToLite, false);
					//the blocking object is on the opposite side of the light, hence the light is visible
					double distanceToLight = (*itr)->GetDistance(rayToLite.origin);
					if(distanceToLight < blockingInfo.depth)
					{
						bLightOccluded = false;
					}
				}

				if (!bLightOccluded)
				{
					color = color + (*itr)->ComputeColor(ray.direction, info.normal, info.intersectPt,
						nearestShape->GetMaterial());
				}
			}
			Ray reflectRay;
			reflectRay.origin = info.intersectPt;
			reflectRay.direction = ray.direction - 2 * (ray.direction * info.normal) * info.normal;
			color = color + nearestShape->GetMaterial().specular.pieceWiseMultiply(GetColorForRay_r(reflectRay, depth + 1));
		}
	}
	return color;
}

SmartShape CRayTracer::GetNearestShape(const Ray& ray, bool backFaceCulling)
{	
	SmartShape ptr_shape = nullptr;
	double min_distance = DBL_MAX;

	for(auto itr = m_shapes.begin(); itr != m_shapes.end(); itr++)
	{
		IntersectInfo info = (*itr)->IntersectWith(ray, backFaceCulling);
		if(info.ifIntersect)
		{
			if(info.depth < min_distance)			
			{
				ptr_shape = (*itr);
				min_distance = info.depth;
			}
		}
	}
	return ptr_shape;
}

int CRayTracer::ConvertColorDoubleToInt(double colorDouble)
{
	int colorInt = (int)(colorDouble * 256);
	colorInt = std::min(255, std::max(0, colorInt));
	return colorInt;
}

std::vector<Material>* CRayTracer::GetMaterialCachePtr()
{
	return &m_materialCache;
}
