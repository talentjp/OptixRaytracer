// GPURayTracer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <memory>
#include <regex>
#include <iostream>
#include <fstream>
#include <stack>
#include "..\GenericGraphicsEngine\GCVector.h"
#include "..\GenericGraphicsEngine\GCPoint.h"
#include "..\GenericGraphicsEngine\GCMatrix.h"
#include "..\GenericGraphicsEngine\GCDegree.h"
#include "LinearAlgebraTool.h"
#include "CRayTracer.h"

std::stack<GCMatrix> transform_stack;
std::vector<GCPoint> vertices;

void PushTransform()
{
	transform_stack.push(transform_stack.top());
}

void PopTransform()
{
	transform_stack.pop();
}

void RightMultiply(const GCMatrix& transform)
{
	transform_stack.top() = transform_stack.top() * transform;
}


std::vector<std::string> SplitString(const std::string& str)
{
	std::vector<std::string> substrs;
	uint32_t i = 0;
	int start = -1;
	int end = -1;
	while(i < str.length())
	{
		//Find first non-space character
		if(start == -1)
		{
			if(str.at(i) != ' ')
			{
				start = i;
			}
		}

		//Find end of substr if start is found
		if(start != -1 && end == -1)
		{
			if(str.at(i) == ' ') //First space after a substring
			{
				end = i;
			}
			else if(i == str.length() - 1) //End of string, assign i to end + 1
			{
				end = str.length();
			}
		}

		if(start != -1 && end != -1) //Both start and end are found, there is a substring
		{
			substrs.push_back(str.substr(start, end - start));
			start = -1;
			end = -1;
		}
		i++;
	}

	return substrs;
}

class CParserString
{
private:
	std::string m_str;
public:
	CParserString(std::string str)
	{
		m_str = str;
	}

	bool Is(std::string str)
	{
		if(m_str.compare(str) == 0)
			return true;
		return false;
	}
};

int main(int argc, char* argv[])
{
	//Have to push an initial identity matrix to stack
	transform_stack.push(GCMatrix());
	Material currMaterial;
	double   currAttenuation[3] = {1.0, 0, 0};
	CRayTracer tracer;

	if(argc < 2)
		std::cout<<"Usage: parser.py [filename]\n";
	else
	{
		std::string fileStr(argv[1]);
		std::cout<<"Parsing file: "<<fileStr<<std::endl;
		std::string line;
		std::ifstream myfile;
		myfile.open(fileStr);
		if (myfile)
		{
			while ( myfile.good() )
			{
				getline (myfile,line);
				if(!line.empty()) //line can be empty string
				{
					std::vector<std::string> substrs = SplitString(line);
					if (!substrs.empty()) //if there is only space in the string this will be empty
					{
						CParserString leadingStr(substrs[0]);
						if (leadingStr.Is("camera"))
						{
							GCPoint  eye(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str()));
							GCPoint  lookAt(atof(substrs[4].c_str()), atof(substrs[5].c_str()), atof(substrs[6].c_str()));
							GCVector up(atof(substrs[7].c_str()), atof(substrs[8].c_str()), atof(substrs[9].c_str()));
							double fov = atof(substrs[10].c_str());
							std::cout << "Camera:" << eye << lookAt << up << fov << std::endl;
							tracer.SetCamera(eye, lookAt, up, fov);
						}
						else if (leadingStr.Is("integrator"))
						{
							if (substrs[1] == "analyticdirect")
							{
								tracer.SetIntegrator(IntegratorType::AnalyticDirect);
							}
							else if (substrs[1] == "direct")
							{
								tracer.SetIntegrator(IntegratorType::Direct);
							}
							else if (substrs[1] == "raytracer")
							{
								tracer.SetIntegrator(IntegratorType::Raytracer);
							}
							else if (substrs[1] == "pathtracer")
							{
								tracer.SetIntegrator(IntegratorType::Pathtracer);
							}
						}
						else if (leadingStr.Is("size"))
						{
							int width = atoi(substrs[1].c_str());
							int height = atoi(substrs[2].c_str());
							tracer.SetSize(width, height);
						}
						else if (leadingStr.Is("vertex"))
						{
							vertices.push_back(GCPoint(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str())));
						}
						else if (leadingStr.Is("tri"))
						{
							GCPoint a = vertices[atoi(substrs[1].c_str())];
							GCPoint b = vertices[atoi(substrs[2].c_str())];
							GCPoint c = vertices[atoi(substrs[3].c_str())];
							CTriangle* new_triangle = new CTriangle(a, b, c);
							new_triangle->SetMaterialCache(tracer.GetMaterialCachePtr());
							new_triangle->SetMaterial(currMaterial);
							new_triangle->ApplyTransform(transform_stack.top());
							tracer.AddShape((IShape*)new_triangle);
						}
						else if (leadingStr.Is("emission"))
						{
							currMaterial.emission = GCVector(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str()));
						}
						else if (leadingStr.Is("ambient"))
						{
							currMaterial.ambient = GCVector(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str()));
						}
						else if (leadingStr.Is("specular"))
						{
							currMaterial.specular = GCVector(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str()));
						}
						else if (leadingStr.Is("diffuse"))
						{
							currMaterial.diffuse = GCVector(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str()));
						}
						else if (leadingStr.Is("shininess"))
						{
							currMaterial.shininess = atof(substrs[1].c_str());
						}
						else if (leadingStr.Is("roughness"))
						{
							currMaterial.roughness = atof(substrs[1].c_str());
						}
						else if (leadingStr.Is("brdf"))
						{
							if (substrs[1] == "phong")
							{
								currMaterial.brdfType = BRDFType::Phong;
							}
							else if (substrs[1] == "ggx")
							{
								currMaterial.brdfType = BRDFType::GGX;
							}
						}
						else if (leadingStr.Is("gamma"))
						{
							tracer.SetGamma(atof(substrs[1].c_str()));
						}
						else if (leadingStr.Is("attenuation"))
						{
							currAttenuation[0] = atof(substrs[1].c_str());
							currAttenuation[1] = atof(substrs[2].c_str());
							currAttenuation[2] = atof(substrs[3].c_str());
						}
						else if (leadingStr.Is("point"))
						{
							CPointLite* new_pointLite = new CPointLite(GCPoint(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str())),
								GCVector(atof(substrs[4].c_str()), atof(substrs[5].c_str()), atof(substrs[6].c_str())));
							new_pointLite->SetAttenuation(currAttenuation[0], currAttenuation[1], currAttenuation[2]);
							tracer.AddLight((ILight*)new_pointLite);
							std::cout << "Point light: (" << atof(substrs[1].c_str()) << ", " <<
								atof(substrs[2].c_str()) << ", " <<
								atof(substrs[3].c_str()) << ")" << std::endl;
						}
						else if (leadingStr.Is("quadLight"))
						{
							CPolygonalLite* new_quadLight = new CPolygonalLite(
								GCPoint(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str())),
								GCVector(atof(substrs[4].c_str()), atof(substrs[5].c_str()), atof(substrs[6].c_str())),
								GCVector(atof(substrs[7].c_str()), atof(substrs[8].c_str()), atof(substrs[9].c_str())),
								GCVector(atof(substrs[10].c_str()), atof(substrs[11].c_str()), atof(substrs[12].c_str()))
							);
							tracer.AddLight((ILight*)new_quadLight);
							std::cout << "Quad light A: " << new_quadLight->GetCorner()
								<< "AB: " << new_quadLight->GetVectorAB()
								<< "AC: " << new_quadLight->GetVectorAC()
								<< "Color: " << new_quadLight->GetColor()
								<< std::endl;
							//we need to visualize the quadlight too
							GCPoint B = new_quadLight->GetCorner() + new_quadLight->GetVectorAB();
							GCPoint C = new_quadLight->GetCorner() + new_quadLight->GetVectorAC();
							GCPoint D = new_quadLight->GetCorner() + new_quadLight->GetVectorAC() + new_quadLight->GetVectorAB();

							Material lightMaterial;
							lightMaterial.emission = new_quadLight->GetColor();
							lightMaterial.shininess = 0;
							lightMaterial.lightNum = tracer.GetLightCount() - 1;
							//Triangle 1
							CTriangle* new_triangle = new CTriangle(new_quadLight->GetCorner(), C, B);
							new_triangle->SetMaterialCache(tracer.GetMaterialCachePtr());
							new_triangle->SetMaterial(lightMaterial);
							tracer.AddShape((IShape*)new_triangle);
							//Triangle 2
							new_triangle = new CTriangle(C, D, B);
							new_triangle->SetMaterialCache(tracer.GetMaterialCachePtr());
							new_triangle->SetMaterial(lightMaterial);
							tracer.AddShape((IShape*)new_triangle);
						}
						else if (leadingStr.Is("directional"))
						{
							CDirectionalLite* new_directionalLite = new CDirectionalLite(GCVector(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str())),
								GCVector(atof(substrs[4].c_str()), atof(substrs[5].c_str()), atof(substrs[6].c_str())));
							tracer.AddLight((ILight*)new_directionalLite);
							std::cout << "Directional light: (" << atof(substrs[1].c_str()) << ", " <<
								atof(substrs[2].c_str()) << ", " <<
								atof(substrs[3].c_str()) << ")" << std::endl;
						}
						else if (leadingStr.Is("sphere"))
						{
							GCPoint center(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str()));
							double radius = atof(substrs[4].c_str());
							CSphere* new_sphere = new CSphere(center, radius);
							new_sphere->SetMaterialCache(tracer.GetMaterialCachePtr());
							new_sphere->SetMaterial(currMaterial);
							new_sphere->SetTransform(transform_stack.top());
							tracer.AddShape((IShape*)new_sphere);
						}
						else if (leadingStr.Is("maxdepth"))
						{
							tracer.SetMaxDepth(atoi(substrs[1].c_str()));
						}
						else if (leadingStr.Is("output"))
						{
							tracer.SetFilename(substrs[1]);
						}
						else if (leadingStr.Is("pushTransform"))
						{
							PushTransform();
						}
						else if (leadingStr.Is("popTransform"))
						{
							PopTransform();
						}
						else if (leadingStr.Is("translate"))
						{
							RightMultiply(GCMatrix::createTranslation(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str())));
						}
						else if (leadingStr.Is("rotate"))
						{
							RightMultiply(GCMatrix::createRotation(GCDegree(atof(substrs[4].c_str())), GCVector(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str()))));
						}
						else if (leadingStr.Is("scale"))
						{
							RightMultiply(GCMatrix::createScale(atof(substrs[1].c_str()), atof(substrs[2].c_str()), atof(substrs[3].c_str())));
						}
						else if (leadingStr.Is("lightsamples"))
						{
							tracer.SetNumSamples(atoi(substrs[1].c_str()));
						}
						else if (leadingStr.Is("spp"))
						{
							tracer.SetSamplesPerPixel(atoi(substrs[1].c_str()));
						}
						else if (leadingStr.Is("lightstratify"))
						{
							if (substrs[1] == "on")
							{
								tracer.EnableStratification(true);
							}
							else
							{
								tracer.EnableStratification(false);
							}
						}
						else if (leadingStr.Is("nexteventestimation"))
						{
							if (substrs[1] == "on")
							{
								tracer.SetNEE(NEEMode::On);
							}
							else if(substrs[1] == "off")
							{
								tracer.SetNEE(NEEMode::Off);
							}
							else if (substrs[1] == "mis")
							{
								tracer.SetNEE(NEEMode::MIS);
							}
						}
						else if (leadingStr.Is("russianroulette"))
						{
							if (substrs[1] == "on")
							{
								tracer.EnableRR(true);
							}
							else
							{
								tracer.EnableRR(false);
							}
						}
						else if (leadingStr.Is("importancesampling"))
						{
							if (substrs[1] == "hemisphere")
							{
								tracer.SetImportanceSampling(ImportanceSamplingType::Hemisphere);
							}
							else if (substrs[1] == "cosine")
							{
								tracer.SetImportanceSampling(ImportanceSamplingType::Cosine);
							}
							else if (substrs[1] == "brdf")
							{
								tracer.SetImportanceSampling(ImportanceSamplingType::BRDF);
							}
						}	
					}

				}
			}
			myfile.close();
		}
	} //End of argc else
	//tracer.IterateRays();
	tracer.Render();
	return 0;
}

