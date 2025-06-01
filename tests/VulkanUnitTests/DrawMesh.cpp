#include "DrawTester.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <xcb/xcb.h>

#include <cmath>

#include <fstream>

struct Object
{
	std::vector<float> vertices;
	std::vector<float> normals;
	std::vector<int> faces;

	void get_vertex_buffer(float *vbuf, size_t count)
	{
		int start = 0;
		for(int i = start; i < start + count; i++)
		{
			for(int j = 0; j < 3; j++)
			{
				int v = faces[6 * i + j * 2 + 0] - 1;
				int vn = faces[6 * i + j * 2 + 1] - 1;

				*vbuf++ = vertices[3 * v + 0];
				*vbuf++ = vertices[3 * v + 1];
				*vbuf++ = vertices[3 * v + 2];
				*vbuf++ = normals[3 * vn + 0];
				*vbuf++ = normals[3 * vn + 1];
				*vbuf++ = normals[3 * vn + 2];
				*vbuf++ = 1.0f;
				*vbuf++ = 1.0f;
				*vbuf++ = 1.0f;
				// printf("%f %f %f\n", vertices[3 * vn + 0], vertices[3 * vn + 1], vertices[3 * vn + 2]);
			}
		}
	}
};

void read_obj(const std::string &name, Object &obj)
{
	std::ifstream ifs(name, std::ios::in);
	char c;

	obj.vertices.clear();
	obj.normals.clear();
	obj.faces.clear();

	int f = 0;
	while(!ifs.eof())
	{
		ifs >> c;

		if(c == '#')
		{
			std::string line;
			std::getline(ifs, line);
			printf("Skipping line %s\n", line.c_str());
		}
		else if(c == 'v')
		{
			double v1, v2, v3;
			ifs.get(c);
			ifs >> v1 >> v2 >> v3;

			if(c == 'n')
			{
				obj.normals.push_back(v1);
				obj.normals.push_back(v2);
				obj.normals.push_back(v3);
			}
			else
			{
				v1 *= 2.f;
				v2 *= 2.f;
				v3 *= 2.f;
				obj.vertices.push_back(v1);
				obj.vertices.push_back(v2);
				obj.vertices.push_back(v3);
			}
		}
		else if(c == 'f')
		{
			int v0, vn0, v1, vn1, v2, vn2;
			ifs >> v0 >> c >> c >> vn0 >> v1 >> c >> c >> vn1 >> v2 >> c >> c >> vn2;
			f++;

			obj.faces.push_back(v0);
			obj.faces.push_back(vn0);
			obj.faces.push_back(v1);
			obj.faces.push_back(vn1);
			obj.faces.push_back(v2);
			obj.faces.push_back(vn2);
		}
	}
	printf("Faces: %lu\n", obj.faces.size());
}

int main()
{
	DrawTester tester;
	Object obj;
	std::vector<float> vbuf;
	size_t fcount = 4968;
	// size_t fcount = 128;

	read_obj("bunny.obj", obj);
	vbuf.resize(27 * fcount);
	obj.get_vertex_buffer(&vbuf[0], fcount);

	tester.onCreateVertexBuffers([&vbuf](DrawTester &tester) {
		struct Vertex
		{
			float position[3];
			float normal[3];
			float color[3];
		};

		Vertex vertexBufferData[] = {
			{ { -0.5f, -0.5f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
			{ { 0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
			{ { 0.0f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
		};

		std::vector<vk::VertexInputAttributeDescription> inputAttributes;
		inputAttributes.push_back(vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)));
		inputAttributes.push_back(vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)));
		inputAttributes.push_back(vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)));

		tester.addVertexBuffer((Vertex *)&vbuf[0], sizeof(float) * vbuf.size(), std::move(inputAttributes));
	});

	tester.onCreateVertexShader([](DrawTester &tester) {
		const char *vertexShader = R"(#version 310 es
            layout(location = 0) in vec3 inPos;
            layout(location = 1) in vec3 inNormal;
            layout(location = 2) in vec3 inColor;

            layout(location = 0) out vec3 outColor;
            layout(location = 1) out vec3 outPos;
            layout(location = 2) out vec3 outNormal;

            layout(binding = 0) uniform UniformBufferObject { mat4 mvp; mat4 model; } ubo;

            void main()
            {
                outColor = inColor;
                outNormal = inNormal;
                gl_Position = ubo.mvp * vec4(inPos.xyz, 1.0);
                outPos = vec3(ubo.model * vec4(inPos.xyz, 1.0f));
            })";

		return tester.createShaderModule(vertexShader, EShLanguage::EShLangVertex);
	});

	tester.onCreateFragmentShader([](DrawTester &tester) {
		const char *fragmentShader = R"(#version 310 es
            precision highp float;

            layout(location = 0) in vec3 inColor;
            layout(location = 1) in vec3 inPos;
            layout(location = 2) in vec3 inNormal;

            layout(location = 0) out vec4 outColor;

            layout(binding = 1) uniform UniformBufferObject { vec3 lightPos; } ubo;

            void main()
            {
                vec3 norm = normalize(inNormal);
                vec3 lightDir = normalize(ubo.lightPos - inPos);
                float diff = max(dot(norm, lightDir), 0.0);
                vec3 diffuse = diff * inColor;
                outColor = vec4(diffuse, 1.0);
            })";

		return tester.createShaderModule(fragmentShader, EShLanguage::EShLangFragment);
	});

	tester.onCreateDescriptorSetLayouts([](DrawTester &tester) -> std::vector<vk::DescriptorSetLayoutBinding> {
		std::vector<vk::DescriptorSetLayoutBinding> uniformLayoutBinding(2);
		uniformLayoutBinding[0].binding = 0;
		uniformLayoutBinding[0].descriptorCount = 1;
		uniformLayoutBinding[0].descriptorType = vk::DescriptorType::eUniformBuffer;
		uniformLayoutBinding[0].pImmutableSamplers = nullptr;
		uniformLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eVertex;

		uniformLayoutBinding[1].binding = 1;
		uniformLayoutBinding[1].descriptorCount = 1;
		uniformLayoutBinding[1].descriptorType = vk::DescriptorType::eUniformBuffer;
		uniformLayoutBinding[1].pImmutableSamplers = nullptr;
		uniformLayoutBinding[1].stageFlags = vk::ShaderStageFlagBits::eFragment;

		tester.addUniformBuffer(sizeof(float) * 32);
		tester.addUniformBuffer(sizeof(float) * 3);

		return uniformLayoutBinding;
	});

	tester.onUpdateDescriptorSet([](DrawTester &tester, vk::CommandPool &commandPool, vk::DescriptorSet &descriptorSet) {
		auto &device = tester.getDevice();
		auto &physicalDevice = tester.getPhysicalDevice();
		auto &queue = tester.getQueue();

		auto &uniform = tester.getUniformBufferById(0);
		auto &uniformFrag = tester.getUniformBufferById(1);

		vk::DescriptorBufferInfo bufferInfo[2] = {};
		bufferInfo[0].buffer = uniform.buffer;
		bufferInfo[0].offset = 0;
		bufferInfo[0].range = sizeof(float) * 32;

		bufferInfo[1].buffer = uniformFrag.buffer;
		bufferInfo[1].offset = 0;
		bufferInfo[1].range = sizeof(float) * 3;

		std::array<vk::WriteDescriptorSet, 2> descriptorWrites = {};
		descriptorWrites[0].dstSet = descriptorSet;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &bufferInfo[0];

		descriptorWrites[1].dstSet = descriptorSet;
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = vk::DescriptorType::eUniformBuffer;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &bufferInfo[1];

		device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	});

	tester.initialize();

	int ticks = 0;

	// for(;;)
	for(int i = 0; i < 1; i++)
	{
		auto &device = tester.getDevice();
		auto &uniform = tester.getUniformBufferById(0);
		auto &uniformFrag = tester.getUniformBufferById(1);

		ticks++;

		glm::mat4 proj =
		    glm::perspective(glm::pi<float>() * 0.25f, 4.0f / 3.0f, 0.1f, 100.f);
		glm::mat4 view =
		    glm::lookAt(glm::vec3(0.f, 0.f, 0.5f), glm::vec3(0.f, 0.f, 0.f),
		                glm::vec3(0.f, -1.f, 0.f));
		glm::mat4 model = glm::translate(glm::rotate(glm::mat4(1.0f), glm::radians(180.f), glm::vec3(0.f, 1.f, 0.f)), glm::vec3(0.0f, -0.2f, 0.0f));
		// glm::mat4 model = glm::mat4(1.0f);

		glm::mat4 mvp = proj * view * model;
		// glm::mat4 mvp = glm::mat4(1.0f);

		std::vector<float> light_pos{ 0.0f, -0.3f, -0.5f };

		float *data = (float *)device.mapMemory(uniform.memory, 0, VK_WHOLE_SIZE);
		memcpy(data, &mvp[0][0], sizeof(float) * 16);
		memcpy(data + 16, &model[0][0], sizeof(float) * 16);
		device.unmapMemory(uniform.memory);

		void *light_data = device.mapMemory(uniformFrag.memory, 0, VK_WHOLE_SIZE);
		memcpy(light_data, &light_pos[0], sizeof(float) * 3);
		device.unmapMemory(uniformFrag.memory);

		tester.renderFrame();
	}

	return 0;
}
