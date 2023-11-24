#include "DrawTester.hpp"

#include <xcb/xcb.h>

#include <cmath>

int main()
{
	DrawTester tester;

	tester.onCreateVertexBuffers([](DrawTester &tester) {
		struct Vertex
		{
			float position[3];
			float color[3];
		};

		Vertex vertexBufferData[] = {
			{ { -0.5f, -0.5f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
			{ { 0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
			{ { 0.0f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
		};

		std::vector<vk::VertexInputAttributeDescription> inputAttributes;
		inputAttributes.push_back(vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)));
		inputAttributes.push_back(vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)));

		tester.addVertexBuffer(vertexBufferData, sizeof(vertexBufferData), std::move(inputAttributes));
	});

	tester.onCreateVertexShader([](DrawTester &tester) {
		const char *vertexShader = R"(#version 310 es
			layout(location = 0) in vec3 inPos;
			layout(location = 1) in vec3 inColor;

			layout(location = 0) out vec3 outColor;

			layout(binding = 0) uniform UniformBufferObject { mat4 model; } ubo;

			void main()
			{
				outColor = inColor;
				gl_Position = vec4(inPos.xyz, 1.0);
			})";

		return tester.createShaderModule(vertexShader, EShLanguage::EShLangVertex);
	});

	tester.onCreateFragmentShader([](DrawTester &tester) {
		const char *fragmentShader = R"(#version 310 es
			precision highp float;

			layout(location = 0) in vec3 inColor;

			layout(location = 0) out vec4 outColor;

			void main()
			{
				outColor = vec4(inColor, 1.0);
			})";

		return tester.createShaderModule(fragmentShader, EShLanguage::EShLangFragment);
	});

	tester.onCreateDescriptorSetLayouts([](DrawTester &tester) -> std::vector<vk::DescriptorSetLayoutBinding> {
		vk::DescriptorSetLayoutBinding uniformLayoutBinding;
		uniformLayoutBinding.binding = 0;
		uniformLayoutBinding.descriptorCount = 1;
		uniformLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
		uniformLayoutBinding.pImmutableSamplers = nullptr;
		uniformLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;

		tester.addUniformBuffer(sizeof(float) * 16);

		return { uniformLayoutBinding };
	});

	tester.onUpdateDescriptorSet([](DrawTester &tester, vk::CommandPool &commandPool, vk::DescriptorSet &descriptorSet) {
		static unsigned int tick = 0;

		auto &device = tester.getDevice();
		auto &physicalDevice = tester.getPhysicalDevice();
		auto &queue = tester.getQueue();

		auto &uniform = tester.getUniformBufferById(0);

		float model[] = { 0.2f, 0.0f, 0.0f, 0.0f,
			              0.0f, 0.2f, 0.0f, 0.0f,
			              0.0f, 0.0f, 1.0f, 0.0f,
			              0.0f, 0.0f, 0.0f, 1.0f };

		void *data = device.mapMemory(uniform.memory, 0, VK_WHOLE_SIZE);
		memcpy(data, model, sizeof(model));
		device.unmapMemory(uniform.memory);

		vk::DescriptorBufferInfo bufferInfo;
		bufferInfo.buffer = uniform.buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(model);

		std::array<vk::WriteDescriptorSet, 1> descriptorWrites = {};

		descriptorWrites[0].dstSet = descriptorSet;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &bufferInfo;

		device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	});

	tester.initialize();

	for(;;)
	{
		tester.renderFrame();
	}

	return 0;
}
