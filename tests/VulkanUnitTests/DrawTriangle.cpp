#include "DrawTester.hpp"

#include <xcb/xcb.h>

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
			{ { 0.5f, 0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
			{ { -0.5f, 0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f } },
			{ { 0.0f, -0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f } }
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

	tester.initialize();

	for(;;)
	{
		tester.renderFrame();
	}

	return 0;
}
