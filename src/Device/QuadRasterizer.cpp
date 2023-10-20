// Copyright 2016 The SwiftShader Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "QuadRasterizer.hpp"

#include "Primitive.hpp"
#include "Renderer.hpp"
#include "Pipeline/Constants.hpp"
#include "System/Debug.hpp"
#include "System/Math.hpp"
#include "Vulkan/VkDevice.hpp"

namespace sw {

QuadRasterizer::QuadRasterizer(const PixelProcessor::State &state, const SpirvShader *spirvShader)
    : state(state)
    , spirvShader{ spirvShader }
{
}

QuadRasterizer::~QuadRasterizer()
{
}

void QuadRasterizer::generate()
{
	constants = device + OFFSET(vk::Device, constants);
	occlusion = 0;

	Pointer<Byte> primitives = primitive;

	primitive = primitives + pid * sizeof(Primitive) * state.multiSampleCount;

	rasterize(x, y, mask);

	Return();
}

void QuadRasterizer::rasterize(Int &x, Int &y, UInt &mask)
{
	Pointer<Byte> cBuffer[MAX_COLOR_BUFFERS];
	Pointer<Byte> zBuffer;
	Pointer<Byte> sBuffer;

	for(int index = 0; index < MAX_COLOR_BUFFERS; index++)
	{
		if(state.colorWriteActive(index))
		{
			cBuffer[index] = *Pointer<Pointer<Byte>>(data + OFFSET(DrawData, colorBuffer[index])) + y * *Pointer<Int>(data + OFFSET(DrawData, colorPitchB[index]));
		}
	}

	if(state.depthTestActive || state.depthBoundsTestActive)
	{
		zBuffer = *Pointer<Pointer<Byte>>(data + OFFSET(DrawData, depthBuffer)) + y * *Pointer<Int>(data + OFFSET(DrawData, depthPitchB));
	}

	if(state.stencilActive)
	{
		sBuffer = *Pointer<Pointer<Byte>>(data + OFFSET(DrawData, stencilBuffer)) + y * *Pointer<Int>(data + OFFSET(DrawData, stencilPitchB));
	}

	Int cMask[4];

	cMask[0] = mask;
	cMask[1] = mask;
	cMask[2] = mask;
	cMask[3] = mask;

	quad(cBuffer, zBuffer, sBuffer, cMask, x, y);
}

SIMD::Float QuadRasterizer::interpolate(SIMD::Float &x, SIMD::Float &y, SIMD::Float &rhw, Pointer<Byte> planeEquation, bool flat, bool perspective)
{
	SIMD::Float D = y * SIMD::Float(*Pointer<Float>(planeEquation + OFFSET(PlaneEquation, B))) +
	                SIMD::Float(*Pointer<Float>(planeEquation + OFFSET(PlaneEquation, C)));

	if(flat)
	{
		return D;
	}

	SIMD::Float interpolant = MulAdd(x, SIMD::Float(*Pointer<Float>(planeEquation + OFFSET(PlaneEquation, A))), D);

	if(perspective)
	{
		interpolant *= rhw;
	}

	return interpolant;
}

bool QuadRasterizer::interpolateZ() const
{
	return state.depthTestActive || (spirvShader && spirvShader->hasBuiltinInput(spv::BuiltInFragCoord));
}

bool QuadRasterizer::interpolateW() const
{
	// Note: could optimize cases where there is a fragment shader but it has no
	// perspective-correct inputs, but that's vanishingly rare.
	return spirvShader != nullptr;
}

}  // namespace sw
