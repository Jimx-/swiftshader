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

#include "TileRasterizer.hpp"

#include "Primitive.hpp"
#include "Renderer.hpp"
#include "Pipeline/Constants.hpp"
#include "System/Debug.hpp"
#include "System/Math.hpp"
#include "Vulkan/VkDevice.hpp"

namespace sw {

TileRasterizer::TileRasterizer(const PixelProcessor::State &state, const SpirvShader *spirvShader)
    : state(state)
    , spirvShader{ spirvShader }
{
}

TileRasterizer::~TileRasterizer()
{
}

void TileRasterizer::generate()
{
	constants = device + OFFSET(vk::Device, constants);
	occlusion = 0;

	Pointer<Byte> primitives = primitive;

	Do
	{
		UInt primCount = UInt(*Pointer<UShort>(tile + OFFSET(Tile, count)));
		Pointer<UInt> pidList = Pointer<UInt>(tile + OFFSET(Tile, pid));

		If(primCount > 0)
		{
			UInt i = 0;
			Do
			{
				UInt pid = *pidList;
				primitive = primitives + pid * sizeof(Primitive) * state.multiSampleCount;

				scanTile(tile, tileQueue);

				pidList = Pointer<UInt>(Pointer<Byte>(pidList) + sizeof(unsigned int));
				i++;
			}
			Until(i >= primCount);
		}

		tile += sizeof(Tile) + UInt(primCount) * sizeof(unsigned int);
		count--;
	}
	Until(count == 0);

	UInt cluster = 0;
	if(state.occlusionEnabled)
	{
		UInt clusterOcclusion = *Pointer<UInt>(data + OFFSET(DrawData, occlusion) + 4 * cluster);
		clusterOcclusion += occlusion;
		*Pointer<UInt>(data + OFFSET(DrawData, occlusion) + 4 * cluster) = clusterOcclusion;
	}

	Return();
}

void TileRasterizer::scanTile(Pointer<Byte> topTile, Pointer<Byte> tileQueue)
{
	UInt head = 0;
	UInt tail = 1;

	*Pointer<UShort>(tileQueue + OFFSET(Tile, startX)) = *Pointer<UShort>(topTile + OFFSET(Tile, startX));
	*Pointer<UShort>(tileQueue + OFFSET(Tile, startY)) = *Pointer<UShort>(topTile + OFFSET(Tile, startY));
	*Pointer<UShort>(tileQueue + OFFSET(Tile, level)) = *Pointer<UShort>(topTile + OFFSET(Tile, level));

	Do
	{
		Pointer<Byte> tile(tileQueue + sizeof(Tile) * head);
		head++;

		Int startX = Int(*Pointer<UShort>(tile + OFFSET(Tile, startX)));
		Int startY = Int(*Pointer<UShort>(tile + OFFSET(Tile, startY)));
		Int level = Int(*Pointer<UShort>(tile + OFFSET(Tile, level)));

		If(level > 1)
		{
			Int P[9][2];

			for(int i = 0; i < 9; i++)
			{
				P[i][0] = startX + (i % 3) * (1 << (level - 1));
				P[i][1] = startY + (i / 3) * (1 << (level - 1));
			}

			int subTileIndex[4][4] = { { 0, 1, 4, 3 },
				                       { 1, 2, 5, 4 },
				                       { 3, 4, 7, 6 },
				                       { 4, 5, 8, 7 } };

			for(int i = 0; i < 4; i++)
			{
				Int4 X;
				Int4 Y;

				X.x = P[subTileIndex[i][0]][0];
				X.y = P[subTileIndex[i][1]][0];
				X.z = P[subTileIndex[i][2]][0];
				X.w = P[subTileIndex[i][3]][0];
				Y.x = P[subTileIndex[i][0]][1];
				Y.y = P[subTileIndex[i][1]][1];
				Y.z = P[subTileIndex[i][2]][1];
				Y.w = P[subTileIndex[i][3]][1];

				Int edge[3];
				Int outside[3];

				for(int e = 0; e < 3; e++)
				{
					Int4 sample;

					edge[0] = *Pointer<Int>(primitive + OFFSET(Primitive, edge[e].A));
					edge[1] = *Pointer<Int>(primitive + OFFSET(Primitive, edge[e].B));
					edge[2] = *Pointer<Int>(primitive + OFFSET(Primitive, edge[e].C));

					sample = Int4(edge[0]) * X + Int4(edge[1]) * Y + Int4(edge[2]);
					outside[e] = SignMask(sample);
				}

				If((outside[0] != 0xf) && (outside[1] != 0xf) && (outside[2] != 0xf))
				{
					Pointer<Byte> newTile(tileQueue + sizeof(Tile) * tail);

					*Pointer<UShort>(newTile + OFFSET(Tile, startX)) = UShort(startX + ((i & 1) << (level - 1)));
					*Pointer<UShort>(newTile + OFFSET(Tile, startY)) = UShort(startY + ((i >> 1) << (level - 1)));
					*Pointer<UShort>(newTile + OFFSET(Tile, level)) = UShort(level - 1);
					tail++;
				}
			}
		}
		Else
		{
			If(startX >= *Pointer<Int>(data + OFFSET(DrawData, scissorX0)) &&
			   startX + 2 <= *Pointer<Int>(data + OFFSET(DrawData, scissorX1)) &&
			   startY >= *Pointer<Int>(data + OFFSET(DrawData, scissorY0)) &&
			   startY + 2 <= *Pointer<Int>(data + OFFSET(DrawData, scissorY1)))
			    rasterize(startX, startY);
		}
	}
	Until(head >= tail);
}

void TileRasterizer::rasterize(Int &x, Int &y)
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

	for(unsigned int q = 0; q < state.multiSampleCount; q++)
	{
		if(state.multiSampleMask & (1 << q))
		{
			Int4 X, Y;

			X.x = x;
			X.y = x + 1;
			X.z = x;
			X.w = x + 1;
			Y.x = y;
			Y.y = y;
			Y.z = y + 1;
			Y.w = y + 1;

			Int sample = 0;
			for(int e = 0; e < 3; e++)
			{
				Int edge[3];

				edge[0] = *Pointer<Int>(primitive + q * sizeof(Primitive) + OFFSET(Primitive, edge[e].A));
				edge[1] = *Pointer<Int>(primitive + q * sizeof(Primitive) + OFFSET(Primitive, edge[e].B));
				edge[2] = *Pointer<Int>(primitive + q * sizeof(Primitive) + OFFSET(Primitive, edge[e].C));

				sample |= SignMask(Int4(edge[0]) * X + Int4(edge[1]) * Y + Int4(edge[2]));
			}

			cMask[q] = ~sample & 0x0000000F;
		}
	}

	quad(cBuffer, zBuffer, sBuffer, cMask, x, y);
}

SIMD::Float TileRasterizer::interpolate(SIMD::Float &x, SIMD::Float &y, SIMD::Float &rhw, Pointer<Byte> planeEquation, bool flat, bool perspective)
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

bool TileRasterizer::interpolateZ() const
{
	return state.depthTestActive || (spirvShader && spirvShader->hasBuiltinInput(spv::BuiltInFragCoord));
}

bool TileRasterizer::interpolateW() const
{
	// Note: could optimize cases where there is a fragment shader but it has no
	// perspective-correct inputs, but that's vanishingly rare.
	return spirvShader != nullptr;
}

}  // namespace sw
