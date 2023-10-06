#include "BinningRoutine.hpp"

#include "Constants.hpp"
#include "Device/Primitive.hpp"
#include "Device/Renderer.hpp"
#include "Reactor/Reactor.hpp"
#include "Vulkan/VkDevice.hpp"

namespace sw {

BinningRoutine::~BinningRoutine()
{
}

void BinningRoutine::generate()
{
	BinningFunction function;
	{
		Pointer<Byte> device(function.Arg<0>());
		Pointer<Byte> primitive(function.Arg<1>());
		UInt count(function.Arg<2>());
		Pointer<UInt> primCount(function.Arg<3>());
		Pointer<Byte> tile(function.Arg<4>());
		Pointer<Byte> data(function.Arg<5>());
		UInt index(function.Arg<6>());

		UInt prefixSum = *Pointer<UInt>(Pointer<Byte>(primCount) + sizeof(uint32_t) * index);
		Pointer<Byte> curTile(tile + index * sizeof(Tile) + prefixSum * sizeof(unsigned int));
		Pointer<UInt> pidList(curTile + OFFSET(Tile, pid));

		UInt tileSizeLog2 = *Pointer<UInt>(data + OFFSET(DrawData, tileSizeLog2));
		UInt tileStride = *Pointer<UInt>(data + OFFSET(DrawData, tileStride));

		Int startX = (index % tileStride) << tileSizeLog2;
		Int startY = (index / tileStride) << tileSizeLog2;

		Int4 X;
		Int4 Y;

		X.x = startX;
		X.y = startX + Int(1 << tileSizeLog2);
		X.z = startX;
		X.w = startX + Int(1 << tileSizeLog2);
		Y.x = startY;
		Y.y = startY;
		Y.z = startY + Int(1 << tileSizeLog2);
		Y.w = startY + Int(1 << tileSizeLog2);

		UInt hit = 0;
		UInt i = 0;
		Do
		{
			Pointer<Byte> curPrim(primitive + i * sizeof(Primitive));
			Int edge[3];
			Int outside[3];

			for(int e = 0; e < 3; e++)
			{
				Int4 sample;

				edge[0] = *Pointer<Int>(curPrim + OFFSET(Primitive, edge[e].A));
				edge[1] = *Pointer<Int>(curPrim + OFFSET(Primitive, edge[e].B));
				edge[2] = *Pointer<Int>(curPrim + OFFSET(Primitive, edge[e].C));

				sample = Int4(edge[0]) * X + Int4(edge[1]) * Y + Int4(edge[2]);
				outside[e] = SignMask(sample);
			}

			If((outside[0] != 0xf) && (outside[1] != 0xf) && (outside[2] != 0xf))
			{
				hit++;
				*pidList = i;
				pidList = Pointer<UInt>(Pointer<Byte>(pidList) + sizeof(unsigned int));
			}

			i++;
		}
		Until(i >= count);

		*Pointer<UShort>(Pointer<Byte>(curTile) + OFFSET(Tile, startX)) = UShort(startX);
		*Pointer<UShort>(Pointer<Byte>(curTile) + OFFSET(Tile, startY)) = UShort(startY);
		*Pointer<UShort>(Pointer<Byte>(curTile) + OFFSET(Tile, count)) = UShort(hit);
		*Pointer<UShort>(Pointer<Byte>(curTile) + OFFSET(Tile, level)) = UShort(tileSizeLog2);

		Return();
	}

	routine = function("BinningRoutine");
}

BinningFunction::RoutineType BinningRoutine::getRoutine()
{
	return routine;
}

}  // namespace sw
