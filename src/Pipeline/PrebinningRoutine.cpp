#include "PrebinningRoutine.hpp"

#include "Constants.hpp"
#include "Device/Primitive.hpp"
#include "Device/Renderer.hpp"
#include "Reactor/Reactor.hpp"
#include "Vulkan/VkDevice.hpp"

namespace sw {

PrebinningRoutine::~PrebinningRoutine()
{
}

void PrebinningRoutine::generate()
{
	PrebinningFunction function;
	{
		Pointer<Byte> device(function.Arg<0>());
		Pointer<Byte> primitive(function.Arg<1>());
		Int count(function.Arg<2>());
		Pointer<UInt> primCount(function.Arg<3>());
		Pointer<Byte> primMask(function.Arg<4>());
		Pointer<Byte> data(function.Arg<5>());
		UInt index(function.Arg<6>());

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
		Int i = 0;
		Do
		{
			Pointer<Byte> mask(primMask + i);

			If(*mask != Byte(0))
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
				}
			}

			i++;
		}
		Until(i >= count);

		*Pointer<UInt>(Pointer<Byte>(primCount) + sizeof(unsigned int) * index) = hit;

		Return();
	}

	routine = function("PrebinningRoutine");
}

PrebinningFunction::RoutineType PrebinningRoutine::getRoutine()
{
	return routine;
}

}  // namespace sw
