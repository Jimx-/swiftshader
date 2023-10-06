#ifndef sw_Tile_hpp
#define sw_Tile_hpp

#include "Device/Config.hpp"
#include "System/Types.hpp"

namespace sw {

struct Tile
{
	unsigned short startX;
	unsigned short startY;
	unsigned short count;
	unsigned short level;
	unsigned int pid[];
};

}  // namespace sw

#endif
