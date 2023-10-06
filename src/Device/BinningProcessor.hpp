#ifndef sw_BinningProcessor_hpp
#define sw_BinningProcessor_hpp

#include "Context.hpp"
#include "RoutineCache.hpp"
#include "System/Types.hpp"

#include <memory>

namespace sw {

struct Primitive;
struct Tile;
struct DrawData;

using BinningFunction = FunctionT<void(const vk::Device *device, Primitive *primitive, unsigned int count, unsigned int *primCount, Tile *tile, const DrawData *draw, unsigned int index)>;

class BinningProcessor
{
public:
	using RoutineType = BinningFunction::RoutineType;

	BinningProcessor();

	RoutineType routine();

private:
	std::unique_ptr<RoutineType> routine_;
};

}  // namespace sw

#endif
