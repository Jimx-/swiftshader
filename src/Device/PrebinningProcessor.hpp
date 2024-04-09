#ifndef sw_PrebinningProcessor_hpp
#define sw_PrebinningProcessor_hpp

#include "Context.hpp"
#include "RoutineCache.hpp"
#include "System/Types.hpp"

#include <memory>

namespace sw {

struct Primitive;
struct DrawData;

using PrebinningFunction = FunctionT<void(const vk::Device *device, Primitive *primitive, unsigned int count, unsigned int *primCount, uint8_t *primMask, const DrawData *draw, unsigned int index)>;

class PrebinningProcessor
{
public:
	using RoutineType = PrebinningFunction::RoutineType;

	PrebinningProcessor();

	RoutineType routine();

private:
	std::unique_ptr<RoutineType> routine_;
};

}  // namespace sw

#endif
