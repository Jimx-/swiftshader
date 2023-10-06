#include "BinningProcessor.hpp"

#include "Primitive.hpp"
#include "Renderer.hpp"
#include "Pipeline/BinningRoutine.hpp"
#include "Pipeline/Constants.hpp"
#include "System/Debug.hpp"

#include <cstring>

namespace sw {

BinningProcessor::BinningProcessor()
    : routine_(nullptr)
{
}

BinningProcessor::RoutineType BinningProcessor::routine()
{
	if(!routine_)
	{
		BinningRoutine *generator = new BinningRoutine();
		generator->generate();
		routine_ = std::make_unique<RoutineType>(generator->getRoutine());
		delete generator;
	}

	return *routine_;
}

}  // namespace sw
