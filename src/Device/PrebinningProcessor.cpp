#include "PrebinningProcessor.hpp"

#include "Polygon.hpp"
#include "Primitive.hpp"
#include "Renderer.hpp"
#include "Pipeline/Constants.hpp"
#include "Pipeline/PrebinningRoutine.hpp"
#include "System/Debug.hpp"

#include <cstring>

namespace sw {

PrebinningProcessor::PrebinningProcessor()
    : routine_(nullptr)
{
}

PrebinningProcessor::RoutineType PrebinningProcessor::routine()
{
	if(!routine_)
	{
		PrebinningRoutine *generator = new PrebinningRoutine();
		generator->generate();
		routine_ = std::make_unique<RoutineType>(generator->getRoutine());
		delete generator;
	}

	return *routine_;
}

}  // namespace sw
