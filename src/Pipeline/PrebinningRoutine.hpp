#ifndef sw_PrebinningRoutine_hpp
#define sw_PrebinningRoutine_hpp

#include "Device/PrebinningProcessor.hpp"
#include "Reactor/Reactor.hpp"

namespace sw {

class Context;

class PrebinningRoutine
{
public:
	virtual ~PrebinningRoutine();

	void generate();
	PrebinningFunction::RoutineType getRoutine();

private:
	PrebinningFunction::RoutineType routine;
};

}  // namespace sw

#endif  // sw_SetupRoutine_hpp
