#ifndef sw_BinningRoutine_hpp
#define sw_BinningRoutine_hpp

#include "Device/BinningProcessor.hpp"
#include "Reactor/Reactor.hpp"

namespace sw {

class Context;

class BinningRoutine
{
public:
	virtual ~BinningRoutine();

	void generate();
	BinningFunction::RoutineType getRoutine();

private:
	BinningFunction::RoutineType routine;
};

}  // namespace sw

#endif  // sw_BinningRoutine_hpp
