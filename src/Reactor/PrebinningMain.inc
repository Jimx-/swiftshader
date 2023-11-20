R"(
#include "generated/platform.h"
#include <groom/intrinsics.h>
#include <groom/spawn.h>
#include <stddef.h>
#include <stdint.h>

struct prebinning_arg
{
	void *device;
	void *primitive;
	unsigned int count;
	unsigned int *primCount;
	void *draw;
	unsigned int numTiles;
};

void __jit_function(void *device, void *primitive, unsigned int count, unsigned int *primCount, void *draw, unsigned int index);

void prebinning_kernel(int task_id, void *arg)
{
	struct prebinning_arg *parg = (struct prebinning_arg *)arg;

	__jit_function(parg->device, parg->primitive, parg->count, parg->primCount, parg->draw, task_id);
}

int main()
{
	void *arg;
	struct prebinning_arg *parg;

	arg = (void *)(*(volatile uint32_t *)CTRL_SCRATCH_ADDRESS);
	parg = (struct prebinning_arg *)arg;

	gpu_spawn_tasks(parg->numTiles, prebinning_kernel, arg);

	return 0;
}
)"
