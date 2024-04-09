R"(
#include "generated/platform.h"
#include <groom/intrinsics.h>
#include <groom/spawn.h>
#include <stddef.h>
#include <stdint.h>

struct binning_arg
{
	void *device;
	void *primitive;
	unsigned int count;
	unsigned int *primCount;
	void *tile;
	uint8_t* primMask;
	void *draw;
	unsigned int numTiles;
};

void __jit_function(void *device, void *primitive, unsigned int count, unsigned int *primCount, void *tile, uint8_t* primMask, void *draw, unsigned int index);

void binning_kernel(int task_id, void *arg)
{
	struct binning_arg *barg = (struct binning_arg *)arg;

	__jit_function(barg->device, barg->primitive, barg->count, barg->primCount, barg->tile, barg->primMask, barg->draw, task_id);
}

int main()
{
	void *arg;
	struct binning_arg *barg;

	arg = (void *)(*(volatile uint32_t *)CTRL_SCRATCH_ADDRESS);
	barg = (struct binning_arg *)arg;

	gpu_spawn_tasks(barg->numTiles, binning_kernel, arg);

	return 0;
}
)"
