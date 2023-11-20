R"(
#include "generated/platform.h"
#include <groom/intrinsics.h>
#include <groom/spawn.h>
#include <stddef.h>
#include <stdint.h>

#define FLT_EPSILON 1.19209290E-07F
#define EPS FLT_EPSILON
static const float toint = 1 / EPS;

struct pixel_arg
{
	void *device;
	void *primitive;
	void *draw;
};

void __jit_function(void *device, void *primitive, void *draw, int x, int y, unsigned int mask, unsigned int pid);

float rintf(float x)
{
	union
	{
		float f;
		uint32_t i;
	} u = { x };
	int e = u.i >> 23 & 0xff;
	int s = u.i >> 31;
	float y;

	if(e >= 0x7f + 23)
		return x;
	if(s)
		y = x - toint + toint;
	else
		y = x + toint - toint;
	if(y == 0)
		return s ? -0.0f : 0.0f;
	return y;
}

float nearbyintf(float x)
{
	x = rintf(x);
	return x;
}

void pixel_kernel(int task_id, void *arg)
{
	struct pixel_arg *parg = (struct pixel_arg *)arg;
	unsigned int x, y;
	unsigned int mask;
	unsigned int pid;
	int __UNIFORM__ i;

	while(1)
	{
		int __DIVERGENT__ status = gpu_rast();
		if(status == 0) break;

		gpu_rast_position(&x, &y);
		mask = gpu_rast_mask();
		pid = gpu_rast_primitive();

		__jit_function(parg->device, parg->primitive, parg->draw, x, y, mask, pid);
	}
}

int main()
{
	void *arg;

	arg = (void *)(*(volatile uint32_t *)CTRL_SCRATCH_ADDRESS);

	gpu_spawn_tasks(32, pixel_kernel, arg);

	return 0;
}
)"
