R"(
#include "generated/platform.h"
#include <groom/intrinsics.h>
#include <groom/spawn.h>
#include <stddef.h>
#include <stdint.h>

#define FLT_EPSILON 1.19209290E-07F
#define EPS FLT_EPSILON
static const float toint = 1 / EPS;

struct setup_arg
{
	void *device;
	void *primitive;
	void *triangle;
	void *polygon;
	uint8_t* prim_mask;
	void *draw;
	unsigned int num_primtives;
};

int __jit_function(void *device, void *primitive, void *triangle, void *polygon, void *draw, unsigned int index);

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

void setup_kernel(int task_id, void *arg)
{
	struct setup_arg *sarg = (struct setup_arg *)arg;

	sarg->prim_mask[task_id] = (uint8_t)__jit_function(sarg->device, sarg->primitive, sarg->triangle, sarg->polygon, sarg->draw, task_id);
}

int main()
{
	void *arg;
	struct setup_arg *sarg;

	arg = (void *)(*(volatile uint32_t *)CTRL_SCRATCH_ADDRESS);
	sarg = (struct setup_arg *)arg;

	gpu_spawn_tasks(sarg->num_primtives, setup_kernel, arg);

	return 0;
}
)"
