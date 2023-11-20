R"(
#include "generated/platform.h"
#include <groom/intrinsics.h>
#include <groom/spawn.h>
#include <stddef.h>
#include <stdint.h>

#define FLT_EPSILON 1.19209290E-07F
#define EPS FLT_EPSILON
static const float toint = 1 / EPS;

#define NR_THREADS 32

struct vertex_task
{
	uint32_t vertex_count;
	uint32_t primitive_start;
};

struct vertex_arg
{
	void *device;
	void *output;
	void *batch;
	struct vertex_task *vertextask;
	void *draw;
	void *cache[NR_THREADS];
};

void __jit_function(void *device, void *output, void *batch, void *vertextask, void *draw, void *cache, unsigned int index);

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

void vertex_kernel(int task_id, void *arg)
{
	struct vertex_arg *varg = (struct vertex_arg *)arg;
	unsigned int thread_id = gpu_thread_global_id();
	void *cache = varg->cache[thread_id];

	__jit_function(varg->device, varg->output, varg->batch, varg->vertextask, varg->draw, cache, task_id);
}

int main()
{
	void *arg;
	struct vertex_arg *varg;

	arg = (void *)(*(volatile uint32_t *)CTRL_SCRATCH_ADDRESS);
	varg = (struct vertex_arg *)arg;

	gpu_spawn_tasks(varg->vertextask->vertex_count, vertex_kernel, arg);

	return 0;
}
)"
