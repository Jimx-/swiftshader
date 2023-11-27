#include "VkDeviceMemory.hpp"

#include "System/Debug.hpp"

#include <errno.h>
#include <string.h>

#include <groom.h>

class GroomDeviceMemory : public vk::DeviceMemory, public vk::ObjectBase<GroomDeviceMemory, VkDeviceMemory>
{
public:
	explicit GroomDeviceMemory(const VkMemoryAllocateInfo *pCreateInfo, void *mem, const vk::DeviceMemory::ExtendedAllocationInfo &extendedAllocationInfo, vk::Device *pDevice);

	~GroomDeviceMemory();

	VkResult allocateBuffer() override;

	void freeBuffer() override;

	void unmap() override;
	VkResult flush(VkDeviceSize offset, VkDeviceSize size) override;

	void *getDevicePointer(VkDeviceSize pOffset) const override;

private:
	groom_device_t gpuDevice;

	groom_dev_buffer_t dev_buffer;
	groom_buffer_t host_buffer;
};
