#include "VkDeviceMemoryGroom.hpp"

#include "VkDestroy.hpp"
#include "VkObject.hpp"
#include "VkPhysicalDevice.hpp"
#include "System/Debug.hpp"

GroomDeviceMemory::GroomDeviceMemory(const VkMemoryAllocateInfo *pCreateInfo, void *mem, const vk::DeviceMemory::ExtendedAllocationInfo &extendedAllocationInfo, vk::Device *pDevice)
    : vk::DeviceMemory(pCreateInfo, extendedAllocationInfo, pDevice)
    , gpuDevice(pDevice->getPhysicalDevice()->getGpuDevice())
    , dev_buffer(INVALID_DEVICE_BUFFER)
    , host_buffer(INVALID_BUFFER)
{
	uint64_t framebuffer = 0;
	hasFramebuffer = groom_device_get_feature(gpuDevice, GROOM_FEATURE_FRAMEBUFFER, &framebuffer) == 0 && framebuffer != 0;
}

GroomDeviceMemory::~GroomDeviceMemory()
{
	freeBuffer();
}

VkResult GroomDeviceMemory::allocateBuffer()
{
	host_buffer = groom_buf_alloc(gpuDevice, allocationSize);
	if(host_buffer == INVALID_BUFFER)
	{
		return VK_ERROR_OUT_OF_HOST_MEMORY;
	}

	dev_buffer = groom_mem_alloc(gpuDevice, allocationSize);
	if(dev_buffer == INVALID_DEVICE_BUFFER)
	{
		groom_buf_free(host_buffer);
		host_buffer = INVALID_BUFFER;
		return VK_ERROR_OUT_OF_DEVICE_MEMORY;
	}

	void *addr = groom_map_buffer(host_buffer);
	if(!addr)
	{
		freeBuffer();
		return VK_ERROR_MEMORY_MAP_FAILED;
	}

	buffer = addr;
	return VK_SUCCESS;
}

void GroomDeviceMemory::freeBuffer()
{
	if(host_buffer != INVALID_BUFFER)
	{
		groom_buf_free(host_buffer);
		host_buffer = INVALID_BUFFER;
	}

	if(dev_buffer != INVALID_DEVICE_BUFFER)
	{
		groom_mem_free(gpuDevice, dev_buffer);
		dev_buffer = INVALID_DEVICE_BUFFER;
	}
}

VkResult GroomDeviceMemory::flush(VkDeviceSize offset, VkDeviceSize size)
{
	if(size == VK_WHOLE_SIZE) size = allocationSize;

	groom_copy_to_device(groom_dev_buf_addr(dev_buffer) + offset, host_buffer, size, offset);

	return VK_SUCCESS;
}

VkResult GroomDeviceMemory::flushFramebuffer()
{
	if(!hasFramebuffer)
	{
		return VK_NOT_READY;
	}

	return groom_flush_framebuffer(gpuDevice) == 0 ? VK_SUCCESS : VK_ERROR_DEVICE_LOST;
}

void GroomDeviceMemory::unmap()
{
	(void)flush(0, VK_WHOLE_SIZE);
}

void *GroomDeviceMemory::getDevicePointer(VkDeviceSize pOffset) const
{
	return reinterpret_cast<void *>(groom_dev_buf_addr(dev_buffer) + pOffset);
}
