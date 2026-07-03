//! Minimal headless Vulkan **compute** foundation for the Keryx Android miner.
//!
//! Vendored from `Muskwak/keryx-miner-rdna3`'s `keryx-vulkan` crate (the desktop AMD Vulkan PoM
//! backend), trimmed for a mobile GPU (Adreno/Mali) target:
//!   - dropped `khh` (legacy kHeavyHash PoW) and the `bench` micro-benchmark module — both are
//!     dead weight on Android, which only ever mines PoM (the legacy pre-fork lineup is gone);
//!   - requests Vulkan **1.2** instead of 1.3 and drops `shaderIntegerDotProduct` — that feature
//!     existed only to serve `khh`'s packed-dot kernel. The PoM walk itself needs nothing beyond
//!     `shaderInt64` (core since 1.0) and `bufferDeviceAddress` (core in 1.2), which is much more
//!     broadly supported across mid-range mobile GPUs than a full Vulkan 1.3 device.
//!
//! `ash` loads `vulkan-1` (Android: `libvulkan.so`) at runtime — no Vulkan SDK needed to build or
//! run, only the loader that ships with the device's GPU driver. Fully headless/surfaceless: no
//! window/surface code, so there's nothing Android-windowing-specific to strip.

use ash::vk;
use std::ffi::{CStr, CString};

pub mod pom_walk;

/// Quick probe: the name of the Vulkan compute device, or None if no loader/device is usable.
pub fn probe_device() -> Option<String> {
    Vk::new().ok().map(|vk| vk.device_name().to_string())
}

/// Quick probe: total VRAM/shared-memory (MiB) of the Vulkan compute device the miner would mine
/// on, or None if no usable device. On a phone this is a slice of the shared system RAM (unified
/// memory), not dedicated VRAM — used by the model capability gate the same way the desktop forks
/// use their GPU probes.
pub fn probe_vram_mb() -> Option<u64> {
    Vk::new().ok().map(|vk| vk.device_local_vram_mb())
}

/// A ready-to-use compute device: instance, the chosen physical device, a logical device with a
/// compute queue, and a command pool. One per process (the miner uses a single GPU).
pub struct Vk {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub device: ash::Device,
    #[allow(dead_code)]
    pdevice: vk::PhysicalDevice,
    queue: vk::Queue,
    #[allow(dead_code)]
    queue_family: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    cmd_pool: vk::CommandPool,
    device_name: String,
}

impl Vk {
    /// Open a compute-capable Vulkan device. Enables `shaderInt64` (64-bit shader arithmetic) and
    /// `bufferDeviceAddress` (lets the PoM weight blob be read through a 64-bit pointer instead of
    /// a bound storage descriptor, since it can exceed `maxStorageBufferRange`).
    pub fn new() -> Result<Self, String> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("Vulkan loader not found: {e}"))?;
            let app_info = vk::ApplicationInfo::default()
                .application_name(c"keryx-miner-android")
                .api_version(vk::make_api_version(0, 1, 2, 0));
            let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
            let instance = entry
                .create_instance(&create_info, None)
                .map_err(|e| format!("create_instance failed: {e}"))?;

            // Pick a physical device: first DISCRETE_GPU, else the first available. Phones have no
            // discrete GPU, so this just takes whichever integrated GPU is enumerated first.
            let pdevices = instance
                .enumerate_physical_devices()
                .map_err(|e| format!("enumerate_physical_devices: {e}"))?;
            if pdevices.is_empty() {
                instance.destroy_instance(None);
                return Err("no Vulkan physical devices found".into());
            }
            let mut pick: Option<(vk::PhysicalDevice, String, bool)> = None;
            for pd in pdevices {
                let props = instance.get_physical_device_properties(pd);
                let name = cstr_array_to_string(&props.device_name);
                let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
                match &pick {
                    None => pick = Some((pd, name, discrete)),
                    Some((_, _, picked_discrete)) if !picked_discrete && discrete => {
                        pick = Some((pd, name, discrete))
                    }
                    _ => {}
                }
            }
            let (pdevice, device_name, _) = pick.unwrap();

            // Find a queue family that supports COMPUTE.
            let qfams = instance.get_physical_device_queue_family_properties(pdevice);
            let queue_family = qfams
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("no compute-capable queue family")? as u32;

            let priorities = [1.0f32];
            let qcis = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qcis)
                .enabled_features(&features)
                .push_next(&mut features12);
            let device = instance
                .create_device(pdevice, &dci, None)
                .map_err(|e| format!("create_device failed: {e}"))?;
            let queue = device.get_device_queue(queue_family, 0);
            let mem_props = instance.get_physical_device_memory_properties(pdevice);
            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| format!("create_command_pool: {e}"))?;

            Ok(Self {
                _entry: entry,
                instance,
                device,
                pdevice,
                queue,
                queue_family,
                mem_props,
                cmd_pool,
                device_name,
            })
        }
    }

    /// Human-readable name of the selected GPU (e.g. "Adreno (TM) 740").
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Total device-local memory (MiB) = the largest `DEVICE_LOCAL` heap. On a UMA mobile GPU this
    /// is a slice of system RAM the driver reports as device-local, not dedicated VRAM.
    pub fn device_local_vram_mb(&self) -> u64 {
        let heaps = &self.mem_props.memory_heaps[..self.mem_props.memory_heap_count as usize];
        let bytes = heaps
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0);
        bytes / (1024 * 1024)
    }

    /// The device's single-allocation ceiling (`maxMemoryAllocationSize`), 0 if unreported. Mobile
    /// GPUs commonly report a much smaller cap here than desktop AMD's ~2 GiB, which is why the
    /// PoM weight blob's shard size must be picked from this at runtime rather than assumed.
    pub fn max_alloc_bytes(&self) -> u64 {
        unsafe { self.max_memory_allocation_size() }
    }

    fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, String> {
        for i in 0..self.mem_props.memory_type_count {
            let suitable = (type_bits & (1 << i)) != 0;
            if suitable && self.mem_props.memory_types[i as usize].property_flags.contains(flags) {
                return Ok(i);
            }
        }
        Err(format!("no memory type for flags {flags:?}"))
    }

    /// Allocate a host-visible, coherent STORAGE buffer (mapped reads/writes; GPU reads over the
    /// bus). Used for small buffers (address tables, the winner flag), not the weight blob.
    pub fn create_buffer(&self, size: u64) -> Result<GpuBuffer, String> {
        assert!(size > 0, "zero-size buffer");
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).map_err(|e| e.to_string())?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mt = self.find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
                    None,
                )
                .map_err(|e| e.to_string())?;
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(|e| e.to_string())?;
            Ok(GpuBuffer { buffer, memory, size })
        }
    }

    /// Allocate a **DEVICE_LOCAL** buffer that exposes a GPU device address, and fill it from
    /// `data` via a host-visible staging copy. Used for the PoM weight shards: the walk's 256
    /// data-dependent reads per nonce need device-local memory to stay fast.
    pub fn create_device_local_address_buffer(&self, data: &[u8]) -> Result<(GpuBuffer, u64), String> {
        let size = data.len() as u64;
        assert!(size > 0, "zero-size buffer");
        unsafe {
            let max_alloc = self.max_memory_allocation_size();
            if max_alloc != 0 && size > max_alloc {
                return Err(format!(
                    "weight shard {size} bytes exceeds maxMemoryAllocationSize {max_alloc} bytes"
                ));
            }

            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).map_err(|e| e.to_string())?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mt = self.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mt)
                        .push_next(&mut flags),
                    None,
                )
                .map_err(|e| e.to_string())?;
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(|e| e.to_string())?;

            // Host-visible staging source: fill it, copy to device-local, then free it.
            let staging_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let staging = self.device.create_buffer(&staging_info, None).map_err(|e| e.to_string())?;
            let sreq = self.device.get_buffer_memory_requirements(staging);
            let smt = self.find_memory_type(
                sreq.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let smem = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(sreq.size).memory_type_index(smt),
                    None,
                )
                .map_err(|e| e.to_string())?;
            self.device.bind_buffer_memory(staging, smem, 0).map_err(|e| e.to_string())?;
            let ptr = self
                .device
                .map_memory(smem, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| e.to_string())? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.unmap_memory(smem);

            let copy_res = self.immediate_copy(staging, buffer, size);
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(smem, None);
            if let Err(e) = copy_res {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
                return Err(e);
            }

            let address = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            Ok((GpuBuffer { buffer, memory, size }, address))
        }
    }

    /// The device's single-allocation ceiling (`maxMemoryAllocationSize`), 0 if unreported.
    unsafe fn max_memory_allocation_size(&self) -> u64 {
        let mut limits11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut limits11);
        self.instance.get_physical_device_properties2(self.pdevice, &mut props2);
        limits11.max_memory_allocation_size
    }

    /// Submit a one-shot buffer→buffer copy and block until it completes.
    unsafe fn immediate_copy(&self, src: vk::Buffer, dst: vk::Buffer, size: u64) -> Result<(), String> {
        let cmd = self
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| e.to_string())?[0];
        let cmds = [cmd];
        let run = || -> Result<(), String> {
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| e.to_string())?;
            self.device.cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy::default().size(size)]);
            self.device.end_command_buffer(cmd).map_err(|e| e.to_string())?;
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?;
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let res = self
                .device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(|e| e.to_string())
                .and_then(|_| self.device.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| e.to_string()));
            self.device.destroy_fence(fence, None);
            res
        };
        let res = run();
        self.device.free_command_buffers(self.cmd_pool, &cmds);
        res
    }

    /// Copy `data` into a host-visible buffer (`data.len()` must be ≤ the buffer size).
    pub fn write_buffer(&self, b: &GpuBuffer, data: &[u8]) {
        assert!(data.len() as u64 <= b.size, "write past buffer end");
        unsafe {
            let ptr = self
                .device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
                .expect("map_memory") as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.unmap_memory(b.memory);
        }
    }

    /// Read `out.len()` bytes back from a host-visible buffer.
    pub fn read_buffer(&self, b: &GpuBuffer, out: &mut [u8]) {
        assert!(out.len() as u64 <= b.size, "read past buffer end");
        unsafe {
            let ptr = self
                .device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
                .expect("map_memory") as *const u8;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
            self.device.unmap_memory(b.memory);
        }
    }

    /// Build a compute pipeline from SPIR-V with `n_bindings` storage buffers and `push_size`
    /// bytes of push constants. The returned `Kernel` is reusable across dispatches.
    pub fn make_kernel(&self, spirv: &[u32], n_bindings: u32, push_size: u32) -> Result<Kernel, String> {
        unsafe {
            let bindings: Vec<_> = (0..n_bindings)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let pc_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_size)];
            let set_layouts = [set_layout];
            let pipeline_layout = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&pc_ranges),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
                .map_err(|e| e.to_string())?;
            let entry = CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry);
            let pipeline = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| e.to_string())?[0];
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(n_bindings.max(1))];
            let desc_pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes)
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let cmd = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| e.to_string())?[0];
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| e.to_string())?;
            Ok(Kernel { set_layout, pipeline_layout, pipeline, module, desc_pool, cmd, fence })
        }
    }

    /// Bind `buffers` (in binding order 0..n) + `push` constants and dispatch `groups` workgroups
    /// on x. Blocks until the GPU finishes (fence wait).
    pub fn dispatch(&self, k: &Kernel, buffers: &[&GpuBuffer], push: &[u8], groups: u32) {
        unsafe {
            let dev = &self.device;
            dev.reset_descriptor_pool(k.desc_pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
            let layouts = [k.set_layout];
            let set = dev
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default().descriptor_pool(k.desc_pool).set_layouts(&layouts),
                )
                .unwrap()[0];
            let infos: Vec<_> = buffers
                .iter()
                .map(|b| vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE))
                .collect();
            let writes: Vec<_> = (0..buffers.len())
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]))
                })
                .collect();
            dev.update_descriptor_sets(&writes, &[]);

            dev.reset_command_buffer(k.cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            dev.begin_command_buffer(
                k.cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
            dev.cmd_bind_pipeline(k.cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            dev.cmd_bind_descriptor_sets(k.cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline_layout, 0, &[set], &[]);
            if !push.is_empty() {
                dev.cmd_push_constants(k.cmd, k.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            dev.cmd_dispatch(k.cmd, groups, 1, 1);
            dev.end_command_buffer(k.cmd).unwrap();

            let cmds = [k.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            dev.reset_fences(&[k.fence]).unwrap();
            dev.queue_submit(self.queue, &[submit], k.fence).unwrap();
            dev.wait_for_fences(&[k.fence], true, u64::MAX).unwrap();
        }
    }

    /// Release a kernel's Vulkan objects.
    pub fn destroy_kernel(&self, k: &Kernel) {
        unsafe {
            self.device.destroy_fence(k.fence, None);
            self.device.destroy_descriptor_pool(k.desc_pool, None);
            self.device.destroy_pipeline(k.pipeline, None);
            self.device.destroy_shader_module(k.module, None);
            self.device.destroy_pipeline_layout(k.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(k.set_layout, None);
        }
    }

    /// Release a buffer's Vulkan objects.
    pub fn destroy_buffer(&self, b: &GpuBuffer) {
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }
}

impl Drop for Vk {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A storage buffer + its backing device memory. Destroy via [`Vk::destroy_buffer`].
pub struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

/// A compiled compute pipeline (+ descriptor pool, command buffer, fence), reusable across
/// dispatches. Destroy via [`Vk::destroy_kernel`].
pub struct Kernel {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    module: vk::ShaderModule,
    desc_pool: vk::DescriptorPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

fn cstr_array_to_string(arr: &[std::os::raw::c_char]) -> String {
    unsafe { CStr::from_ptr(arr.as_ptr()).to_string_lossy().into_owned() }
}
