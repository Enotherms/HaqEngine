//! Global per-instance and per-device state, and the log.

use ash::vk;
use haqengine_core::{HandleMap, Logger, Settings, LOG_CAP_BYTES};
use std::sync::{Arc, LazyLock};

/// The loader's dispatch key: the first pointer-sized word a dispatchable handle points to.
/// Queues and physical devices share the key of their device and instance.
pub fn dispatch_key<H: vk::Handle>(h: H) -> usize {
    let raw = h.as_raw() as usize;
    if raw == 0 {
        return 0;
    }
    unsafe { *(raw as *const usize) }
}

pub struct InstanceData {
    pub next_gipa: vk::PFN_vkGetInstanceProcAddr,
    pub instance: ash::Instance,
    pub surface: ash::khr::surface::InstanceFn,
    /// The `apiVersion` the app asked for (1.0 when it gave none).
    pub api_version: u32,
}

pub struct DeviceData {
    pub next_gdpa: vk::PFN_vkGetDeviceProcAddr,
    pub device: ash::Device,
    pub swapchain: ash::khr::swapchain::DeviceFn,
    pub physical_device: vk::PhysicalDevice,
    pub instance: Arc<InstanceData>,
    /// Read once at device creation.
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    /// The loader's callback for dispatchable objects HaqEngine creates (command buffers).
    pub set_loader_data: Option<crate::loader_abi::PfnSetDeviceLoaderData>,
    /// Battery-mode pipelines, created on the first battery swapchain.
    pub pipelines: std::sync::OnceLock<Result<Arc<crate::pipelines::Pipelines>, vk::Result>>,
    /// Whether this device can run Quality mode, or why not.
    pub quality: Result<(), String>,
}

pub static INSTANCES: LazyLock<HandleMap<Arc<InstanceData>>> = LazyLock::new(HandleMap::new);
pub static DEVICES: LazyLock<HandleMap<Arc<DeviceData>>> = LazyLock::new(HandleMap::new);
pub static SETTINGS: LazyLock<Settings> = LazyLock::new(|| Settings::from_lookup(|k| std::env::var(k).ok()));
pub static LOG: LazyLock<Logger> = LazyLock::new(|| Logger::open(SETTINGS.log_path.as_deref(), LOG_CAP_BYTES));

#[cfg(test)]
static TEST_MULTIPLIER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Test-only: overrides the multiplier (0 restores the environment's value).
#[cfg(test)]
pub fn set_test_multiplier(m: u32) {
    TEST_MULTIPLIER.store(m, std::sync::atomic::Ordering::SeqCst);
}

/// The multiplier this process runs with.
pub fn multiplier() -> u32 {
    #[cfg(test)]
    {
        let m = TEST_MULTIPLIER.load(std::sync::atomic::Ordering::SeqCst);
        if m != 0 {
            return m;
        }
    }
    SETTINGS.multiplier
}

/// True when any queue family of the device can record copies.
pub fn device_can_transfer(device: &DeviceData) -> bool {
    device.queue_families.iter().any(|f| {
        f.queue_flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
    })
}

/// Fake Vulkan objects for unit tests on the Mac.
#[cfg(test)]
pub mod test_support {
    use super::*;
    use ash::vk::Handle;
    use std::ffi::{c_char, c_void, CStr};

    pub unsafe extern "system" fn gdpa_nothing(_: vk::Device, _: *const c_char) -> vk::PFN_vkVoidFunction {
        None
    }

    /// A handle whose first word (the dispatch key) is a fresh, unique address.
    pub fn fake_handle() -> u64 {
        let word: &'static mut usize = Box::leak(Box::new(0));
        *word = word as *mut usize as usize;
        word as *mut usize as u64
    }

    /// A device whose ash tables resolve names through `load`, registered in `DEVICES`.
    /// Its next-layer `vkGetDeviceProcAddr` returns nothing.
    pub fn fake_device(
        load: impl Fn(&CStr) -> *const c_void + Copy,
        queue_families: Vec<vk::QueueFamilyProperties>,
    ) -> (vk::Device, Arc<DeviceData>) {
        fake_device_with_gdpa(load, queue_families, gdpa_nothing)
    }

    /// Like `fake_device`, with a chosen next-layer `vkGetDeviceProcAddr`.
    pub fn fake_device_with_gdpa(
        load: impl Fn(&CStr) -> *const c_void + Copy,
        queue_families: Vec<vk::QueueFamilyProperties>,
        next_gdpa: vk::PFN_vkGetDeviceProcAddr,
    ) -> (vk::Device, Arc<DeviceData>) {
        fake_device_full(load, queue_families, next_gdpa, |_| std::ptr::null())
    }

    /// Like `fake_device_with_gdpa`, with instance-level functions resolved through `instance_load`.
    pub fn fake_device_full(
        load: impl Fn(&CStr) -> *const c_void + Copy,
        queue_families: Vec<vk::QueueFamilyProperties>,
        next_gdpa: vk::PFN_vkGetDeviceProcAddr,
        instance_load: impl Fn(&CStr) -> *const c_void + Copy,
    ) -> (vk::Device, Arc<DeviceData>) {
        let instance = vk::Instance::from_raw(fake_handle());
        let instance = Arc::new(InstanceData {
            next_gipa: {
                unsafe extern "system" fn gipa(_: vk::Instance, _: *const c_char) -> vk::PFN_vkVoidFunction {
                    None
                }
                gipa
            },
            instance: unsafe { ash::Instance::load_with(instance_load, instance) },
            surface: ash::khr::surface::InstanceFn::load(instance_load),
            api_version: vk::API_VERSION_1_0,
        });
        let handle = vk::Device::from_raw(fake_handle());
        let data = Arc::new(DeviceData {
            next_gdpa,
            device: unsafe { ash::Device::load_with(load, handle) },
            swapchain: ash::khr::swapchain::DeviceFn::load(load),
            physical_device: vk::PhysicalDevice::null(),
            instance,
            queue_families,
            set_loader_data: None,
            pipelines: std::sync::OnceLock::new(),
            quality: Err("test device".into()),
        });
        DEVICES.insert(dispatch_key(handle), Arc::clone(&data));
        (handle, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    #[test]
    fn dispatch_key_reads_first_word_and_handles_null() {
        let table: usize = 0xdead_beef;
        let handle = vk::Instance::from_raw(&table as *const usize as u64);
        assert_eq!(dispatch_key(handle), 0xdead_beef);
        assert_eq!(dispatch_key(vk::Instance::null()), 0);
    }
}
