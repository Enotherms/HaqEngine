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
}

pub struct DeviceData {
    pub next_gdpa: vk::PFN_vkGetDeviceProcAddr,
    pub device: ash::Device,
    pub swapchain: ash::khr::swapchain::DeviceFn,
}

pub static INSTANCES: LazyLock<HandleMap<Arc<InstanceData>>> = LazyLock::new(HandleMap::new);
pub static DEVICES: LazyLock<HandleMap<Arc<DeviceData>>> = LazyLock::new(HandleMap::new);
pub static LOG: LazyLock<Logger> = LazyLock::new(|| {
    let settings = Settings::from_lookup(|k| std::env::var(k).ok());
    Logger::open(settings.log_path.as_deref(), LOG_CAP_BYTES)
});

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
