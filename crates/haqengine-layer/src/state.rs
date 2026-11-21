//! Global per-instance and per-device state, and the log.

use ash::vk;
use haqengine_core::config::{apply_reload, config_path, parse_config, resolve, ConfigWatch, FileConfig, Live};
use haqengine_core::{process_name, HandleMap, Logger, Mode, Settings, LOG_CAP_BYTES, MAX_MULTIPLIER};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

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
    /// Quality-mode pipelines, created on the first Quality swapchain.
    pub quality_pipelines: std::sync::OnceLock<Result<Arc<crate::pipelines::QualityPipelines>, vk::Result>>,
    /// `vkWaitForPresentKHR` when the device has present ids and waits on (pacing is possible).
    pub wait_for_present: Option<vk::PFN_vkWaitForPresentKHR>,
}

pub static INSTANCES: LazyLock<HandleMap<Arc<InstanceData>>> = LazyLock::new(HandleMap::new);
pub static DEVICES: LazyLock<HandleMap<Arc<DeviceData>>> = LazyLock::new(HandleMap::new);
/// This process's name as Linux reports it (at most 15 bytes).
pub static PROCESS: LazyLock<String> =
    LazyLock::new(|| process_name(&std::fs::read_to_string("/proc/self/comm").unwrap_or_default()));
pub static SETTINGS: LazyLock<Settings> = LazyLock::new(|| {
    let env = |k: &str| std::env::var(k).ok();
    let mut notes = Vec::new();
    let file = config_path(env).and_then(|p| std::fs::read_to_string(p).ok()).and_then(|text| match parse_config(&text) {
        Ok((c, n)) => {
            notes.extend(n);
            Some(c)
        }
        Err(e) => {
            notes.push(format!("conf.toml not used: {e}"));
            None
        }
    });
    let mut s = Settings::from_lookup_and_file(env, file.as_ref(), &PROCESS);
    s.config_notes = notes;
    s
});

/// The mode and multiplier in force now, with a counter that moves on every change.
pub struct LiveState {
    pub live: Live,
    pub generation: u64,
    watch: Option<ConfigWatch>,
    env_set: bool,
    env_noted: bool,
}

pub static LIVE: LazyLock<Mutex<LiveState>> = LazyLock::new(|| {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    Mutex::new(LiveState {
        live: Live { mode: SETTINGS.mode, multiplier: SETTINGS.multiplier },
        generation: 0,
        watch: config_path(env).map(|p| ConfigWatch::new(p, Instant::now())),
        env_set: env("HAQENGINE_MODE").is_some() || env("HAQENGINE_MULTIPLIER").is_some(),
        env_noted: false,
    })
});

/// Looks at `conf.toml` (at most every `WATCH_EVERY`) and applies an edit to the live settings.
pub fn poll_config(now: Instant) {
    let mut st = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(text) = st.watch.as_mut().and_then(|w| w.poll(now)) else { return };
    let reloaded = parse_config(&text).map(|(c, notes): (FileConfig, Vec<String>)| {
        for n in notes {
            LOG.log(&format!("warning: {n}"));
        }
        resolve(Some(&c), &PROCESS, std::env::var("HAQENGINE_PROFILE").ok().as_deref().filter(|p| !p.is_empty()))
    });
    let (live, note) = apply_reload(st.live, reloaded, MAX_MULTIPLIER);
    if let Some(n) = note {
        LOG.log(&format!("warning: {n}"));
    }
    if live != st.live {
        st.live = live;
        st.generation += 1;
        LOG.log(&format!("config reloaded: mode {:?} multiplier {}", live.mode, live.multiplier));
        if st.env_set && !st.env_noted {
            st.env_noted = true;
            LOG.log("note: conf.toml edits now override HAQENGINE_MODE/HAQENGINE_MULTIPLIER from the launch options");
        }
    }
}

/// The live settings and their change counter.
pub fn live() -> (u64, Live) {
    let st = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    (st.generation, st.live)
}

/// The mode new swapchains start in.
pub fn live_mode() -> Mode {
    live().1.mode
}
pub static LOG: LazyLock<Logger> = LazyLock::new(|| Logger::open(SETTINGS.log_path.as_deref(), LOG_CAP_BYTES));

#[cfg(test)]
static TEST_MULTIPLIER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Test-only: overrides the multiplier (0 restores the environment's value).
#[cfg(test)]
pub fn set_test_multiplier(m: u32) {
    TEST_MULTIPLIER.store(m, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
static TEST_MEASURE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Test-only: overrides measure-only mode (`None` restores the environment's value).
#[cfg(test)]
pub fn set_test_measure(on: Option<bool>) {
    TEST_MEASURE.store(match on { None => 0, Some(true) => 1, Some(false) => 2 }, std::sync::atomic::Ordering::SeqCst);
}

/// Whether HaqEngine only measures (passes every present through and logs the real frame rate).
pub fn measuring() -> bool {
    #[cfg(test)]
    {
        match TEST_MEASURE.load(std::sync::atomic::Ordering::SeqCst) {
            1 => return true,
            2 => return false,
            _ => {}
        }
    }
    SETTINGS.measure
}

/// The multiplier this process runs with (always 1 while measuring).
pub fn multiplier() -> u32 {
    if measuring() {
        return 1;
    }
    #[cfg(test)]
    {
        let m = TEST_MULTIPLIER.load(std::sync::atomic::Ordering::SeqCst);
        if m != 0 {
            return m;
        }
    }
    live().1.multiplier
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
    /// Points the live-settings watcher at `path` (or none), returning the live settings before.
    pub fn watch_config(path: Option<std::path::PathBuf>) -> Live {
        let mut st = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        st.watch = path.map(|p| ConfigWatch::new(p, Instant::now()));
        st.live
    }

    /// Sets the live settings back.
    pub fn set_live(live: Live) {
        let mut st = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        st.live = live;
        st.generation += 1;
    }

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
            quality_pipelines: std::sync::OnceLock::new(),
            wait_for_present: None,
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
