//! Per-swapchain resources for generated frames. Created once per swapchain, never per frame.

use crate::state::DeviceData;
use ash::vk;
use haqengine_core::FrameStats;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::time::Instant;

/// One frame's worth of reusable copy resources.
pub struct Slot {
    pub cmd: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub acquired: vk::Semaphore,
    pub copy_ready: vk::Semaphore,
    pub real_ready: vk::Semaphore,
}

/// Command pool and slots, created on the first generating present (the queue family is only known then).
pub struct Ready {
    pub family: u32,
    pub pool: vk::CommandPool,
    pub slots: Vec<Slot>,
}

pub struct GenStats {
    pub real: u64,
    pub generated: u64,
    pub skipped: u64,
    pub intervals: FrameStats,
    pub present_cost: FrameStats,
    pub last_present: Option<Instant>,
    pub last_log_ms: u64,
}

pub struct GenState {
    pub ready: Option<Ready>,
    pub next: usize,
    pub stats: GenStats,
}

pub struct FrameGen {
    pub images: Vec<vk::Image>,
    pub extent: vk::Extent2D,
    pub disabled: AtomicBool,
    pub state: Mutex<GenState>,
}

impl FrameGen {
    pub fn new(images: Vec<vk::Image>, extent: vk::Extent2D) -> FrameGen {
        FrameGen {
            images,
            extent,
            disabled: AtomicBool::new(false),
            state: Mutex::new(GenState {
                ready: None,
                next: 0,
                stats: GenStats {
                    real: 0,
                    generated: 0,
                    skipped: 0,
                    intervals: FrameStats::new(),
                    present_cost: FrameStats::new(),
                    last_present: None,
                    last_log_ms: 0,
                },
            }),
        }
    }

    /// Creates the command pool and slots for `family` on first use.
    ///
    /// # Safety
    /// `device` must be the live device that owns this swapchain.
    pub unsafe fn ensure_ready(&self, state: &mut GenState, device: &DeviceData, family: u32) -> Result<(), vk::Result> {
        if let Some(ready) = &state.ready {
            return if ready.family == family { Ok(()) } else { Err(vk::Result::ERROR_FEATURE_NOT_PRESENT) };
        }
        let d = &device.device;
        let pool = d.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER | vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )?;
        let mut ready = Ready { family, pool, slots: Vec::with_capacity(self.images.len() + 2) };
        let result = (|| -> Result<(), vk::Result> {
            let cmds = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(self.images.len() as u32 + 2),
            )?;
            for cmd in cmds {
                let fence = d.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?;
                let sem = || d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None);
                let slot = Slot { cmd, fence, acquired: vk::Semaphore::null(), copy_ready: vk::Semaphore::null(), real_ready: vk::Semaphore::null() };
                ready.slots.push(slot);
                let last = ready.slots.last_mut().expect("just pushed");
                last.acquired = sem()?;
                last.copy_ready = sem()?;
                last.real_ready = sem()?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                state.ready = Some(ready);
                Ok(())
            }
            Err(e) => {
                destroy_ready(device, &ready);
                Err(e)
            }
        }
    }

    /// Waits for in-flight copies (up to 1 s) and destroys the pool and slots.
    ///
    /// # Safety
    /// `device` must be the live device that owns this swapchain.
    pub unsafe fn destroy(&self, device: &DeviceData) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ready) = state.ready.take() {
            destroy_ready(device, &ready);
        }
    }
}

/// Waits (up to 1 s) for the slots' copies, then destroys everything in `ready`.
unsafe fn destroy_ready(device: &DeviceData, ready: &Ready) {
    let d = &device.device;
    let fences: Vec<vk::Fence> = ready.slots.iter().map(|s| s.fence).filter(|f| *f != vk::Fence::null()).collect();
    if !fences.is_empty() {
        let _ = d.wait_for_fences(&fences, true, 1_000_000_000);
    }
    for s in &ready.slots {
        for sem in [s.acquired, s.copy_ready, s.real_ready] {
            if sem != vk::Semaphore::null() {
                d.destroy_semaphore(sem, None);
            }
        }
        if s.fence != vk::Fence::null() {
            d.destroy_fence(s.fence, None);
        }
    }
    d.destroy_command_pool(ready.pool, None);
}
