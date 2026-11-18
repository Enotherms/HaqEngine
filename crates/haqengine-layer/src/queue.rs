//! Remembers which queue family each queue belongs to, so presents know what their queue can do.

use crate::guard::guard;
use crate::state::{dispatch_key, DEVICES};
use ash::vk;
use ash::vk::Handle;
use haqengine_core::HandleMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueInfo {
    pub family: u32,
    pub device_key: usize,
}

/// Queues by raw handle.
pub static QUEUES: LazyLock<HandleMap<QueueInfo>> = LazyLock::new(HandleMap::new);

fn remember(device: vk::Device, family: u32, queue: vk::Queue) {
    if queue != vk::Queue::null() {
        QUEUES.insert(queue.as_raw() as usize, QueueInfo { family, device_key: dispatch_key(device) });
    }
}

pub unsafe extern "system" fn get_device_queue(device: vk::Device, family: u32, index: u32, out: *mut vk::Queue) {
    guard("vkGetDeviceQueue", (), || unsafe {
        if let Some(data) = DEVICES.get(dispatch_key(device)) {
            (data.device.fp_v1_0().get_device_queue)(device, family, index, out);
            if !out.is_null() {
                remember(device, family, *out);
            }
        }
    })
}

pub unsafe extern "system" fn get_device_queue2(
    device: vk::Device,
    info: *const vk::DeviceQueueInfo2,
    out: *mut vk::Queue,
) {
    guard("vkGetDeviceQueue2", (), || unsafe {
        if let Some(data) = DEVICES.get(dispatch_key(device)) {
            (data.device.fp_v1_1().get_device_queue2)(device, info, out);
            if !out.is_null() && !info.is_null() {
                remember(device, (*info).queue_family_index, *out);
            }
        }
    })
}

/// Forgets every queue of a device. Runs only at device destruction.
pub fn forget_device(device_key: usize) {
    QUEUES.retain(|info| info.device_key != device_key);
}

/// True when the queue's family can record copies (graphics, compute or transfer).
pub fn queue_can_transfer(queue: vk::Queue) -> bool {
    let Some(info) = QUEUES.get(queue.as_raw() as usize) else { return false };
    let Some(device) = DEVICES.get(info.device_key) else { return false };
    device.queue_families.get(info.family as usize).is_some_and(|f| {
        f.queue_flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::test_support::{fake_device, fake_handle};

    #[test]
    fn queue_can_transfer_unknown_is_false() {
        assert!(!queue_can_transfer(vk::Queue::from_raw(fake_handle())));
    }

    #[test]
    fn queue_can_transfer_reads_family_flags() {
        let families = vec![
            vk::QueueFamilyProperties { queue_flags: vk::QueueFlags::GRAPHICS, ..Default::default() },
            vk::QueueFamilyProperties { queue_flags: vk::QueueFlags::empty(), ..Default::default() },
        ];
        let (device, _) = fake_device(|_| std::ptr::null(), families);
        let key = dispatch_key(device);
        let q0 = vk::Queue::from_raw(fake_handle());
        let q1 = vk::Queue::from_raw(fake_handle());
        QUEUES.insert(q0.as_raw() as usize, QueueInfo { family: 0, device_key: key });
        QUEUES.insert(q1.as_raw() as usize, QueueInfo { family: 1, device_key: key });
        assert!(queue_can_transfer(q0));
        assert!(!queue_can_transfer(q1));
        DEVICES.remove(key);
    }
}
