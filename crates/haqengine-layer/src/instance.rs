//! Instance-level hooks: create and destroy.

use crate::guard::guard;
use crate::loader_abi::find_instance_link;
use crate::state::{dispatch_key, InstanceData, INSTANCES, LOG};
use ash::vk;
use haqengine_core::process_name;
use std::sync::{Arc, Once};

static LOADED: Once = Once::new();

fn log_loaded() {
    LOADED.call_once(|| {
        let comm = std::fs::read_to_string("/proc/self/comm").unwrap_or_default();
        LOG.log(&format!("loaded: {} pid {} version {}", process_name(&comm), std::process::id(), env!("CARGO_PKG_VERSION")));
        for note in crate::state::SETTINGS.notes() {
            LOG.log(&format!("warning: {note}"));
        }
    });
}

pub unsafe extern "system" fn create_instance(
    create_info: *const vk::InstanceCreateInfo,
    allocator: *const vk::AllocationCallbacks,
    out: *mut vk::Instance,
) -> vk::Result {
    guard("vkCreateInstance", vk::Result::ERROR_INITIALIZATION_FAILED, || unsafe { create_instance_inner(create_info, allocator, out) })
}

unsafe fn create_instance_inner(
    create_info: *const vk::InstanceCreateInfo,
    allocator: *const vk::AllocationCallbacks,
    out: *mut vk::Instance,
) -> vk::Result {
    let Some(info) = find_instance_link((*create_info).p_next) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let link = (*info).u.p_layer_info;
    if link.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let next_gipa = (*link).pfn_next_get_instance_proc_addr;
    (*info).u.p_layer_info = (*link).p_next;

    let Some(next_create) = next_gipa(vk::Instance::null(), c"vkCreateInstance".as_ptr()) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let next_create: vk::PFN_vkCreateInstance = std::mem::transmute(next_create);
    let app = (*create_info).p_application_info;
    let api_version = if app.is_null() || (*app).api_version == 0 { vk::API_VERSION_1_0 } else { (*app).api_version };
    let result = next_create(create_info, allocator, out);
    if result != vk::Result::SUCCESS {
        return result;
    }

    let handle = *out;
    let instance = ash::Instance::load(&ash::StaticFn { get_instance_proc_addr: next_gipa }, handle);
    let surface = ash::khr::surface::InstanceFn::load(|name| std::mem::transmute(next_gipa(handle, name.as_ptr())));
    INSTANCES.insert(dispatch_key(handle), Arc::new(InstanceData { next_gipa, instance, surface, api_version }));
    log_loaded();
    vk::Result::SUCCESS
}

pub unsafe extern "system" fn destroy_instance(instance: vk::Instance, allocator: *const vk::AllocationCallbacks) {
    guard("vkDestroyInstance", (), || unsafe {
        if let Some(data) = INSTANCES.remove(dispatch_key(instance)) {
            data.instance.destroy_instance(allocator.as_ref());
        }
    })
}
