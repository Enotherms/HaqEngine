//! A thread-safe map from Vulkan handle keys to per-object data.

use std::collections::HashMap;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Maps a `usize` key (a Vulkan dispatch key or raw handle) to a cloneable value.
/// A poisoned lock is recovered instead of panicking.
pub struct HandleMap<T: Clone> {
    map: RwLock<HashMap<usize, T>>,
}

impl<T: Clone> Default for HandleMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> HandleMap<T> {
    pub fn new() -> Self {
        HandleMap { map: RwLock::new(HashMap::new()) }
    }
    fn read(&self) -> RwLockReadGuard<'_, HashMap<usize, T>> {
        self.map.read().unwrap_or_else(|e| e.into_inner())
    }
    fn write(&self) -> RwLockWriteGuard<'_, HashMap<usize, T>> {
        self.map.write().unwrap_or_else(|e| e.into_inner())
    }
    pub fn insert(&self, key: usize, value: T) {
        self.write().insert(key, value);
    }
    pub fn get(&self, key: usize) -> Option<T> {
        self.read().get(&key).cloned()
    }
    pub fn remove(&self, key: usize) -> Option<T> {
        self.write().remove(&key)
    }
    /// Keeps only the entries whose value passes `keep`. Not for per-frame use.
    pub fn retain(&self, mut keep: impl FnMut(&T) -> bool) {
        self.write().retain(|_, v| keep(v));
    }
    pub fn len(&self) -> usize {
        self.read().len()
    }
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn dispatch_map_insert_get_remove() {
        let m = HandleMap::new();
        m.insert(1, "a");
        assert_eq!(m.get(1), Some("a"));
        assert_eq!(m.remove(1), Some("a"));
        assert_eq!(m.get(1), None);
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn dispatch_map_reuses_key_after_remove() {
        let m = HandleMap::new();
        m.insert(7, "old");
        m.remove(7);
        m.insert(7, "new");
        assert_eq!(m.get(7), Some("new"));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn dispatch_map_is_shared_across_threads() {
        let m = Arc::new(HandleMap::<u32>::new());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let m = Arc::clone(&m);
                std::thread::spawn(move || {
                    for i in 0..1000u32 {
                        let key = (t * 1000 + i) as usize;
                        m.insert(key, i);
                        assert_eq!(m.get(key), Some(i));
                        assert_eq!(m.remove(key), Some(i));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(m.len(), 0);
    }
}
