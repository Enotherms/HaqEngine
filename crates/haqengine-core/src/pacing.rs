//! How many frames are shown per real frame, and when each generated frame sits in time.

/// The most frames shown per real frame.
pub const MAX_MULTIPLIER: u32 = 4;
/// The most generated frames per real frame.
pub const MAX_GENERATED: usize = (MAX_MULTIPLIER - 1) as usize;

/// When each generated frame sits between the previous real frame (0) and the current one (1), at
/// multiplier `m`: `k / m` for `k = 1..m`, in presentation order.
pub fn blend_times(m: u32) -> impl Iterator<Item = f32> {
    (1..m).map(move |k| k as f32 / m as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_times_split_evenly() {
        assert_eq!(blend_times(2).collect::<Vec<_>>(), [0.5]);
        assert_eq!(blend_times(3).collect::<Vec<_>>(), [1.0 / 3.0, 2.0 / 3.0]);
        assert_eq!(blend_times(4).collect::<Vec<_>>(), [0.25, 0.5, 0.75]);
        assert_eq!(MAX_GENERATED as u32, MAX_MULTIPLIER - 1);
    }
}
