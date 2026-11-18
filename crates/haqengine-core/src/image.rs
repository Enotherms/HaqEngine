//! Small row-major images and deterministic test patterns.

/// An RGBA pixel with 8 bits per channel.
pub type Rgba8 = [u8; 4];

/// A row-major image.
#[derive(Debug, Clone, PartialEq)]
pub struct Image<T> {
    pub width: u32,
    pub height: u32,
    pub data: Vec<T>,
}

impl<T: Copy> Image<T> {
    pub fn new(width: u32, height: u32, fill: T) -> Self {
        Image { width, height, data: vec![fill; width as usize * height as usize] }
    }
    pub fn get(&self, x: u32, y: u32) -> T {
        self.data[(y * self.width + x) as usize]
    }
    pub fn set(&mut self, x: u32, y: u32, v: T) {
        let w = self.width;
        self.data[(y * w + x) as usize] = v;
    }
}

/// Black and white squares of `cell` pixels.
pub fn checker(width: u32, height: u32, cell: u32) -> Image<Rgba8> {
    let cell = cell.max(1);
    let mut img = Image::new(width, height, [0, 0, 0, 255]);
    for y in 0..height {
        for x in 0..width {
            if (x / cell + y / cell) % 2 == 1 {
                img.set(x, y, [255, 255, 255, 255]);
            }
        }
    }
    img
}

/// Red rises left to right, green top to bottom, blue is their average.
pub fn gradient(width: u32, height: u32) -> Image<Rgba8> {
    let ramp = |i: u32, n: u32| if n <= 1 { 0 } else { (i * 255 / (n - 1)) as u8 };
    let mut img = Image::new(width, height, [0, 0, 0, 255]);
    for y in 0..height {
        for x in 0..width {
            let (r, g) = (ramp(x, width), ramp(y, height));
            img.set(x, y, [r, g, ((r as u16 + g as u16) / 2) as u8, 255]);
        }
    }
    img
}

/// Seeded xorshift64 noise, identical on every machine.
pub fn noise(width: u32, height: u32, seed: u64) -> Image<Rgba8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut img = Image::new(width, height, [0, 0, 0, 255]);
    for v in img.data.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let b = state.to_le_bytes();
        *v = [b[0], b[1], b[2], 255];
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_deterministic() {
        assert_eq!(noise(16, 9, 7), noise(16, 9, 7));
        assert_ne!(noise(16, 9, 7), noise(16, 9, 8));
    }

    #[test]
    fn checker_and_gradient_shapes() {
        let c = checker(4, 4, 2);
        assert_eq!(c.get(0, 0), [0, 0, 0, 255]);
        assert_eq!(c.get(2, 0), [255, 255, 255, 255]);
        let g = gradient(256, 2);
        assert_eq!(g.get(0, 0)[0], 0);
        assert_eq!(g.get(255, 0)[0], 255);
    }
}
