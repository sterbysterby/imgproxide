use image::{GrayImage, Luma};

pub fn _map_image<F>(img: &GrayImage, mut f: F) -> GrayImage
where 
    F: FnMut(u32, u32) -> u8 {
        let (w,h) = img.dimensions();
        let mut out = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                out.put_pixel(x, y, Luma([f(x,y)]));
            }
        }
        out
}

pub fn _map_image_f32<F>(img: &GrayImage, mut f: F) -> Vec<f32>
where 
    F: FnMut(u32, u32) -> f32 {
        let (w,h) = img.dimensions();
        let mut out = Vec::with_capacity((w*h) as usize);
        for y in 0..h {
            for x in 0..w {
                out.push(f(x,y));
            }
        }
        out
}