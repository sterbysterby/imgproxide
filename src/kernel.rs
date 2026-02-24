use image::{ImageBuffer, Luma};

pub struct Kernel {
    height : u32,
    width  : u32,
    content : Vec<f32>,
}
impl Kernel {
    pub fn new(height : u32, width : u32, content : Vec<f32>) -> Self {
        Kernel {height, width, content}
    }


    pub fn apply_kernel_on_pixel(&self, input : &ImageBuffer<Luma<u8>, Vec<u8>>, ix : u32, iy : u32) -> f32 {
        let (width, height) = input.dimensions();
        let mut sum = 0.0;

        let offset_x = self.width as i32/ 2;
        let offset_y = self.height as i32/ 2;

        for y  in 0..self.height {
            for x in 0.. self.width {
                let img_x = (ix as i32 + x as i32 - offset_x)
                    .clamp(0, width as i32 - 1) as u32;
                let img_y = (iy as i32 + y as i32 - offset_y)
                    .clamp(0, height as i32 - 1) as u32;

                let pixel = input.get_pixel(img_x, img_y).0[0] as f32;
                let weight = self.content[(y * self.width + x) as usize];

                sum += pixel * weight;
            }
        }
        sum
    }
}