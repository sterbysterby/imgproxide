use image::{ImageBuffer, Luma};

pub struct Kernel {
    height : u32,
    width  : u32,
    content : Vec<f32>,
}

pub enum SobelDirection {
    Horizontal,
    Vertical,
}

impl Kernel {
    pub fn new(height : u32, width : u32, content : Vec<f32>) -> Self {
        Kernel {height, width, content}
    }

    pub fn gaussian(len : u32, sigma : f32) -> Self{
        let centre = len / 2;
        let mut sum = 0.0;
        let mut filter = vec![0.0; (len * len) as usize];

        for y in 0..len {
            for x  in 0..len {
                let dy = y as f32 - centre as f32;
                let dx = x as f32 - centre as f32;

                let distance = dy * dy + dx * dx;
                // let value = (1./sigma * ((2.0 * 3.1417) as f32).sqrt()).powf(- 1.0/2.0 * (distance as f32/ (2.0 * sigma as f32).powi(2)));
                let weight = (-distance / (2.0 * sigma.powi(2))).exp();
                filter[(y * len + x) as usize] = weight;
                sum += weight;
            }
        }
        
        // normalise
        for weight in filter.iter_mut(){
            *weight /= sum;
        }

        Kernel::new(len, len, filter)
    }

    pub fn sobel(direction : SobelDirection) -> Self {
        let content = match direction {
            SobelDirection::Horizontal =>  vec![
                1.0,2.0,1.0,
                0.0,0.0,0.0,
                -1.0,-2.0,-1.0
            ],
            SobelDirection::Vertical => vec![
                -1.0,0.0,1.0,
                -2.0,0.0,2.0,
                -1.0,0.0,1.0
            ],
        };

        Kernel::new(3,3, content)
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