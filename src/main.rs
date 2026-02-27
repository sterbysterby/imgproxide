mod kernel;
mod feature_detect;
mod helper;

use image::{ImageBuffer, Luma, GrayImage};
use crate::kernel::Kernel;

fn main() {
    let img = image::open("input.jpg")
        .expect("Image not found at specified path.")
        .to_luma8();
    let (width, height) = img.dimensions();

    let gauss = Kernel::gaussian(15, 10.0);
    let sobel_y = Kernel::sobel(kernel::SobelDirection::Vertical);
    let sobel_x = Kernel::sobel(kernel::SobelDirection::Horizontal);
    
    let mut gauss_out = GrayImage::new(width,height);

    // sharpen image
    for y in 0..height {
        for x in 0..width {
            let p = gauss.apply_kernel_on_pixel(&img, x, y);
            gauss_out.put_pixel(x, y, Luma([p.clamp(0.0,255.0) as u8]));
        }
    }

    // apply gradient magnitude
    let mut output = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let gx = sobel_x.apply_kernel_on_pixel(&gauss_out, x, y);
            let gy = sobel_y.apply_kernel_on_pixel(&gauss_out, x, y);
            
            let mag = (gx.powi(2) + gy.powi(2)).sqrt();

            let p = mag.min(255.0) as u8;
            output.put_pixel(x, y, Luma([p]));
        }
    }

    output.save("output.png").expect("Failed to save output Image.");
    println!("Image Processing Completed!");
}