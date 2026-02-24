
mod kernel;
// use std::process::Output;

// use kernel;
// use std::fs;
use image::{ImageBuffer, Luma, GrayImage};

use crate::kernel::Kernel;
// use Str
// use image
fn main() {
    let img = image::open("input.jpg")
        .expect("Image not found at specified path.")
        .to_luma8();
    let (width, height) = img.dimensions();
    

    let filter_y = vec![
        1.0,2.0,1.0,
        0.0,0.0,0.0,
        -1.0,-2.0,-1.0
    ];

    let filter_x = vec![
        -1.0,0.0,1.0,
        -2.0,0.0,2.0,
        -1.0,0.0,1.0
    ];
    let kernel_y = Kernel::new(3,3, filter_y);
    let kernel_x = Kernel::new(3,3, filter_x);
    let kernel_gauss = Kernel::gaussian(3, 1.0);
    
    let mut gauss_out = GrayImage::new(width,height);

    // sharpen image
    for y in 0..height {
        for x in 0..width {
            let p = kernel_gauss.apply_kernel_on_pixel(&img, x, y);
            gauss_out.put_pixel(x, y, Luma([p.clamp(0.0,255.0) as u8]));
        }
    }

    // apply gradient magnitude
    let mut output = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let gx = kernel_x.apply_kernel_on_pixel(&gauss_out, x, y);
            let gy = kernel_y.apply_kernel_on_pixel(&gauss_out, x, y);
            
            let mag = (gx.powi(2) + gy.powi(2)).sqrt();

            let p = mag.min(255.0) as u8;
            output.put_pixel(x, y, Luma([p]));
        }
    }

    output.save("output.png").expect("Failed to save output Image.");
    println!("Image Processing Completed!");
}

fn _sobel(input : ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>>{
    let height = input.height();
    let width = input.width();
    let mut buffer : ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    for y in 1..height-1 {
        for x in 1..width-1 {
            let val0 = input.get_pixel(x-1, y-1).0[0] as i32; // [x-1, y-1]
            let val1 = input.get_pixel(x, y-1).0[0] as i32; // [x, y-1]
            let val2 = input.get_pixel(x+1, y-1).0[0] as i32; // [x+1, y-1]
            let val3 = input.get_pixel(x-1, y).0[0] as i32; // [x-1, y]
            let val5 = input.get_pixel(x+1, y).0[0] as i32; // [x+1, y]
            let val6 = input.get_pixel(x-1, y-1).0[0] as i32; // [x-1, y+1]
            let val7 = input.get_pixel(x, y).0[0] as i32; // [x, y+1]
            let val8 = input.get_pixel(x+1, y+1).0[0] as i32; // [x+1, y+1]
            
            let gx : i32 = val0 + (val3 * 2) + val6
                - (val2 + (val5 * 2) + val8);
            let gy : i32 = val0 + (2 * val1) + val2
                - (val6 + (2 * val7) + val8);

            let g = ((gx as f64).powi(2) + (gy as f64).powi(2)).sqrt();
            buffer.put_pixel(x, y, Luma([g as u8]));
        }
    }

    buffer
}


fn hogs(input : &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Vec<f64>> {
    // let input = input
    let (width, height) = input.dimensions();
    let hists = hogs_calc_hists(input);

    let blocks_x = (width / 8) as usize;
    let blocks_y = (height / 8) as usize;

    let mut features = vec![vec![0.0; 36]; blocks_x * blocks_y];

    for y in 0.. blocks_y-1 {
        for x in 0..blocks_x-1 {
            let mut feature_vec = [0.0; 36];
            let indices = [
                (y * blocks_x + x),
                (y * blocks_x + x + 1),
                ((y+1) * blocks_x + x),
                ((y+1) * blocks_x + x + 1)];
            
            for i in 0.. 4 {
                feature_vec[(i * 9) .. (i * 9 + 9)].copy_from_slice(&hists[indices[i]]);
            }
            
            let norm = feature_vec.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            feature_vec.iter_mut().for_each(|x| *x /= norm + 0.00001);
            features[y * blocks_x + x] = feature_vec.to_vec();
        }
    }
    features
}

fn hogs_calc_hists(input : &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<[f64; 9]>{
    // thank you https://builtin.com/articles/histogram-of-oriented-gradients
    let height = input.height();
    let width = input.width();
    let mut gy_buffer : ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(width, height);
    let mut gx_buffer : ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(width, height);
    
    for y in 1..height-1 {
        for x in 1..width-1 {
            let val0 = input.get_pixel(x-1, y).0[0] as i16; // [x-1, y]
            let val2 = input.get_pixel(x+1, y).0[0] as i16; // [x+1, y]
            gx_buffer.put_pixel(x, y, Luma([-val0 + val2]));
            
            let val0 = input.get_pixel(x, y-1).0[0] as i16; // [x-1, y]
            let val2 = input.get_pixel(x, y+1).0[0] as i16; // [x+1, y]
            gy_buffer.put_pixel(x, y, Luma([-val0 + val2]));

        }
    }
    
    let mut angles : Vec<f64> = Vec::new();
    let mut mags : Vec<f64> = Vec::new();

    for (gx, gy) in gx_buffer.iter().zip(gy_buffer.iter()) {
        let gx = *gx as f64;
        let gy = *gy as f64;
        angles.push(gy.atan2(gx).to_degrees());
        mags.push((gx * gx + gy * gy).sqrt());
    }

    // calculate the normalised histogram values of each cell of the image 
    // let mut normed_hists : Vec<[f64; 9]> = Vec::new();
    let cells_y = ((height-1) / 8) as usize;
    let cells_x = ((width-1) / 8) as usize;
    let mut hists : Vec<[f64; 9]> = vec![[0.0; 9]; cells_x * cells_y];
    for y in 0..  cells_y {
        for x in 0.. cells_x {

            let mut hist : [f64; 9] = [0.0; 9];
            // can this be done through the use of a zip with mag + dir?
            for i in 0..8 {
                for j in 0..8 {
                    let index = ((y * 8 + i) * width as usize + (x * 8) + j) as usize;
                    let magnitude = mags[index];
                    let angle = angles[index];
                    let bin = (angle / 20.0).floor() as usize;
                    
                    let share = angle % 20.0 / 20.0;
                    hist[bin%9] += (1.0 - share) * magnitude;
                    hist[(bin+1) % 9] += share * magnitude;

                }
            }
            hists[y * cells_x + x] = hist;
        }
    }
    hists
}