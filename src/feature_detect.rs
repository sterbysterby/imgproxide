use image::{GrayImage, Luma};
use serde::{Deserialize, Serialize};

use crate::kernel::{self, Kernel};

#[derive(Serialize, Deserialize, Debug)]
pub struct HogResult {
    pub dimensions: (u32, u32),
    pub cell_size: usize,
    pub block_size: usize,
    pub data: Vec<Vec<f64>>,
}

impl HogResult {
    pub fn save(&self, path: &str) -> std::io::Result<()>{
        let file = std::fs::File::create(path)?;
        serde_json::to_writer(file, self)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let result = serde_json::from_reader(reader)?;
        Ok(result)
    }
}

pub fn calc_hog(input: GrayImage) -> HogResult {
    let (w,h) = input.dimensions();

    let hists = hog_hists(&input);
    let cells_x = (w / 8) as usize;
    let cells_y = (h / 8) as usize;

    let normed = normalise_blocks(hists, cells_x, cells_y);

    HogResult {
        dimensions: (w,h),
        cell_size: 8,
        block_size: 2,
        data: normed }
}

fn normalise_blocks(hists: Vec<[f64; 9]>, cells_x : usize, cells_y : usize) -> Vec<Vec<f64>> {
    let h_ref = &hists;
    
    (0..cells_y.saturating_sub(1)).flat_map(|y| {
        (0..cells_x.saturating_sub(1)).map(move |x| {
            let mut block = Vec::with_capacity(36);
            
            let top_l = y *cells_x + x; 
            let top_r = top_l + 1; 
            let bot_l = (y +1) *cells_x + x; 
            let bot_r = bot_l + 1;

            for &idx in &[top_l, top_r, bot_l, bot_r] {
                block.extend_from_slice(&h_ref[idx]);
            }

            let l2 = (block.iter().map(|v| v * v).sum::<f64>() + 1e-5).sqrt();

            block.iter().map(|v| v / l2).collect::<Vec<f64>>()
        })
    }).collect()
}


fn hog_hists(input : &GrayImage) -> Vec<[f64; 9]>{
    let (w,h) = input.dimensions();

    let x_kernel = Kernel::sobel(kernel::SobelDirection::Horizontal);
    let y_kernel = Kernel::sobel(kernel::SobelDirection::Vertical);

    let kx = &x_kernel;
    let ky = &y_kernel;

    let mags_and_angs : Vec<(f64, f64)>= (0..h).flat_map(move |y| {
        (0..w).map(move |x| {
            let gx = kx.apply_kernel_on_pixel(&input, x, y) as f64;
            let gy = ky.apply_kernel_on_pixel(&input, x, y) as f64;

            let mag = (gx * gx + gy * gy).sqrt();
            let mut ang = gy.atan2(gx).to_degrees();

            // ensure angle within 0->180
            if ang < 0.0 {ang += 180.0}
            if ang > 180.0 {ang -= 180.0}
            
            (mag, ang)
        })
    }).collect();

    let cells_y = (h / 8) as usize;
    let cells_x = (w / 8) as usize;
    let mut hists : Vec<[f64; 9]> = vec![[0.0; 9]; cells_x * cells_y];

    for (cell_idx, hist) in hists.iter_mut().enumerate() {
        let cx = (cell_idx % cells_x) * 8;
        let cy = (cell_idx / cells_y) * 8;

        for i in 0..8 {
            for j in 0..8 {
                let imgx = cx + j;
                let imgy = cy + i;

                if imgx < w as usize && imgy < h as usize {
                    let (mag, ang) = mags_and_angs[imgy * w as usize + imgx];
                    
                    let bin_pos = ang / 20.0;
                    let bin = bin_pos.floor() as usize;
                    let share = bin_pos.fract();

                    hist[bin % 9] += (1.0 - share) * mag;
                    hist[(bin + 1) % 9] += share * mag;

                }
            }
        }
    }
    hists
}

use std::f64::consts::PI;

pub fn visualize_hog(hog: &HogResult, strength_scale: f64) -> GrayImage {
    // Create a canvas based on the original dimensions
    let mut canvas = GrayImage::new(hog.dimensions.0, hog.dimensions.1);
    
    let cells_x = (hog.dimensions.0 / hog.cell_size as u32) as usize;
    let cells_y = (hog.dimensions.1 / hog.cell_size as u32) as usize;
    
    // The HOG data we saved is normalized blocks (36 elements).
    // For visualization, it's often easier to use the raw 9-bin hists,
    // but we can reconstruct a "view" from the normalized blocks.
    for y in 0..cells_y - 1 {
        for x in 0..cells_x - 1 {
            // Get the first 9 elements of the block (the top-left cell of the 2x2)
            let cell_hist = &hog.data[y * (cells_x - 1) + x][0..9];
            
            let center_x = (x * hog.cell_size + hog.cell_size / 2) as f64;
            let center_y = (y * hog.cell_size + hog.cell_size / 2) as f64;

            for bin in 0..9 {
                let magnitude = cell_hist[bin] * strength_scale;
                if magnitude < 0.1 { continue; } // Skip weak gradients

                // Calculate angle in radians (bin 0 = 0°, bin 1 = 20°, etc.)
                let angle = (bin as f64 * 20.0).to_radians();
                
                // Draw a small line representing this bin
                // We draw from (center - dir) to (center + dir)
                let x_dir = angle.cos() * (hog.cell_size as f64 / 2.0);
                let y_dir = angle.sin() * (hog.cell_size as f64 / 2.0);

                draw_line(
                    &mut canvas,
                    (center_x - x_dir, center_y - y_dir),
                    (center_x + x_dir, center_y + y_dir),
                    (magnitude * 255.0).min(255.0) as u8
                );
            }
        }
    }
    canvas
}

// Simple Bresenham or linear interpolation line drawer
fn draw_line(img: &mut GrayImage, start: (f64, f64), end: (f64, f64), color: u8) {
    let steps = 10; // Simple fixed-step drawing for visualization
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let x = start.0 + t * (end.0 - start.0);
        let y = start.1 + t * (end.1 - start.1);
        if x >= 0.0 && x < img.width() as f64 && y >= 0.0 && y < img.height() as f64 {
            img.put_pixel(x as u32, y as u32, Luma([color]));
        }
    }
}