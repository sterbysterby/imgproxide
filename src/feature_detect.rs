use image::GrayImage;

use crate::kernel::{self, Kernel};

pub fn normalise_blocks(hists: Vec<[f64; 9]>, cells_x : usize, cells_y : usize) -> Vec<Vec<f64>> {
    let h_ref = &hists;
    
    (0..cells_y).flat_map(|y| {
        (0..cells_x).map(move |x| {
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


pub fn hogs_hists(input : &GrayImage) -> Vec<[f64; 9]>{
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
                    hist[bin + 1] += share * mag;

                }
            }
        }
    }
    hists
}