/* Biomimeta - Biomimetic Video Compression & Streaming Engine
*  Copyright (C) 2025 Neo Qiss. All Rights Reserved.
*
*  PROPRIETARY NOTICE: This software and all associated intellectual property,
*  including but not limited to algorithms, biological models, neural architectures,
*  and compression methodologies, are the exclusive property of Neo Qiss.
*
*  COMMERCIAL RESTRICTION: Commercial use, distribution, or integration of this
*  software is STRICTLY PROHIBITED without explicit written authorization and
*  formal partnership agreements. Unauthorized commercial use constitutes
*  copyright infringement and may result in legal action.
*
*  RESEARCH LICENSE: This software is made available under the Biological Research
*  Public License (BRPL) v1.0 EXCLUSIVELY for academic research, educational purposes,
*  and non-commercial scientific collaboration. Commercial entities must obtain
*  separate licensing agreements.
*
*  BIOLOGICAL RESEARCH ATTRIBUTION: This software implements proprietary biological
*  models derived from extensive neuroscientific research. All use must maintain
*  complete scientific attribution as specified in the BRPL license terms.
*
*  NO WARRANTIES: This software is provided for research purposes only. No warranties
*  are made regarding biological accuracy, medical safety, or fitness for any purpose.
*
*  For commercial licensing: commercial@biomimeta.com
*  For research partnerships: research@biomimeta.com
*  Legal inquiries: legal@biomimeta.com
*
*  VIOLATION OF THESE TERMS MAY RESULT IN IMMEDIATE LICENSE TERMINATION AND LEGAL ACTION.
*/

//! Simple Cells Implementation for V1 Primary Visual Cortex
//! 
//! Simple cells are orientation-selective edge detectors with linear receptive fields.
//! They respond optimally to oriented edges and gratings at specific orientations.

use crate::AfiyahError;
use ndarray::{Array2, Array3};

/// Simple cell for orientation-selective edge detection
#[derive(Debug, Clone)]
pub struct SimpleCell {
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub phase: f64,
    pub gabor_kernel: Array2<f64>,
    pub adaptation_state: f64,
}

/// Bank of simple cells for comprehensive orientation processing
pub struct SimpleCellBank {
    cells: Vec<SimpleCell>,
    orientations: Vec<f64>,
    spatial_frequencies: Vec<f64>,
}

impl SimpleCell {
    /// Creates a new simple cell with specified parameters
    pub fn new(orientation: f64, spatial_frequency: f64, phase: f64, size: usize) -> Result<Self, AfiyahError> {
        let gabor_kernel = Self::create_gabor_kernel(orientation, spatial_frequency, phase, size)?;
        
        Ok(Self {
            orientation,
            spatial_frequency,
            phase,
            gabor_kernel,
            adaptation_state: 1.0,
        })
    }

    /// Creates Gabor filter kernel
    fn create_gabor_kernel(orientation: f64, spatial_frequency: f64, phase: f64, size: usize) -> Result<Array2<f64>, AfiyahError> {
        let mut kernel = Array2::zeros((size, size));
        let center = size as f64 / 2.0;
        
        let sigma_x = size as f64 / 4.0;
        let sigma_y = size as f64 / 6.0;
        
        for i in 0..size {
            for j in 0..size {
                let x = j as f64 - center;
                let y = i as f64 - center;
                
                let x_rot = x * orientation.to_radians().cos() + y * orientation.to_radians().sin();
                let y_rot = -x * orientation.to_radians().sin() + y * orientation.to_radians().cos();
                
                let gaussian = (-(x_rot * x_rot) / (2.0 * sigma_x * sigma_x) - 
                               (y_rot * y_rot) / (2.0 * sigma_y * sigma_y)).exp();
                
                let sinusoid = (spatial_frequency * x_rot + phase).cos();
                
                kernel[[i, j]] = gaussian * sinusoid;
            }
        }
        
        Ok(kernel)
    }

    /// Computes response to visual input
    pub fn compute_response(&mut self, input: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (input_height, input_width) = input.dim();
        let (kernel_height, kernel_width) = self.gabor_kernel.dim();
        
        let mut response = 0.0;
        let mut weight_sum = 0.0;
        
        for kh in 0..kernel_height {
            for kw in 0..kernel_width {
                let input_h = input_height / 2 + kh;
                let input_w = input_width / 2 + kw;
                
                if input_h < input_height && input_w < input_width {
                    let kernel_value = self.gabor_kernel[[kh, kw]];
                    let input_value = input[[input_h, input_w]];
                    
                    response += kernel_value * input_value;
                    weight_sum += kernel_value.abs();
                }
            }
        }
        
        if weight_sum > 0.0 {
            response /= weight_sum;
        }
        
        Ok(response * self.adaptation_state)
    }
}

impl SimpleCellBank {
    /// Creates a new simple cell bank
    pub fn new() -> Result<Self, AfiyahError> {
        let mut cells = Vec::new();
        let orientations = vec![0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5];
        let spatial_frequencies = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        
        for &orientation in &orientations {
            for &spatial_freq in &spatial_frequencies {
                for &phase in &[0.0, 90.0] {
                    let cell = SimpleCell::new(orientation, spatial_freq, phase, 7)?;
                    cells.push(cell);
                }
            }
        }
        
        Ok(Self {
            cells,
            orientations,
            spatial_frequencies,
        })
    }

    /// Processes input through all simple cells
    pub fn process(&mut self, input: &Array2<f64>) -> Result<Array3<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let orientations = self.orientations.len();
        let spatial_freqs = self.spatial_frequencies.len();
        
        let mut responses = Array3::zeros((orientations, spatial_freqs, height * width));
        
        let mut cell_idx = 0;
        for orientation_idx in 0..orientations {
            for spatial_freq_idx in 0..spatial_freqs {
                if cell_idx < self.cells.len() {
                    let cell = &mut self.cells[cell_idx];
                    
                    for h in 0..height {
                        for w in 0..width {
                            let response = cell.compute_response(input)?;
                            let spatial_idx = h * width + w;
                            responses[[orientation_idx, spatial_freq_idx, spatial_idx]] = response;
                        }
                    }
                    
                    cell_idx += 1;
                }
            }
        }
        
        Ok(responses)
    }
}