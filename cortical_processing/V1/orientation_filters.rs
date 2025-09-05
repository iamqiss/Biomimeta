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

//! Orientation Filters Implementation for V1 Primary Visual Cortex
//! 
//! Orientation filters implement the systematic organization of orientation preferences
//! found in V1, including orientation columns and pinwheel structures.

use crate::AfiyahError;
use ndarray::{Array2, Array3};

/// Gabor filter for orientation-selective processing
#[derive(Debug, Clone)]
pub struct GaborFilter {
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub phase: f64,
    pub sigma_x: f64,
    pub sigma_y: f64,
    pub kernel: Array2<f64>,
}

/// Orientation filter bank for comprehensive orientation analysis
pub struct OrientationFilterBank {
    filters: Vec<GaborFilter>,
    orientations: Vec<f64>,
    spatial_frequencies: Vec<f64>,
}

impl GaborFilter {
    /// Creates a new Gabor filter
    pub fn new(orientation: f64, spatial_frequency: f64, phase: f64, sigma_x: f64, sigma_y: f64) -> Result<Self, AfiyahError> {
        let kernel = Self::create_gabor_kernel(orientation, spatial_frequency, phase, sigma_x, sigma_y, 7)?;
        
        Ok(Self {
            orientation,
            spatial_frequency,
            phase,
            sigma_x,
            sigma_y,
            kernel,
        })
    }

    /// Creates Gabor filter kernel
    fn create_gabor_kernel(orientation: f64, spatial_frequency: f64, phase: f64, sigma_x: f64, sigma_y: f64, size: usize) -> Result<Array2<f64>, AfiyahError> {
        let mut kernel = Array2::zeros((size, size));
        let center = size as f64 / 2.0;
        
        for i in 0..size {
            for j in 0..size {
                let x = j as f64 - center;
                let y = i as f64 - center;
                
                // Rotate coordinates according to orientation
                let x_rot = x * orientation.to_radians().cos() + y * orientation.to_radians().sin();
                let y_rot = -x * orientation.to_radians().sin() + y * orientation.to_radians().cos();
                
                // Gaussian envelope
                let gaussian = (-(x_rot * x_rot) / (2.0 * sigma_x * sigma_x) - 
                               (y_rot * y_rot) / (2.0 * sigma_y * sigma_y)).exp();
                
                // Sinusoidal carrier
                let sinusoid = (spatial_frequency * x_rot + phase).cos();
                
                kernel[[i, j]] = gaussian * sinusoid;
            }
        }
        
        Ok(kernel)
    }

    /// Applies filter to input
    pub fn apply(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (input_height, input_width) = input.dim();
        let (kernel_height, kernel_width) = self.kernel.dim();
        let mut filtered = Array2::zeros((input_height, input_width));
        
        for i in 0..input_height {
            for j in 0..input_width {
                let mut response = 0.0;
                let mut weight_sum = 0.0;
                
                for ki in 0..kernel_height {
                    for kj in 0..kernel_width {
                        let input_i = i + ki;
                        let input_j = j + kj;
                        
                        if input_i < input_height && input_j < input_width {
                            let kernel_value = self.kernel[[ki, kj]];
                            let input_value = input[[input_i, input_j]];
                            
                            response += kernel_value * input_value;
                            weight_sum += kernel_value.abs();
                        }
                    }
                }
                
                if weight_sum > 0.0 {
                    filtered[[i, j]] = response / weight_sum;
                }
            }
        }
        
        Ok(filtered)
    }

    /// Gets orientation bandwidth (half-width at half-maximum)
    pub fn get_orientation_bandwidth(&self) -> f64 {
        // Simplified calculation - in reality, this would be measured experimentally
        15.0 // degrees
    }

    /// Gets spatial frequency bandwidth
    pub fn get_spatial_frequency_bandwidth(&self) -> f64 {
        // Simplified calculation
        1.5 // octaves
    }
}

impl OrientationFilterBank {
    /// Creates a new orientation filter bank
    pub fn new() -> Result<Self, AfiyahError> {
        let mut filters = Vec::new();
        let orientations = vec![0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5];
        let spatial_frequencies = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        
        for &orientation in &orientations {
            for &spatial_freq in &spatial_frequencies {
                let filter = GaborFilter::new(orientation, spatial_freq, 0.0, 1.0, 1.0)?;
                filters.push(filter);
            }
        }
        
        Ok(Self {
            filters,
            orientations,
            spatial_frequencies,
        })
    }

    /// Processes input through all orientation filters
    pub fn process(&self, input: &Array2<f64>) -> Result<Array3<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let orientations = self.orientations.len();
        let spatial_freqs = self.spatial_frequencies.len();
        
        let mut responses = Array3::zeros((orientations, spatial_freqs, height * width));
        
        let mut filter_idx = 0;
        for orientation_idx in 0..orientations {
            for spatial_freq_idx in 0..spatial_freqs {
                if filter_idx < self.filters.len() {
                    let filter = &self.filters[filter_idx];
                    let filtered = filter.apply(input)?;
                    
                    // Flatten and store response
                    for (i, &value) in filtered.iter().enumerate() {
                        responses[[orientation_idx, spatial_freq_idx, i]] = value;
                    }
                    
                    filter_idx += 1;
                }
            }
        }
        
        Ok(responses)
    }

    /// Gets dominant orientation at each spatial location
    pub fn get_dominant_orientations(&self, responses: &Array3<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (orientations, spatial_freqs, spatial_size) = responses.dim();
        let size = (spatial_size as f64).sqrt() as usize;
        let mut orientation_map = Array2::zeros((size, size));
        
        for h in 0..size {
            for w in 0..size {
                let spatial_idx = h * size + w;
                let mut max_response = 0.0;
                let mut dominant_orientation = 0.0;
                
                for o in 0..orientations {
                    let response = responses[[o, 0, spatial_idx]]; // Use first spatial frequency
                    if response > max_response {
                        max_response = response;
                        dominant_orientation = self.orientations[o];
                    }
                }
                
                orientation_map[[h, w]] = dominant_orientation;
            }
        }
        
        Ok(orientation_map)
    }

    /// Gets orientation strength at each spatial location
    pub fn get_orientation_strength(&self, responses: &Array3<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (orientations, spatial_freqs, spatial_size) = responses.dim();
        let size = (spatial_size as f64).sqrt() as usize;
        let mut strength_map = Array2::zeros((size, size));
        
        for h in 0..size {
            for w in 0..size {
                let spatial_idx = h * size + w;
                let mut max_response = 0.0;
                
                for o in 0..orientations {
                    let response = responses[[o, 0, spatial_idx]].abs();
                    if response > max_response {
                        max_response = response;
                    }
                }
                
                strength_map[[h, w]] = max_response;
            }
        }
        
        Ok(strength_map)
    }

    /// Gets orientation selectivity (sharpness of tuning)
    pub fn get_orientation_selectivity(&self, responses: &Array3<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (orientations, spatial_freqs, spatial_size) = responses.dim();
        let size = (spatial_size as f64).sqrt() as usize;
        let mut selectivity_map = Array2::zeros((size, size));
        
        for h in 0..size {
            for w in 0..size {
                let spatial_idx = h * size + w;
                let mut responses_at_location = Vec::new();
                
                for o in 0..orientations {
                    responses_at_location.push(responses[[o, 0, spatial_idx]].abs());
                }
                
                // Calculate selectivity as ratio of max to mean response
                let max_response = responses_at_location.iter().fold(0.0_f64, |a, &b| a.max(b));
                let mean_response = responses_at_location.iter().sum::<f64>() / responses_at_location.len() as f64;
                
                if mean_response > 0.0 {
                    selectivity_map[[h, w]] = max_response / mean_response;
                }
            }
        }
        
        Ok(selectivity_map)
    }
}