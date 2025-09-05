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

//! Multi-Scale Attention Mechanisms
//! 
//! Implements hierarchical attention processing across multiple spatial scales,
//! mimicking the multi-scale processing in the human visual system.
//! 
//! Biological Basis:
//! - Felleman & Van Essen (1991): Hierarchical organization of visual cortex
//! - Olshausen & Field (1996): Sparse coding in visual cortex
//! - Serre et al. (2007): Hierarchical models of visual cortex
//! - Yamins et al. (2014): Deep neural networks and visual cortex

use ndarray::{Array2, Array3, Array4, Axis};
use crate::AfiyahError;

/// Multi-scale attention configuration
#[derive(Debug, Clone)]
pub struct MultiScaleConfig {
    pub scales: Vec<f64>,
    pub scale_weights: Vec<f64>,
    pub spatial_frequencies: Vec<f64>,
    pub orientation_bands: usize,
    pub temporal_windows: Vec<usize>,
    pub integration_threshold: f64,
    pub competition_strength: f64,
}

impl Default for MultiScaleConfig {
    fn default() -> Self {
        Self {
            scales: vec![0.5, 1.0, 2.0, 4.0, 8.0], // Multiple spatial scales
            scale_weights: vec![0.1, 0.2, 0.3, 0.25, 0.15], // Weighted combination
            spatial_frequencies: vec![0.1, 0.2, 0.4, 0.8, 1.6], // Cycles per degree
            orientation_bands: 8, // 8 orientation bands (0°, 22.5°, 45°, etc.)
            temporal_windows: vec![1, 3, 7, 15], // Temporal integration windows
            integration_threshold: 0.3,
            competition_strength: 0.7,
        }
    }
}

/// Multi-scale attention map
#[derive(Debug, Clone)]
pub struct MultiScaleMap {
    pub scale_maps: Vec<Array2<f64>>,
    pub orientation_maps: Vec<Array2<f64>>,
    pub frequency_maps: Vec<Array2<f64>>,
    pub integrated_map: Array2<f64>,
    pub competition_map: Array2<f64>,
    pub attention_regions: Vec<AttentionRegion>,
}

/// Attention region with scale-specific properties
#[derive(Debug, Clone)]
pub struct AttentionRegion {
    pub center: (usize, usize),
    pub scale: f64,
    pub orientation: f64,
    pub frequency: f64,
    pub strength: f64,
    pub temporal_stability: f64,
    pub competition_score: f64,
}

/// Multi-scale attention processor
pub struct MultiScaleProcessor {
    config: MultiScaleConfig,
    gabor_filters: Vec<Array2<f64>>,
    scale_filters: Vec<Array2<f64>>,
    temporal_filters: Vec<Array2<f64>>,
}

impl MultiScaleProcessor {
    /// Creates a new multi-scale attention processor
    pub fn new() -> Result<Self, AfiyahError> {
        let config = MultiScaleConfig::default();
        Self::with_config(config)
    }

    /// Creates a new multi-scale attention processor with custom configuration
    pub fn with_config(config: MultiScaleConfig) -> Result<Self, AfiyahError> {
        let mut processor = Self {
            config: config.clone(),
            gabor_filters: Vec::new(),
            scale_filters: Vec::new(),
            temporal_filters: Vec::new(),
        };

        // Initialize Gabor filters for orientation selectivity
        processor.initialize_gabor_filters()?;
        
        // Initialize scale-specific filters
        processor.initialize_scale_filters()?;
        
        // Initialize temporal filters
        processor.initialize_temporal_filters()?;

        Ok(processor)
    }

    /// Processes multi-scale attention for visual input
    pub fn process_multi_scale_attention(&self, input: &Array2<f64>, temporal_context: Option<&Array3<f64>>) -> Result<MultiScaleMap, AfiyahError> {
        let (height, width) = input.dim();
        
        // Process at multiple scales
        let scale_maps = self.process_scale_analysis(input)?;
        
        // Process orientation analysis
        let orientation_maps = self.process_orientation_analysis(input)?;
        
        // Process frequency analysis
        let frequency_maps = self.process_frequency_analysis(input)?;
        
        // Integrate across scales
        let integrated_map = self.integrate_scale_maps(&scale_maps)?;
        
        // Apply competition between regions
        let competition_map = self.apply_competition(&integrated_map)?;
        
        // Extract attention regions
        let attention_regions = self.extract_attention_regions(&competition_map)?;
        
        // Apply temporal integration if context provided
        let attention_regions = if let Some(context) = temporal_context {
            self.apply_temporal_integration(&attention_regions, context)?
        } else {
            attention_regions
        };

        Ok(MultiScaleMap {
            scale_maps,
            orientation_maps,
            frequency_maps,
            integrated_map,
            competition_map,
            attention_regions,
        })
    }

    /// Updates the multi-scale configuration
    pub fn update_config(&mut self, config: MultiScaleConfig) -> Result<(), AfiyahError> {
        self.config = config.clone();
        
        // Reinitialize filters with new configuration
        self.initialize_gabor_filters()?;
        self.initialize_scale_filters()?;
        self.initialize_temporal_filters()?;
        
        Ok(())
    }

    /// Gets current configuration
    pub fn get_config(&self) -> &MultiScaleConfig {
        &self.config
    }

    fn initialize_gabor_filters(&mut self) -> Result<(), AfiyahError> {
        self.gabor_filters.clear();
        
        for scale in &self.config.scales {
            for orientation_idx in 0..self.config.orientation_bands {
                let orientation = (orientation_idx as f64) * std::f64::consts::PI / (self.config.orientation_bands as f64);
                let filter = self.create_gabor_filter(*scale, orientation)?;
                self.gabor_filters.push(filter);
            }
        }
        
        Ok(())
    }

    fn initialize_scale_filters(&mut self) -> Result<(), AfiyahError> {
        self.scale_filters.clear();
        
        for scale in &self.config.scales {
            let filter = self.create_scale_filter(*scale)?;
            self.scale_filters.push(filter);
        }
        
        Ok(())
    }

    fn initialize_temporal_filters(&mut self) -> Result<(), AfiyahError> {
        self.temporal_filters.clear();
        
        for window_size in &self.config.temporal_windows {
            let filter = self.create_temporal_filter(*window_size)?;
            self.temporal_filters.push(filter);
        }
        
        Ok(())
    }

    fn create_gabor_filter(&self, scale: f64, orientation: f64) -> Result<Array2<f64>, AfiyahError> {
        let size = (scale * 4.0) as usize;
        let mut filter = Array2::zeros((size, size));
        let center = size as f64 / 2.0;
        
        let sigma_x = scale * 0.5;
        let sigma_y = scale * 0.25;
        let frequency = 1.0 / scale;
        
        for i in 0..size {
            for j in 0..size {
                let x = (i as f64) - center;
                let y = (j as f64) - center;
                
                // Rotate coordinates
                let x_rot = x * orientation.cos() + y * orientation.sin();
                let y_rot = -x * orientation.sin() + y * orientation.cos();
                
                // Gaussian envelope
                let gaussian = (-(x_rot * x_rot) / (2.0 * sigma_x * sigma_x) - 
                               (y_rot * y_rot) / (2.0 * sigma_y * sigma_y)).exp();
                
                // Sinusoidal component
                let sinusoid = (2.0 * std::f64::consts::PI * frequency * x_rot).cos();
                
                filter[[i, j]] = gaussian * sinusoid;
            }
        }
        
        Ok(filter)
    }

    fn create_scale_filter(&self, scale: f64) -> Result<Array2<f64>, AfiyahError> {
        let size = (scale * 6.0) as usize;
        let mut filter = Array2::zeros((size, size));
        let center = size as f64 / 2.0;
        let sigma = scale * 0.8;
        
        for i in 0..size {
            for j in 0..size {
                let x = (i as f64) - center;
                let y = (j as f64) - center;
                let distance = (x * x + y * y).sqrt();
                
                // Difference of Gaussians
                let gaussian1 = (-(distance * distance) / (2.0 * sigma * sigma)).exp();
                let gaussian2 = (-(distance * distance) / (2.0 * (sigma * 1.6) * (sigma * 1.6))).exp();
                
                filter[[i, j]] = gaussian1 - gaussian2;
            }
        }
        
        Ok(filter)
    }

    fn create_temporal_filter(&self, window_size: usize) -> Result<Array2<f64>, AfiyahError> {
        let mut filter = Array2::zeros((window_size, window_size));
        let center = window_size as f64 / 2.0;
        let sigma = window_size as f64 * 0.3;
        
        for i in 0..window_size {
            for j in 0..window_size {
                let x = (i as f64) - center;
                let y = (j as f64) - center;
                let distance = (x * x + y * y).sqrt();
                
                filter[[i, j]] = (-(distance * distance) / (2.0 * sigma * sigma)).exp();
            }
        }
        
        Ok(filter)
    }

    fn process_scale_analysis(&self, input: &Array2<f64>) -> Result<Vec<Array2<f64>>, AfiyahError> {
        let mut scale_maps = Vec::new();
        
        for (scale_idx, scale) in self.config.scales.iter().enumerate() {
            let filter = &self.scale_filters[scale_idx];
            let convolved = self.convolve2d(input, filter)?;
            scale_maps.push(convolved);
        }
        
        Ok(scale_maps)
    }

    fn process_orientation_analysis(&self, input: &Array2<f64>) -> Result<Vec<Array2<f64>>, AfiyahError> {
        let mut orientation_maps = Vec::new();
        
        for orientation_idx in 0..self.config.orientation_bands {
            let mut orientation_response = Array2::zeros(input.dim());
            
            for scale_idx in 0..self.config.scales.len() {
                let filter_idx = scale_idx * self.config.orientation_bands + orientation_idx;
                let filter = &self.gabor_filters[filter_idx];
                let convolved = self.convolve2d(input, filter)?;
                
                // Accumulate responses across scales
                for i in 0..orientation_response.len() {
                    orientation_response.as_slice_mut().unwrap()[i] += convolved.as_slice().unwrap()[i];
                }
            }
            
            orientation_maps.push(orientation_response);
        }
        
        Ok(orientation_maps)
    }

    fn process_frequency_analysis(&self, input: &Array2<f64>) -> Result<Vec<Array2<f64>>, AfiyahError> {
        let mut frequency_maps = Vec::new();
        
        for frequency in &self.config.spatial_frequencies {
            let scale = 1.0 / frequency;
            let filter = self.create_scale_filter(scale)?;
            let convolved = self.convolve2d(input, &filter)?;
            frequency_maps.push(convolved);
        }
        
        Ok(frequency_maps)
    }

    fn integrate_scale_maps(&self, scale_maps: &[Array2<f64>]) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = scale_maps[0].dim();
        let mut integrated = Array2::zeros((height, width));
        
        for (scale_idx, scale_map) in scale_maps.iter().enumerate() {
            let weight = self.config.scale_weights[scale_idx];
            
            for i in 0..height {
                for j in 0..width {
                    integrated[[i, j]] += weight * scale_map[[i, j]];
                }
            }
        }
        
        Ok(integrated)
    }

    fn apply_competition(&self, integrated_map: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = integrated_map.dim();
        let mut competition_map = integrated_map.clone();
        
        // Apply lateral inhibition for competition
        let inhibition_radius = 3;
        let inhibition_strength = self.config.competition_strength;
        
        for i in 0..height {
            for j in 0..width {
                let mut inhibition = 0.0;
                let mut count = 0;
                
                for di in -(inhibition_radius as i32)..=(inhibition_radius as i32) {
                    for dj in -(inhibition_radius as i32)..=(inhibition_radius as i32) {
                        if di == 0 && dj == 0 { continue; }
                        
                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;
                        
                        if ni < height && nj < width {
                            inhibition += integrated_map[[ni, nj]];
                            count += 1;
                        }
                    }
                }
                
                if count > 0 {
                    inhibition /= count as f64;
                    competition_map[[i, j]] = (competition_map[[i, j]] - inhibition_strength * inhibition).max(0.0);
                }
            }
        }
        
        Ok(competition_map)
    }

    fn extract_attention_regions(&self, competition_map: &Array2<f64>) -> Result<Vec<AttentionRegion>, AfiyahError> {
        let mut regions = Vec::new();
        let (height, width) = competition_map.dim();
        let threshold = self.config.integration_threshold;
        
        // Find local maxima above threshold
        for i in 1..height-1 {
            for j in 1..width-1 {
                let value = competition_map[[i, j]];
                if value > threshold {
                    // Check if it's a local maximum
                    let mut is_maximum = true;
                    for di in -1..=1 {
                        for dj in -1..=1 {
                            if di == 0 && dj == 0 { continue; }
                            
                            let ni = (i as i32 + di) as usize;
                            let nj = (j as i32 + dj) as usize;
                            
                            if ni < height && nj < width && competition_map[[ni, nj]] >= value {
                                is_maximum = false;
                                break;
                            }
                        }
                        if !is_maximum { break; }
                    }
                    
                    if is_maximum {
                        // Determine scale and orientation for this region
                        let scale = self.estimate_region_scale(competition_map, i, j)?;
                        let orientation = self.estimate_region_orientation(competition_map, i, j)?;
                        let frequency = 1.0 / scale;
                        let strength = value;
                        
                        regions.push(AttentionRegion {
                            center: (i, j),
                            scale,
                            orientation,
                            frequency,
                            strength,
                            temporal_stability: 1.0, // Will be updated with temporal integration
                            competition_score: value,
                        });
                    }
                }
            }
        }
        
        // Sort by strength
        regions.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        
        Ok(regions)
    }

    fn estimate_region_scale(&self, map: &Array2<f64>, center_i: usize, center_j: usize) -> Result<f64, AfiyahError> {
        // Simple scale estimation based on local gradient
        let (height, width) = map.dim();
        let mut max_scale = 0.0;
        
        for scale in &self.config.scales {
            let radius = (*scale * 2.0) as usize;
            let mut gradient_sum = 0.0;
            let mut count = 0;
            
            for di in -(radius as i32)..=(radius as i32) {
                for dj in -(radius as i32)..=(radius as i32) {
                    let ni = (center_i as i32 + di) as usize;
                    let nj = (center_j as i32 + dj) as usize;
                    
                    if ni < height && nj < width {
                        let distance = ((di * di + dj * dj) as f64).sqrt();
                        if distance <= radius as f64 {
                            gradient_sum += map[[ni, nj]];
                            count += 1;
                        }
                    }
                }
            }
            
            if count > 0 {
                let avg_response = gradient_sum / count as f64;
                if avg_response > max_scale {
                    max_scale = *scale;
                }
            }
        }
        
        Ok(max_scale.max(0.5)) // Minimum scale
    }

    fn estimate_region_orientation(&self, map: &Array2<f64>, center_i: usize, center_j: usize) -> Result<f64, AfiyahError> {
        // Simple orientation estimation using local gradients
        let (height, width) = map.dim();
        let mut gx = 0.0;
        let mut gy = 0.0;
        
        for di in -1..=1 {
            for dj in -1..=1 {
                let ni = (center_i as i32 + di) as usize;
                let nj = (center_j as i32 + dj) as usize;
                
                if ni < height && nj < width {
                    gx += map[[ni, nj]] * (dj as f64);
                    gy += map[[ni, nj]] * (di as f64);
                }
            }
        }
        
        let orientation = gy.atan2(gx);
        Ok(orientation)
    }

    fn apply_temporal_integration(&self, regions: &[AttentionRegion], context: &Array3<f64>) -> Result<Vec<AttentionRegion>, AfiyahError> {
        let mut updated_regions = regions.to_vec();
        
        // Apply temporal stability based on context
        for region in &mut updated_regions {
            let stability = self.calculate_temporal_stability(region, context)?;
            region.temporal_stability = stability;
        }
        
        Ok(updated_regions)
    }

    fn calculate_temporal_stability(&self, region: &AttentionRegion, context: &Array3<f64>) -> Result<f64, AfiyahError> {
        // Calculate how stable this region is across temporal context
        let (center_i, center_j) = region.center;
        let mut stability_sum = 0.0;
        let mut count = 0;
        
        for t in 0..context.shape()[0] {
            if center_i < context.shape()[1] && center_j < context.shape()[2] {
                let value = context[[t, center_i, center_j]];
                stability_sum += value;
                count += 1;
            }
        }
        
        if count > 0 {
            let avg_value = stability_sum / count as f64;
            Ok(avg_value.min(1.0).max(0.0))
        } else {
            Ok(0.0)
        }
    }

    fn convolve2d(&self, input: &Array2<f64>, filter: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (input_h, input_w) = input.dim();
        let (filter_h, filter_w) = filter.dim();
        let (output_h, output_w) = (input_h - filter_h + 1, input_w - filter_w + 1);
        
        let mut output = Array2::zeros((output_h, output_w));
        
        for i in 0..output_h {
            for j in 0..output_w {
                let mut sum = 0.0;
                for fi in 0..filter_h {
                    for fj in 0..filter_w {
                        sum += input[[i + fi, j + fj]] * filter[[fi, fj]];
                    }
                }
                output[[i, j]] = sum;
            }
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_scale_config_default() {
        let config = MultiScaleConfig::default();
        assert_eq!(config.scales.len(), 5);
        assert_eq!(config.orientation_bands, 8);
    }

    #[test]
    fn test_multi_scale_processor_creation() {
        let processor = MultiScaleProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_multi_scale_processing() {
        let processor = MultiScaleProcessor::new().unwrap();
        let input = Array2::ones((64, 64));
        
        let result = processor.process_multi_scale_attention(&input, None);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.attention_regions.is_empty());
    }

    #[test]
    fn test_gabor_filter_creation() {
        let processor = MultiScaleProcessor::new().unwrap();
        let filter = processor.create_gabor_filter(2.0, 0.0).unwrap();
        assert_eq!(filter.dim(), (8, 8));
    }
}