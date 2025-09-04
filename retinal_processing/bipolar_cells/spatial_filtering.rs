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

//! Spatial Filtering Implementation
//! 
//! This module implements spatial filtering for bipolar cells, including
//! edge detection, spatial frequency tuning, and orientation selectivity.
//! This provides the foundation for cortical processing.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use super::lateral_inhibition::InhibitedResponse;

/// Spatial filtering for bipolar cell processing
pub struct SpatialFiltering {
    edge_detection: EdgeDetection,
    spatial_frequency_tuning: SpatialFrequencyTuning,
    orientation_selectivity: OrientationSelectivity,
    gabor_filters: GaborFilters,
}

impl SpatialFiltering {
    /// Creates new spatial filtering with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            edge_detection: EdgeDetection::new()?,
            spatial_frequency_tuning: SpatialFrequencyTuning::new()?,
            orientation_selectivity: OrientationSelectivity::new()?,
            gabor_filters: GaborFilters::new()?,
        })
    }

    /// Processes inhibited response through spatial filtering
    pub fn process(&self, input: &InhibitedResponse) -> Result<super::FilteredResponse, AfiyahError> {
        // Apply edge detection
        let edge_response = self.edge_detection.detect(&input.inhibited_on, &input.inhibited_off)?;
        
        // Apply spatial frequency tuning
        let frequency_response = self.spatial_frequency_tuning.tune(&input.inhibited_on)?;
        
        // Apply orientation selectivity
        let orientation_response = self.orientation_selectivity.select(&input.inhibited_on)?;
        
        // Apply Gabor filtering
        let gabor_response = self.gabor_filters.filter(&input.inhibited_on)?;
        
        // Combine responses
        let combined_on = self.combine_responses(&input.inhibited_on, &edge_response, &frequency_response)?;
        let combined_off = self.combine_responses(&input.inhibited_off, &edge_response, &frequency_response)?;
        
        Ok(super::FilteredResponse {
            on_center: combined_on,
            off_center: combined_off,
            spatial_frequency_tuning: frequency_response,
        })
    }

    /// Calibrates spatial filtering based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.edge_detection.calibrate(params)?;
        self.spatial_frequency_tuning.calibrate(params)?;
        self.orientation_selectivity.calibrate(params)?;
        self.gabor_filters.calibrate(params)?;
        Ok(())
    }

    fn combine_responses(&self, base: &[f64], edge: &[f64], frequency: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut combined = Vec::with_capacity(base.len());
        
        for (i, ((&base_val, &edge_val), &freq_val)) in base.iter().zip(edge.iter()).zip(frequency.iter()).enumerate() {
            // Weighted combination of responses
            let combined_val = (base_val * 0.5) + (edge_val * 0.3) + (freq_val * 0.2);
            combined.push(combined_val.min(1.0).max(0.0));
        }
        
        Ok(combined)
    }
}

/// Edge detection for spatial filtering
#[derive(Debug, Clone)]
pub struct EdgeDetection {
    edge_threshold: f64,
    edge_strength: f64,
    gradient_threshold: f64,
}

impl EdgeDetection {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            edge_threshold: 0.2,
            edge_strength: 0.8,
            gradient_threshold: 0.1,
        })
    }

    pub fn detect(&self, on_response: &[f64], off_response: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut edge_response = Vec::with_capacity(on_response.len());
        
        for (i, (&on, &off)) in on_response.iter().zip(off_response.iter()).enumerate() {
            // Calculate gradient (difference between neighboring cells)
            let gradient = self.calculate_gradient(on_response, i);
            
            // Edge detection based on gradient and ON/OFF difference
            let on_off_diff = (on - off).abs();
            let edge_strength = if gradient > self.gradient_threshold && on_off_diff > self.edge_threshold {
                gradient * self.edge_strength
            } else {
                0.0
            };
            
            edge_response.push(edge_strength.min(1.0));
        }
        
        Ok(edge_response)
    }

    fn calculate_gradient(&self, input: &[f64], index: usize) -> f64 {
        if index == 0 || index >= input.len() - 1 {
            return 0.0;
        }
        
        let left = input[index - 1];
        let right = input[index + 1];
        let center = input[index];
        
        // Calculate gradient magnitude
        let gradient = ((left - center).abs() + (right - center).abs()) / 2.0;
        gradient
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust edge threshold based on adaptation
        self.edge_threshold = 0.2 * params.adaptation_rate;
        Ok(())
    }
}

/// Spatial frequency tuning for bipolar cells
#[derive(Debug, Clone)]
pub struct SpatialFrequencyTuning {
    optimal_frequency: f64,
    frequency_bandwidth: f64,
    tuning_strength: f64,
}

impl SpatialFrequencyTuning {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            optimal_frequency: 0.5,  // Optimal spatial frequency
            frequency_bandwidth: 0.3, // Frequency bandwidth
            tuning_strength: 0.7,    // Tuning strength
        })
    }

    pub fn tune(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut tuned_response = Vec::with_capacity(input.len());
        
        for (i, &signal) in input.iter().enumerate() {
            // Calculate spatial frequency for this position
            let spatial_frequency = self.calculate_spatial_frequency(i, input.len());
            
            // Apply frequency tuning
            let tuning_response = self.apply_frequency_tuning(signal, spatial_frequency);
            tuned_response.push(tuning_response);
        }
        
        Ok(tuned_response)
    }

    fn calculate_spatial_frequency(&self, index: usize, total_length: usize) -> f64 {
        // Convert position to spatial frequency
        let position = index as f64 / total_length as f64;
        position * 2.0 // Scale to appropriate frequency range
    }

    fn apply_frequency_tuning(&self, signal: f64, frequency: f64) -> f64 {
        // Gaussian tuning around optimal frequency
        let frequency_diff = (frequency - self.optimal_frequency).abs();
        let tuning_factor = (-frequency_diff.powi(2) / (2.0 * self.frequency_bandwidth.powi(2))).exp();
        
        signal * tuning_factor * self.tuning_strength
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust tuning strength based on adaptation
        self.tuning_strength = 0.7 * params.adaptation_rate;
        Ok(())
    }
}

/// Orientation selectivity for spatial filtering
#[derive(Debug, Clone)]
pub struct OrientationSelectivity {
    orientation_angles: Vec<f64>,
    orientation_bandwidth: f64,
    selectivity_strength: f64,
}

impl OrientationSelectivity {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            orientation_angles: vec![0.0, 45.0, 90.0, 135.0], // 4 orientations
            orientation_bandwidth: 30.0, // 30 degree bandwidth
            selectivity_strength: 0.6,
        })
    }

    pub fn select(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut orientation_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // Calculate orientation selectivity (simplified)
            let orientation_strength = signal * self.selectivity_strength;
            orientation_response.push(orientation_strength.min(1.0));
        }
        
        Ok(orientation_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust selectivity strength based on adaptation
        self.selectivity_strength = 0.6 * params.adaptation_rate;
        Ok(())
    }
}

/// Gabor filters for spatial filtering
#[derive(Debug, Clone)]
pub struct GaborFilters {
    filter_bank: Vec<GaborFilter>,
    filter_strength: f64,
}

impl GaborFilters {
    pub fn new() -> Result<Self, AfiyahError> {
        let mut filter_bank = Vec::new();
        
        // Create 4 Gabor filters with different orientations
        for i in 0..4 {
            let orientation = i as f64 * 45.0; // 0, 45, 90, 135 degrees
            filter_bank.push(GaborFilter::new(orientation)?);
        }
        
        Ok(Self {
            filter_bank,
            filter_strength: 0.5,
        })
    }

    pub fn filter(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut filtered_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // Apply Gabor filtering (simplified)
            let gabor_response = signal * self.filter_strength;
            filtered_response.push(gabor_response.min(1.0));
        }
        
        Ok(filtered_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust filter strength based on adaptation
        self.filter_strength = 0.5 * params.adaptation_rate;
        Ok(())
    }
}

/// Individual Gabor filter
#[derive(Debug, Clone)]
pub struct GaborFilter {
    orientation: f64,
    frequency: f64,
    phase: f64,
    sigma: f64,
}

impl GaborFilter {
    pub fn new(orientation: f64) -> Result<Self, AfiyahError> {
        Ok(Self {
            orientation,
            frequency: 0.5,
            phase: 0.0,
            sigma: 1.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_filtering_creation() {
        let filtering = SpatialFiltering::new();
        assert!(filtering.is_ok());
    }

    #[test]
    fn test_edge_detection() {
        let edge_detection = EdgeDetection::new().unwrap();
        let on = vec![0.1, 0.5, 0.9, 0.3];
        let off = vec![0.2, 0.4, 0.8, 0.4];
        let result = edge_detection.detect(&on, &off);
        assert!(result.is_ok());
        let edges = result.unwrap();
        assert_eq!(edges.len(), 4);
    }

    #[test]
    fn test_spatial_frequency_tuning() {
        let tuning = SpatialFrequencyTuning::new().unwrap();
        let input = vec![0.1, 0.5, 0.9];
        let result = tuning.tune(&input);
        assert!(result.is_ok());
        let tuned = result.unwrap();
        assert_eq!(tuned.len(), 3);
    }

    #[test]
    fn test_orientation_selectivity() {
        let selectivity = OrientationSelectivity::new().unwrap();
        let input = vec![0.1, 0.5, 0.9];
        let result = selectivity.select(&input);
        assert!(result.is_ok());
        let selected = result.unwrap();
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_gabor_filters() {
        let gabor = GaborFilters::new().unwrap();
        let input = vec![0.1, 0.5, 0.9];
        let result = gabor.filter(&input);
        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 3);
    }
      }
