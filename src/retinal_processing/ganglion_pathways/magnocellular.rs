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

//! Magnocellular Pathway Implementation
//! 
//! This module implements the magnocellular (M) pathway for motion and temporal
//! processing. M cells have large receptive fields, high temporal resolution,
//! and are sensitive to motion and low spatial frequencies.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use crate::retinal_processing::amacrine_networks::AmacrineResponse;

/// Magnocellular pathway for motion and temporal processing
pub struct MagnocellularPathway {
    motion_detection: MotionDetection,
    temporal_filtering: TemporalFiltering,
    spatial_integration: SpatialIntegration,
    adaptation_state: MagnocellularAdaptation,
}

impl MagnocellularPathway {
    /// Creates new magnocellular pathway with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            motion_detection: MotionDetection::new()?,
            temporal_filtering: TemporalFiltering::new()?,
            spatial_integration: SpatialIntegration::new()?,
            adaptation_state: MagnocellularAdaptation::default(),
        })
    }

    /// Processes amacrine response through magnocellular pathway
    pub fn process(&self, input: &AmacrineResponse) -> Result<MagnocellularResponse, AfiyahError> {
        // Apply motion detection
        let motion_response = self.motion_detection.detect(&input.lateral_inhibition)?;
        
        // Apply temporal filtering
        let temporal_response = self.temporal_filtering.filter(&input.temporal_filtering)?;
        
        // Apply spatial integration
        let spatial_response = self.spatial_integration.integrate(&input.lateral_inhibition)?;
        
        // Combine responses
        let combined_response = self.combine_responses(&motion_response, &temporal_response, &spatial_response)?;
        
        Ok(MagnocellularResponse {
            motion_signals: combined_response,
            temporal_resolution: self.temporal_filtering.temporal_resolution,
            spatial_resolution: self.spatial_integration.spatial_resolution,
            motion_sensitivity: self.motion_detection.motion_sensitivity,
            adaptation_level: self.adaptation_state.current_level,
        })
    }

    /// Calibrates magnocellular pathway based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.motion_detection.calibrate(params)?;
        self.temporal_filtering.calibrate(params)?;
        self.spatial_integration.calibrate(params)?;
        Ok(())
    }

    fn combine_responses(&self, motion: &[f64], temporal: &[f64], spatial: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut combined = Vec::with_capacity(motion.len());
        
        for (i, ((&motion_val, &temporal_val), &spatial_val)) in motion.iter().zip(temporal.iter()).zip(spatial.iter()).enumerate() {
            // Weighted combination: 40% motion, 30% temporal, 30% spatial
            let combined_val = (motion_val * 0.4) + (temporal_val * 0.3) + (spatial_val * 0.3);
            combined.push(combined_val.min(1.0).max(0.0));
        }
        
        Ok(combined)
    }
}

/// Motion detection for magnocellular pathway
#[derive(Debug, Clone)]
pub struct MotionDetection {
    motion_sensitivity: f64,
    motion_threshold: f64,
    direction_selectivity: f64,
    velocity_tuning: f64,
}

impl MotionDetection {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            motion_sensitivity: 0.8,
            motion_threshold: 0.1,
            direction_selectivity: 0.7,
            velocity_tuning: 0.6,
        })
    }

    pub fn detect(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut motion_response = Vec::with_capacity(input.len());
        
        for (i, &signal) in input.iter().enumerate() {
            // Calculate motion based on temporal changes
            let motion = self.calculate_motion(input, i);
            
            // Apply motion sensitivity and threshold
            let motion_response_val = if motion > self.motion_threshold {
                motion * self.motion_sensitivity * self.direction_selectivity
            } else {
                0.0
            };
            
            motion_response.push(motion_response_val.min(1.0));
        }
        
        Ok(motion_response)
    }

    fn calculate_motion(&self, input: &[f64], index: usize) -> f64 {
        if index == 0 || index >= input.len() - 1 {
            return 0.0;
        }
        
        // Calculate motion as difference between neighboring signals
        let left = input[index - 1];
        let right = input[index + 1];
        let center = input[index];
        
        // Motion is the gradient magnitude
        let motion = ((left - center).abs() + (right - center).abs()) / 2.0;
        motion
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust motion sensitivity based on adaptation
        self.motion_sensitivity = 0.8 * params.adaptation_rate;
        Ok(())
    }
}

/// Temporal filtering for magnocellular pathway
#[derive(Debug, Clone)]
pub struct TemporalFiltering {
    temporal_resolution: f64,
    temporal_bandwidth: f64,
    high_pass_cutoff: f64,
    low_pass_cutoff: f64,
}

impl TemporalFiltering {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            temporal_resolution: 60.0,  // 60 Hz temporal resolution
            temporal_bandwidth: 30.0,   // 30 Hz bandwidth
            high_pass_cutoff: 5.0,      // 5 Hz high-pass cutoff
            low_pass_cutoff: 60.0,      // 60 Hz low-pass cutoff
        })
    }

    pub fn filter(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut filtered_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // Apply temporal filtering (simplified)
            let filtered_signal = signal * self.temporal_resolution / 100.0;
            filtered_response.push(filtered_signal.min(1.0).max(0.0));
        }
        
        Ok(filtered_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust temporal resolution based on adaptation
        self.temporal_resolution = 60.0 * params.adaptation_rate;
        Ok(())
    }
}

/// Spatial integration for magnocellular pathway
#[derive(Debug, Clone)]
pub struct SpatialIntegration {
    spatial_resolution: f64,
    receptive_field_size: f64,
    integration_radius: f64,
    contrast_sensitivity: f64,
}

impl SpatialIntegration {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            spatial_resolution: 0.5,    // Lower spatial resolution
            receptive_field_size: 0.3,  // Large receptive fields
            integration_radius: 0.2,    // Integration radius
            contrast_sensitivity: 0.8,  // High contrast sensitivity
        })
    }

    pub fn integrate(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut integrated_response = Vec::with_capacity(input.len());
        
        for (i, &signal) in input.iter().enumerate() {
            // Calculate spatial integration
            let integration = self.calculate_spatial_integration(input, i);
            
            // Apply contrast sensitivity
            let integrated_signal = integration * self.contrast_sensitivity;
            integrated_response.push(integrated_signal.min(1.0).max(0.0));
        }
        
        Ok(integrated_response)
    }

    fn calculate_spatial_integration(&self, input: &[f64], center_index: usize) -> f64 {
        let mut integration_sum = 0.0;
        let mut neighbor_count = 0;
        
        // Integrate over receptive field
        let integration_range = (self.integration_radius * input.len() as f64) as usize;
        
        for i in 0..input.len() {
            let distance = (i as f64 - center_index as f64).abs() / input.len() as f64;
            
            if distance <= self.integration_radius {
                integration_sum += input[i];
                neighbor_count += 1;
            }
        }
        
        if neighbor_count > 0 {
            integration_sum / neighbor_count as f64
        } else {
            0.0
        }
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust contrast sensitivity based on adaptation
        self.contrast_sensitivity = 0.8 * params.adaptation_rate;
        Ok(())
    }
}

/// Magnocellular adaptation state
#[derive(Debug, Clone)]
pub struct MagnocellularAdaptation {
    pub current_level: f64,
    pub adaptation_rate: f64,
    pub motion_adaptation: f64,
}

impl Default for MagnocellularAdaptation {
    fn default() -> Self {
        Self {
            current_level: 0.5,
            adaptation_rate: 0.1,
            motion_adaptation: 0.5,
        }
    }
}

/// Response from magnocellular pathway processing
#[derive(Debug, Clone)]
pub struct MagnocellularResponse {
    pub motion_signals: Vec<f64>,
    pub temporal_resolution: f64,
    pub spatial_resolution: f64,
    pub motion_sensitivity: f64,
    pub adaptation_level: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magnocellular_pathway_creation() {
        let pathway = MagnocellularPathway::new();
        assert!(pathway.is_ok());
    }

    #[test]
    fn test_motion_detection() {
        let motion_detection = MotionDetection::new().unwrap();
        let input = vec![0.1, 0.5, 0.9, 0.3];
        let result = motion_detection.detect(&input);
        assert!(result.is_ok());
        let motion = result.unwrap();
        assert_eq!(motion.len(), 4);
    }

    #[test]
    fn test_temporal_filtering() {
        let temporal = TemporalFiltering::new().unwrap();
        assert_eq!(temporal.temporal_resolution, 60.0);
        assert_eq!(temporal.temporal_bandwidth, 30.0);
    }

    #[test]
    fn test_spatial_integration() {
        let spatial = SpatialIntegration::new().unwrap();
        assert_eq!(spatial.spatial_resolution, 0.5);
        assert_eq!(spatial.receptive_field_size, 0.3);
    }

    #[test]
    fn test_magnocellular_adaptation() {
        let adaptation = MagnocellularAdaptation::default();
        assert_eq!(adaptation.current_level, 0.5);
        assert_eq!(adaptation.adaptation_rate, 0.1);
    }
}
