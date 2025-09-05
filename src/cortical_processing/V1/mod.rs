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

//! V1 Visual Cortex Processing
//! 
//! This module implements the primary visual cortex (V1) processing including
//! simple and complex cells, orientation selectivity, edge detection, and
//! spatial frequency analysis based on Hubel & Wiesel (1962) research.

pub mod simple_cells;
pub mod complex_cells;
pub mod orientation_filters;
pub mod edge_detection;

use crate::AfiyahError;
use crate::retinal_processing::RetinalOutput;
use super::{V1Output, OrientationMap, EdgeMap};

/// V1 processor that implements primary visual cortex functionality
pub struct V1Processor {
    simple_cells: simple_cells::SimpleCells,
    complex_cells: complex_cells::ComplexCells,
    orientation_filters: orientation_filters::OrientationFilters,
    edge_detector: edge_detection::EdgeDetector,
    v1_state: V1State,
}

impl V1Processor {
    /// Creates a new V1 processor with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            simple_cells: simple_cells::SimpleCells::new()?,
            complex_cells: complex_cells::ComplexCells::new()?,
            orientation_filters: orientation_filters::OrientationFilters::new()?,
            edge_detector: edge_detection::EdgeDetector::new()?,
            v1_state: V1State::default(),
        })
    }

    /// Processes retinal output through V1 cortical processing
    pub fn process(&mut self, retinal_output: &RetinalOutput) -> Result<V1Output, AfiyahError> {
        // Stage 1: Simple cell processing (orientation selectivity)
        let simple_responses = self.simple_cells.process(retinal_output)?;
        
        // Stage 2: Complex cell processing (motion invariance)
        let complex_responses = self.complex_cells.process(&simple_responses)?;
        
        // Stage 3: Orientation filter analysis
        let orientation_maps = self.orientation_filters.analyze(&simple_responses)?;
        
        // Stage 4: Edge detection
        let edge_maps = self.edge_detector.detect(&simple_responses)?;
        
        // Update V1 state
        self.update_v1_state(&simple_responses, &complex_responses)?;
        
        Ok(V1Output {
            orientation_maps,
            edge_maps,
            simple_cell_responses: simple_responses,
            complex_cell_responses: complex_responses,
        })
    }

    /// Trains orientation filters using biological learning mechanisms
    pub fn train_orientation_filters(&mut self, training_data: &[crate::VisualInput]) -> Result<(), AfiyahError> {
        // Train simple cells on orientation patterns
        self.simple_cells.train_orientation_selectivity(training_data)?;
        
        // Train complex cells on motion patterns
        self.complex_cells.train_motion_invariance(training_data)?;
        
        // Train orientation filters
        self.orientation_filters.train_filters(training_data)?;
        
        // Train edge detector
        self.edge_detector.train_detector(training_data)?;
        
        Ok(())
    }

    fn update_v1_state(&mut self, simple_responses: &[f64], complex_responses: &[f64]) -> Result<(), AfiyahError> {
        // Update V1 activity level
        let avg_simple_activity = simple_responses.iter().sum::<f64>() / simple_responses.len() as f64;
        let avg_complex_activity = complex_responses.iter().sum::<f64>() / complex_responses.len() as f64;
        
        self.v1_state.activity_level = (avg_simple_activity + avg_complex_activity) / 2.0;
        
        // Update orientation selectivity
        self.v1_state.orientation_selectivity = self.calculate_orientation_selectivity(simple_responses);
        
        // Update spatial frequency tuning
        self.v1_state.spatial_frequency_tuning = self.calculate_spatial_frequency_tuning(simple_responses);
        
        Ok(())
    }

    fn calculate_orientation_selectivity(&self, responses: &[f64]) -> f64 {
        // Calculate orientation selectivity index based on response variance
        let mean_response = responses.iter().sum::<f64>() / responses.len() as f64;
        let variance = responses.iter()
            .map(|r| (r - mean_response).powi(2))
            .sum::<f64>() / responses.len() as f64;
        
        (variance / (mean_response + 1e-6)).min(1.0)
    }

    fn calculate_spatial_frequency_tuning(&self, responses: &[f64]) -> f64 {
        // Calculate spatial frequency tuning based on response patterns
        let max_response = responses.iter().fold(0.0, |a, &b| a.max(b));
        let min_response = responses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if max_response > 0.0 {
            (max_response - min_response) / max_response
        } else {
            0.0
        }
    }
}

/// V1 processing state
#[derive(Debug, Clone)]
pub struct V1State {
    pub activity_level: f64,
    pub orientation_selectivity: f64,
    pub spatial_frequency_tuning: f64,
    pub adaptation_level: f64,
}

impl Default for V1State {
    fn default() -> Self {
        Self {
            activity_level: 0.5,
            orientation_selectivity: 0.7,
            spatial_frequency_tuning: 0.6,
            adaptation_level: 0.5,
        }
    }
}

/// Simple cell response data
#[derive(Debug, Clone)]
pub struct SimpleCellResponse {
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub phase: f64,
    pub response_strength: f64,
    pub receptive_field: ReceptiveField,
}

/// Complex cell response data
#[derive(Debug, Clone)]
pub struct ComplexCellResponse {
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub response_strength: f64,
    pub motion_direction: f64,
    pub motion_speed: f64,
}

/// Receptive field structure
#[derive(Debug, Clone)]
pub struct ReceptiveField {
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub phase: f64,
}

/// Gabor filter parameters
#[derive(Debug, Clone)]
pub struct GaborFilter {
    pub orientation: f64,
    pub spatial_frequency: f64,
    pub phase: f64,
    pub sigma_x: f64,
    pub sigma_y: f64,
    pub center_x: f64,
    pub center_y: f64,
}

impl GaborFilter {
    /// Creates a new Gabor filter with biological default parameters
    pub fn new(orientation: f64, spatial_frequency: f64, phase: f64) -> Self {
        Self {
            orientation,
            spatial_frequency,
            phase,
            sigma_x: 1.0,
            sigma_y: 1.0,
            center_x: 0.0,
            center_y: 0.0,
        }
    }

    /// Applies the Gabor filter to input data
    pub fn apply(&self, input: &[f64], width: u32, height: u32) -> Result<Vec<f64>, AfiyahError> {
        let mut output = vec![0.0; input.len()];
        
        for y in 0..height {
            for x in 0..width {
                let x_norm = (x as f64 - self.center_x) / self.sigma_x;
                let y_norm = (y as f64 - self.center_y) / self.sigma_y;
                
                // Rotate coordinates by orientation
                let cos_theta = self.orientation.to_radians().cos();
                let sin_theta = self.orientation.to_radians().sin();
                
                let x_rot = x_norm * cos_theta + y_norm * sin_theta;
                let y_rot = -x_norm * sin_theta + y_norm * cos_theta;
                
                // Calculate Gabor response
                let gaussian = (-(x_rot.powi(2) + y_rot.powi(2)) / 2.0).exp();
                let sinusoid = (2.0 * std::f64::consts::PI * self.spatial_frequency * x_rot + self.phase).cos();
                
                let response = gaussian * sinusoid;
                let idx = (y * width + x) as usize;
                output[idx] = response;
            }
        }
        
        Ok(output)
    }
}

/// Orientation tuning curve
#[derive(Debug, Clone)]
pub struct OrientationTuningCurve {
    pub preferred_orientation: f64,
    pub tuning_width: f64,
    pub response_amplitude: f64,
    pub baseline_response: f64,
}

impl OrientationTuningCurve {
    /// Creates a new orientation tuning curve
    pub fn new(preferred_orientation: f64, tuning_width: f64, response_amplitude: f64) -> Self {
        Self {
            preferred_orientation,
            tuning_width,
            response_amplitude,
            baseline_response: 0.1,
        }
    }

    /// Calculates response for given orientation
    pub fn response(&self, orientation: f64) -> f64 {
        let orientation_diff = (orientation - self.preferred_orientation).abs();
        let normalized_diff = orientation_diff / self.tuning_width;
        let gaussian_response = (-normalized_diff.powi(2) / 2.0).exp();
        
        self.baseline_response + self.response_amplitude * gaussian_response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v1_processor_creation() {
        let processor = V1Processor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_v1_state_defaults() {
        let state = V1State::default();
        assert_eq!(state.activity_level, 0.5);
        assert_eq!(state.orientation_selectivity, 0.7);
    }

    #[test]
    fn test_gabor_filter_creation() {
        let filter = GaborFilter::new(45.0, 0.5, 0.0);
        assert_eq!(filter.orientation, 45.0);
        assert_eq!(filter.spatial_frequency, 0.5);
    }

    #[test]
    fn test_orientation_tuning_curve() {
        let curve = OrientationTuningCurve::new(0.0, 30.0, 1.0);
        let response = curve.response(0.0);
        assert!(response > 0.0);
    }
}