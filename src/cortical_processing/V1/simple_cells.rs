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

//! Simple Cells Implementation
//! 
//! This module implements simple cells in V1 that exhibit orientation selectivity
//! and spatial frequency tuning based on Hubel & Wiesel (1962) research.
//! Simple cells have elongated receptive fields and respond to oriented edges.

use crate::AfiyahError;
use crate::retinal_processing::RetinalOutput;
use super::{SimpleCellResponse, ReceptiveField, GaborFilter, OrientationTuningCurve};

/// Simple cells processor for orientation-selective edge detection
pub struct SimpleCells {
    orientation_filters: Vec<GaborFilter>,
    orientation_tuning_curves: Vec<OrientationTuningCurve>,
    spatial_frequencies: Vec<f64>,
    phases: Vec<f64>,
    simple_cell_state: SimpleCellState,
}

impl SimpleCells {
    /// Creates a new simple cells processor with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        let orientations = vec![0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5];
        let spatial_frequencies = vec![0.1, 0.2, 0.4, 0.8, 1.6];
        let phases = vec![0.0, 90.0, 180.0, 270.0];
        
        let mut orientation_filters = Vec::new();
        let mut orientation_tuning_curves = Vec::new();
        
        for &orientation in &orientations {
            for &spatial_frequency in &spatial_frequencies {
                for &phase in &phases {
                    orientation_filters.push(GaborFilter::new(orientation, spatial_frequency, phase));
                    orientation_tuning_curves.push(OrientationTuningCurve::new(
                        orientation,
                        30.0, // tuning width in degrees
                        1.0,  // response amplitude
                    ));
                }
            }
        }
        
        Ok(Self {
            orientation_filters,
            orientation_tuning_curves,
            spatial_frequencies,
            phases,
            simple_cell_state: SimpleCellState::default(),
        })
    }

    /// Processes retinal output through simple cells
    pub fn process(&mut self, retinal_output: &RetinalOutput) -> Result<Vec<f64>, AfiyahError> {
        let mut responses = Vec::new();
        
        // Combine all retinal streams for processing
        let combined_input = self.combine_retinal_streams(retinal_output)?;
        
        // Process through each orientation filter
        for (i, filter) in self.orientation_filters.iter().enumerate() {
            let filter_response = filter.apply(&combined_input, 128, 128)?;
            let response_strength = self.calculate_response_strength(&filter_response);
            
            // Apply orientation tuning curve
            let tuning_curve = &self.orientation_tuning_curves[i];
            let tuned_response = tuning_curve.response(filter.orientation) * response_strength;
            
            responses.push(tuned_response);
        }
        
        // Update simple cell state
        self.update_simple_cell_state(&responses)?;
        
        Ok(responses)
    }

    /// Trains orientation selectivity using biological learning mechanisms
    pub fn train_orientation_selectivity(&mut self, training_data: &[crate::VisualInput]) -> Result<(), AfiyahError> {
        // Implement Hebbian learning for orientation selectivity
        for input in training_data {
            let combined_input = self.extract_luminance_data(input)?;
            let responses = self.process_orientation_training(&combined_input)?;
            
            // Update filter weights based on Hebbian learning
            self.update_filter_weights(&responses, &combined_input)?;
        }
        
        Ok(())
    }

    fn combine_retinal_streams(&self, retinal_output: &RetinalOutput) -> Result<Vec<f64>, AfiyahError> {
        // Combine magnocellular, parvocellular, and koniocellular streams
        let mut combined = Vec::new();
        
        // Weight the streams based on biological importance
        let magno_weight = 0.4;
        let parvo_weight = 0.4;
        let konio_weight = 0.2;
        
        let max_len = retinal_output.magnocellular_stream.len()
            .max(retinal_output.parvocellular_stream.len())
            .max(retinal_output.koniocellular_stream.len());
        
        for i in 0..max_len {
            let magno_val = retinal_output.magnocellular_stream.get(i).copied().unwrap_or(0.0);
            let parvo_val = retinal_output.parvocellular_stream.get(i).copied().unwrap_or(0.0);
            let konio_val = retinal_output.koniocellular_stream.get(i).copied().unwrap_or(0.0);
            
            let combined_val = magno_weight * magno_val + parvo_weight * parvo_val + konio_weight * konio_val;
            combined.push(combined_val);
        }
        
        Ok(combined)
    }

    fn extract_luminance_data(&self, input: &crate::VisualInput) -> Result<Vec<f64>, AfiyahError> {
        // Extract luminance data from visual input
        Ok(input.luminance_data.clone())
    }

    fn calculate_response_strength(&self, filter_response: &[f64]) -> f64 {
        // Calculate response strength as the sum of squared responses
        filter_response.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn process_orientation_training(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut responses = Vec::new();
        
        for filter in &self.orientation_filters {
            let filter_response = filter.apply(input, 128, 128)?;
            let response_strength = self.calculate_response_strength(&filter_response);
            responses.push(response_strength);
        }
        
        Ok(responses)
    }

    fn update_filter_weights(&mut self, responses: &[f64], input: &[f64]) -> Result<(), AfiyahError> {
        // Implement Hebbian learning rule: Δw = η * response * input
        let learning_rate = 0.01;
        
        for (i, &response) in responses.iter().enumerate() {
            if response > 0.1 { // Only update for significant responses
                // Update filter parameters based on Hebbian learning
                // This is a simplified version - in practice, this would update
                // the actual filter weights in the Gabor filters
                self.simple_cell_state.learning_activity += response * learning_rate;
            }
        }
        
        Ok(())
    }

    fn update_simple_cell_state(&mut self, responses: &[f64]) -> Result<(), AfiyahError> {
        // Update simple cell state based on responses
        let avg_response = responses.iter().sum::<f64>() / responses.len() as f64;
        let max_response = responses.iter().fold(0.0, |a, &b| a.max(b));
        
        self.simple_cell_state.activity_level = avg_response;
        self.simple_cell_state.max_response = max_response;
        self.simple_cell_state.orientation_selectivity = self.calculate_orientation_selectivity(responses);
        
        Ok(())
    }

    fn calculate_orientation_selectivity(&self, responses: &[f64]) -> f64 {
        // Calculate orientation selectivity index
        let max_response = responses.iter().fold(0.0, |a, &b| a.max(b));
        let min_response = responses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if max_response > 0.0 {
            (max_response - min_response) / max_response
        } else {
            0.0
        }
    }
}

/// Simple cell processing state
#[derive(Debug, Clone)]
pub struct SimpleCellState {
    pub activity_level: f64,
    pub max_response: f64,
    pub orientation_selectivity: f64,
    pub learning_activity: f64,
    pub adaptation_level: f64,
}

impl Default for SimpleCellState {
    fn default() -> Self {
        Self {
            activity_level: 0.0,
            max_response: 0.0,
            orientation_selectivity: 0.0,
            learning_activity: 0.0,
            adaptation_level: 0.5,
        }
    }
}

/// Simple cell receptive field analysis
pub struct ReceptiveFieldAnalyzer {
    pub field_size: f64,
    pub orientation_preference: f64,
    pub spatial_frequency_preference: f64,
    pub phase_preference: f64,
}

impl ReceptiveFieldAnalyzer {
    /// Creates a new receptive field analyzer
    pub fn new() -> Self {
        Self {
            field_size: 1.0,
            orientation_preference: 0.0,
            spatial_frequency_preference: 0.5,
            phase_preference: 0.0,
        }
    }

    /// Analyzes receptive field properties
    pub fn analyze(&self, input: &[f64], width: u32, height: u32) -> Result<ReceptiveField, AfiyahError> {
        // Calculate receptive field center
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;
        
        // Calculate receptive field size based on input
        let field_width = self.field_size * (width as f64).sqrt();
        let field_height = self.field_size * (height as f64).sqrt();
        
        Ok(ReceptiveField {
            center_x,
            center_y,
            width: field_width,
            height: field_height,
            orientation: self.orientation_preference,
            spatial_frequency: self.spatial_frequency_preference,
            phase: self.phase_preference,
        })
    }
}

/// Orientation selectivity analyzer
pub struct OrientationSelectivityAnalyzer {
    pub preferred_orientation: f64,
    pub tuning_width: f64,
    pub response_amplitude: f64,
}

impl OrientationSelectivityAnalyzer {
    /// Creates a new orientation selectivity analyzer
    pub fn new() -> Self {
        Self {
            preferred_orientation: 0.0,
            tuning_width: 30.0,
            response_amplitude: 1.0,
        }
    }

    /// Analyzes orientation selectivity
    pub fn analyze(&self, responses: &[f64], orientations: &[f64]) -> Result<f64, AfiyahError> {
        if responses.is_empty() || orientations.is_empty() {
            return Ok(0.0);
        }
        
        // Find the orientation with maximum response
        let max_idx = responses.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        let max_response = responses[max_idx];
        let preferred_orientation = orientations[max_idx];
        
        // Calculate tuning width (simplified)
        let tuning_width = self.calculate_tuning_width(responses, orientations, max_idx);
        
        Ok(tuning_width)
    }

    fn calculate_tuning_width(&self, responses: &[f64], orientations: &[f64], max_idx: usize) -> f64 {
        let max_response = responses[max_idx];
        let half_max = max_response / 2.0;
        
        // Find orientations where response is at least half maximum
        let mut tuning_width = 0.0;
        for (i, &response) in responses.iter().enumerate() {
            if response >= half_max {
                let orientation_diff = (orientations[i] - orientations[max_idx]).abs();
                tuning_width = tuning_width.max(orientation_diff);
            }
        }
        
        tuning_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_cells_creation() {
        let cells = SimpleCells::new();
        assert!(cells.is_ok());
    }

    #[test]
    fn test_simple_cell_state_defaults() {
        let state = SimpleCellState::default();
        assert_eq!(state.activity_level, 0.0);
        assert_eq!(state.adaptation_level, 0.5);
    }

    #[test]
    fn test_receptive_field_analyzer() {
        let analyzer = ReceptiveFieldAnalyzer::new();
        let input = vec![0.5; 100];
        let field = analyzer.analyze(&input, 10, 10);
        assert!(field.is_ok());
    }

    #[test]
    fn test_orientation_selectivity_analyzer() {
        let analyzer = OrientationSelectivityAnalyzer::new();
        let responses = vec![0.1, 0.8, 0.3, 0.2];
        let orientations = vec![0.0, 45.0, 90.0, 135.0];
        let selectivity = analyzer.analyze(&responses, &orientations);
        assert!(selectivity.is_ok());
    }
}