//! Parvocellular Pathway Implementation
//! 
//! This module implements the parvocellular (P) pathway for fine detail and color
//! processing. P cells have small receptive fields, high spatial resolution,
//! and are sensitive to fine details and color information.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use crate::retinal_processing::amacrine_networks::AmacrineResponse;

/// Parvocellular pathway for fine detail and color processing
pub struct ParvocellularPathway {
    detail_processing: DetailProcessing,
    color_processing: ColorProcessing,
    spatial_resolution: SpatialResolution,
    adaptation_state: ParvocellularAdaptation,
}

impl ParvocellularPathway {
    /// Creates new parvocellular pathway with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            detail_processing: DetailProcessing::new()?,
            color_processing: ColorProcessing::new()?,
            spatial_resolution: SpatialResolution::new()?,
            adaptation_state: ParvocellularAdaptation::default(),
        })
    }

    /// Processes amacrine response through parvocellular pathway
    pub fn process(&self, input: &AmacrineResponse) -> Result<ParvocellularResponse, AfiyahError> {
        // Apply detail processing
        let detail_response = self.detail_processing.process(&input.lateral_inhibition)?;
        
        // Apply color processing
        let color_response = self.color_processing.process(&input.lateral_inhibition)?;
        
        // Apply spatial resolution enhancement
        let spatial_response = self.spatial_resolution.enhance(&input.lateral_inhibition)?;
        
        // Combine responses
        let combined_response = self.combine_responses(&detail_response, &color_response, &spatial_response)?;
        
        Ok(ParvocellularResponse {
            detail_signals: combined_response,
            color_signals: color_response,
            spatial_resolution: self.spatial_resolution.resolution,
            detail_sensitivity: self.detail_processing.detail_sensitivity,
            color_sensitivity: self.color_processing.color_sensitivity,
            adaptation_level: self.adaptation_state.current_level,
        })
    }

    /// Calibrates parvocellular pathway based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.detail_processing.calibrate(params)?;
        self.color_processing.calibrate(params)?;
        self.spatial_resolution.calibrate(params)?;
        Ok(())
    }

    fn combine_responses(&self, detail: &[f64], color: &[f64], spatial: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut combined = Vec::with_capacity(detail.len());
        
        for (i, ((&detail_val, &color_val), &spatial_val)) in detail.iter().zip(color.iter()).zip(spatial.iter()).enumerate() {
            // Weighted combination: 50% detail, 30% color, 20% spatial
            let combined_val = (detail_val * 0.5) + (color_val * 0.3) + (spatial_val * 0.2);
            combined.push(combined_val.min(1.0).max(0.0));
        }
        
        Ok(combined)
    }
}

/// Detail processing for parvocellular pathway
#[derive(Debug, Clone)]
pub struct DetailProcessing {
    detail_sensitivity: f64,
    detail_threshold: f64,
    edge_enhancement: f64,
    texture_analysis: f64,
}

impl DetailProcessing {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            detail_sensitivity: 0.9,  // High sensitivity to details
            detail_threshold: 0.05,   // Low threshold for detail detection
            edge_enhancement: 0.8,    // Strong edge enhancement
            texture_analysis: 0.7,    // Texture analysis capability
        })
    }

    pub fn process(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut detail_response = Vec::with_capacity(input.len());
        
        for (i, &signal) in input.iter().enumerate() {
            // Calculate detail based on local variations
            let detail = self.calculate_detail(input, i);
            
            // Apply detail sensitivity and threshold
            let detail_response_val = if detail > self.detail_threshold {
                detail * self.detail_sensitivity * self.edge_enhancement
            } else {
                0.0
            };
            
            detail_response.push(detail_response_val.min(1.0));
        }
        
        Ok(detail_response)
    }

    fn calculate_detail(&self, input: &[f64], index: usize) -> f64 {
        if index == 0 || index >= input.len() - 1 {
            return 0.0;
        }
        
        // Calculate detail as local variation
        let left = input[index - 1];
        let right = input[index + 1];
        let center = input[index];
        
        // Detail is the local variance
        let variance = ((left - center).powi(2) + (right - center).powi(2)) / 2.0;
        variance.sqrt()
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust detail sensitivity based on adaptation
        self.detail_sensitivity = 0.9 * params.adaptation_rate;
        Ok(())
    }
}

/// Color processing for parvocellular pathway
#[derive(Debug, Clone)]
pub struct ColorProcessing {
    color_sensitivity: f64,
    color_opponency: f64,
    chromatic_adaptation: f64,
    color_constancy: f64,
}

impl ColorProcessing {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            color_sensitivity: 0.8,  // High color sensitivity
            color_opponency: 0.7,    // Strong color opponency
            chromatic_adaptation: 0.6, // Chromatic adaptation
            color_constancy: 0.5,    // Color constancy
        })
    }

    pub fn process(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut color_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // Apply color processing (simplified)
            let color_response_val = signal * self.color_sensitivity * self.color_opponency;
            color_response.push(color_response_val.min(1.0).max(0.0));
        }
        
        Ok(color_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust color sensitivity based on adaptation
        self.color_sensitivity = 0.8 * params.adaptation_rate;
        Ok(())
    }
}

/// Spatial resolution for parvocellular pathway
#[derive(Debug, Clone)]
pub struct SpatialResolution {
    resolution: f64,
    receptive_field_size: f64,
    spatial_frequency_tuning: f64,
    acuity_enhancement: f64,
}

impl SpatialResolution {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            resolution: 0.9,         // High spatial resolution
            receptive_field_size: 0.1, // Small receptive fields
            spatial_frequency_tuning: 0.8, // High spatial frequency tuning
            acuity_enhancement: 0.9,  // Strong acuity enhancement
        })
    }

    pub fn enhance(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut enhanced_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // Apply spatial resolution enhancement
            let enhanced_signal = signal * self.resolution * self.acuity_enhancement;
            enhanced_response.push(enhanced_signal.min(1.0).max(0.0));
        }
        
        Ok(enhanced_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust resolution based on adaptation
        self.resolution = 0.9 * params.adaptation_rate;
        Ok(())
    }
}

/// Parvocellular adaptation state
#[derive(Debug, Clone)]
pub struct ParvocellularAdaptation {
    pub current_level: f64,
    pub adaptation_rate: f64,
    pub detail_adaptation: f64,
    pub color_adaptation: f64,
}

impl Default for ParvocellularAdaptation {
    fn default() -> Self {
        Self {
            current_level: 0.5,
            adaptation_rate: 0.1,
            detail_adaptation: 0.5,
            color_adaptation: 0.5,
        }
    }
}

/// Response from parvocellular pathway processing
#[derive(Debug, Clone)]
pub struct ParvocellularResponse {
    pub detail_signals: Vec<f64>,
    pub color_signals: Vec<f64>,
    pub spatial_resolution: f64,
    pub detail_sensitivity: f64,
    pub color_sensitivity: f64,
    pub adaptation_level: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parvocellular_pathway_creation() {
        let pathway = ParvocellularPathway::new();
        assert!(pathway.is_ok());
    }

    #[test]
    fn test_detail_processing() {
        let detail = DetailProcessing::new().unwrap();
        let input = vec![0.1, 0.5, 0.9, 0.3];
        let result = detail.process(&input);
        assert!(result.is_ok());
        let details = result.unwrap();
        assert_eq!(details.len(), 4);
    }

    #[test]
    fn test_color_processing() {
        let color = ColorProcessing::new().unwrap();
        assert_eq!(color.color_sensitivity, 0.8);
        assert_eq!(color.color_opponency, 0.7);
    }

    #[test]
    fn test_spatial_resolution() {
        let spatial = SpatialResolution::new().unwrap();
        assert_eq!(spatial.resolution, 0.9);
        assert_eq!(spatial.receptive_field_size, 0.1);
    }

    #[test]
    fn test_parvocellular_adaptation() {
        let adaptation = ParvocellularAdaptation::default();
        assert_eq!(adaptation.current_level, 0.5);
        assert_eq!(adaptation.adaptation_rate, 0.1);
    }
}