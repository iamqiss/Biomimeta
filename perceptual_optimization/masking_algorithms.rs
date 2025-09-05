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

//! Masking Algorithms Module
//! 
//! This module implements sophisticated perceptual masking algorithms based on
//! human visual system limitations and psychophysical studies.
//! 
//! Biological Basis:
//! - Watson & Solomon (1997): Contrast sensitivity functions
//! - Legge & Foley (1980): Contrast masking
//! - Peli (1990): Contrast sensitivity and masking
//! - Daly (1993): Visible differences predictor

use ndarray::Array2;
use crate::AfiyahError;

/// Masking parameters for perceptual optimization
#[derive(Debug, Clone)]
pub struct MaskingParams {
    pub contrast_threshold: f64,
    pub masking_strength: f64,
    pub spatial_frequency_weight: f64,
    pub temporal_frequency_weight: f64,
    pub luminance_adaptation: f64,
}

impl Default for MaskingParams {
    fn default() -> Self {
        Self {
            contrast_threshold: 0.1,
            masking_strength: 0.8,
            spatial_frequency_weight: 0.5,
            temporal_frequency_weight: 0.3,
            luminance_adaptation: 0.7,
        }
    }
}

/// Masking algorithm implementing biological perceptual masking
pub struct MaskingAlgorithm {
    params: MaskingParams,
    contrast_sensitivity: Array2<f64>,
    spatial_frequencies: Vec<f64>,
    temporal_frequencies: Vec<f64>,
}

impl MaskingAlgorithm {
    /// Creates a new masking algorithm
    pub fn new() -> Result<Self, AfiyahError> {
        let params = MaskingParams::default();
        let contrast_sensitivity = Self::create_contrast_sensitivity_function()?;
        let spatial_frequencies = vec![0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8];
        let temporal_frequencies = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

        Ok(Self {
            params,
            contrast_sensitivity,
            spatial_frequencies,
            temporal_frequencies,
        })
    }

    /// Applies perceptual masking to input
    pub fn apply_masking(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut masked_input = input.clone();

        // Apply contrast masking
        self.apply_contrast_masking(&mut masked_input)?;

        // Apply spatial frequency masking
        self.apply_spatial_frequency_masking(&mut masked_input)?;

        // Apply temporal masking (placeholder for temporal input)
        self.apply_temporal_masking(&mut masked_input)?;

        // Apply luminance adaptation
        self.apply_luminance_adaptation(&mut masked_input)?;

        Ok(masked_input)
    }

    fn create_contrast_sensitivity_function() -> Result<Array2<f64>, AfiyahError> {
        // Create contrast sensitivity function based on human visual system
        let height = 64;
        let width = 64;
        let mut csf = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let x = (j as f64) - (width as f64) / 2.0;
                let y = (i as f64) - (height as f64) / 2.0;
                let frequency = (x * x + y * y).sqrt();

                // Contrast sensitivity function (Watson & Solomon, 1997)
                let csf_value = self.calculate_csf_value(frequency);
                csf[[i, j]] = csf_value;
            }
        }

        Ok(csf)
    }

    fn calculate_csf_value(&self, frequency: f64) -> f64 {
        // Contrast sensitivity function based on human psychophysics
        if frequency < 0.1 {
            return 0.0;
        }

        let a = 2.6;
        let b = 0.0192;
        let c = 0.114;
        let d = 1.1;

        let csf = a * (b + c * frequency) * (-c * frequency).exp() * (1.0 + (frequency / d).powi(3));
        csf.max(0.0)
    }

    fn apply_contrast_masking(&self, input: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = input.dim();

        for i in 0..height {
            for j in 0..width {
                let contrast = input[[i, j]];
                let masking_threshold = self.calculate_masking_threshold(contrast)?;
                
                if contrast.abs() < masking_threshold {
                    input[[i, j]] = 0.0;
                } else {
                    input[[i, j]] = contrast * self.params.masking_strength;
                }
            }
        }

        Ok(())
    }

    fn apply_spatial_frequency_masking(&self, input: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = input.dim();

        for i in 0..height {
            for j in 0..width {
                let x = (j as f64) - (width as f64) / 2.0;
                let y = (i as f64) - (height as f64) / 2.0;
                let frequency = (x * x + y * y).sqrt();

                let csf_value = self.calculate_csf_value(frequency);
                let masking_factor = csf_value * self.params.spatial_frequency_weight;
                
                input[[i, j]] *= masking_factor;
            }
        }

        Ok(())
    }

    fn apply_temporal_masking(&self, input: &mut Array2<f64>) -> Result<(), AfiyahError> {
        // Placeholder for temporal masking
        // In a real implementation, this would analyze temporal changes
        let temporal_factor = 1.0 - self.params.temporal_frequency_weight * 0.1;
        
        for value in input.iter_mut() {
            *value *= temporal_factor;
        }

        Ok(())
    }

    fn apply_luminance_adaptation(&self, input: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = input.dim();
        let mut total_luminance = 0.0;
        let mut count = 0;

        // Calculate average luminance
        for value in input.iter() {
            total_luminance += value;
            count += 1;
        }

        if count > 0 {
            let mean_luminance = total_luminance / count as f64;
            let adaptation_factor = self.calculate_adaptation_factor(mean_luminance);

            for value in input.iter_mut() {
                *value *= adaptation_factor;
            }
        }

        Ok(())
    }

    fn calculate_masking_threshold(&self, contrast: f64) -> Result<f64, AfiyahError> {
        // Calculate masking threshold based on contrast
        let base_threshold = self.params.contrast_threshold;
        let masking_effect = contrast.abs() * self.params.masking_strength;
        Ok(base_threshold + masking_effect)
    }

    fn calculate_adaptation_factor(&self, luminance: f64) -> f64 {
        // Calculate luminance adaptation factor
        let adaptation_strength = self.params.luminance_adaptation;
        let log_luminance = luminance.ln().max(0.1);
        let adaptation_factor = 1.0 / (1.0 + log_luminance * adaptation_strength);
        adaptation_factor.clamp(0.1, 1.0)
    }

    /// Updates the masking parameters
    pub fn update_params(&mut self, params: MaskingParams) {
        self.params = params;
    }

    /// Gets current masking parameters
    pub fn get_params(&self) -> &MaskingParams {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masking_params_default() {
        let params = MaskingParams::default();
        assert_eq!(params.contrast_threshold, 0.1);
        assert_eq!(params.masking_strength, 0.8);
    }

    #[test]
    fn test_masking_algorithm_creation() {
        let algorithm = MaskingAlgorithm::new();
        assert!(algorithm.is_ok());
    }

    #[test]
    fn test_masking_application() {
        let algorithm = MaskingAlgorithm::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = algorithm.apply_masking(&input);
        assert!(result.is_ok());
        
        let masked_input = result.unwrap();
        assert_eq!(masked_input.dim(), (32, 32));
    }

    #[test]
    fn test_csf_calculation() {
        let algorithm = MaskingAlgorithm::new().unwrap();
        let csf_value = algorithm.calculate_csf_value(1.0);
        assert!(csf_value >= 0.0);
    }

    #[test]
    fn test_adaptation_factor_calculation() {
        let algorithm = MaskingAlgorithm::new().unwrap();
        let factor = algorithm.calculate_adaptation_factor(0.5);
        assert!(factor >= 0.1 && factor <= 1.0);
    }
}