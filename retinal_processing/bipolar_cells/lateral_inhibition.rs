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

//! Lateral Inhibition Implementation
//! 
//! This module implements lateral inhibition mechanisms in bipolar cells.
//! Lateral inhibition enhances contrast boundaries and spatial frequency
//! selectivity through inhibitory connections between neighboring cells.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use super::on_off_network::OnOffResponse;

/// Lateral inhibition network for contrast enhancement
pub struct LateralInhibition {
    inhibition_strength: f64,
    inhibition_radius: f64,
    spatial_frequency_tuning: f64,
    adaptation_state: InhibitionAdaptation,
}

impl LateralInhibition {
    /// Creates new lateral inhibition network with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            inhibition_strength: 0.3,
            inhibition_radius: 0.2,
            spatial_frequency_tuning: 0.5,
            adaptation_state: InhibitionAdaptation::default(),
        })
    }

    /// Processes ON/OFF response through lateral inhibition
    pub fn process(&self, input: &OnOffResponse) -> Result<InhibitedResponse, AfiyahError> {
        // Apply lateral inhibition to ON response
        let inhibited_on = self.apply_lateral_inhibition(&input.on_response)?;
        
        // Apply lateral inhibition to OFF response
        let inhibited_off = self.apply_lateral_inhibition(&input.off_response)?;
        
        // Calculate contrast enhancement
        let contrast_enhancement = self.calculate_contrast_enhancement(&inhibited_on, &inhibited_off)?;
        
        Ok(InhibitedResponse {
            inhibited_on,
            inhibited_off,
            contrast_enhancement,
            inhibition_strength: self.inhibition_strength,
            spatial_frequency_tuning: self.spatial_frequency_tuning,
        })
    }

    /// Calibrates lateral inhibition based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust inhibition strength based on adaptation rate
        self.inhibition_strength = 0.3 * params.adaptation_rate;
        Ok(())
    }

    fn apply_lateral_inhibition(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut inhibited = Vec::with_capacity(input.len());
        
        for (i, &signal) in input.iter().enumerate() {
            // Calculate lateral inhibition from neighboring cells
            let lateral_inhibition = self.calculate_lateral_inhibition(input, i)?;
            
            // Apply inhibition
            let inhibited_signal = signal - (lateral_inhibition * self.inhibition_strength);
            inhibited.push(inhibited_signal.max(0.0).min(1.0));
        }
        
        Ok(inhibited)
    }

    fn calculate_lateral_inhibition(&self, input: &[f64], center_index: usize) -> Result<f64, AfiyahError> {
        let mut inhibition_sum = 0.0;
        let mut neighbor_count = 0;
        
        // Calculate inhibition from neighboring cells within inhibition radius
        let inhibition_range = (self.inhibition_radius * input.len() as f64) as usize;
        
        for i in 0..input.len() {
            if i != center_index {
                let distance = (i as f64 - center_index as f64).abs() / input.len() as f64;
                
                if distance <= self.inhibition_radius {
                    inhibition_sum += input[i];
                    neighbor_count += 1;
                }
            }
        }
        
        if neighbor_count > 0 {
            Ok(inhibition_sum / neighbor_count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_contrast_enhancement(&self, on_response: &[f64], off_response: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut contrast = Vec::with_capacity(on_response.len());
        
        for (i, (&on, &off)) in on_response.iter().zip(off_response.iter()).enumerate() {
            // Contrast is the difference between ON and OFF responses
            let contrast_value = (on - off).abs();
            contrast.push(contrast_value);
        }
        
        Ok(contrast)
    }
}

/// Lateral inhibition adaptation state
#[derive(Debug, Clone)]
pub struct InhibitionAdaptation {
    pub current_level: f64,
    pub adaptation_rate: f64,
    pub inhibition_balance: f64,
}

impl Default for InhibitionAdaptation {
    fn default() -> Self {
        Self {
            current_level: 0.5,
            adaptation_rate: 0.1,
            inhibition_balance: 0.5,
        }
    }
}

/// Response from lateral inhibition processing
#[derive(Debug, Clone)]
pub struct InhibitedResponse {
    pub inhibited_on: Vec<f64>,
    pub inhibited_off: Vec<f64>,
    pub contrast_enhancement: Vec<f64>,
    pub inhibition_strength: f64,
    pub spatial_frequency_tuning: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lateral_inhibition_creation() {
        let inhibition = LateralInhibition::new();
        assert!(inhibition.is_ok());
    }

    #[test]
    fn test_lateral_inhibition_processing() {
        let inhibition = LateralInhibition::new().unwrap();
        let input = OnOffResponse {
            on_response: vec![0.1, 0.5, 0.9, 0.3],
            off_response: vec![0.2, 0.4, 0.8, 0.4],
            center_surround_ratio: 0.3,
            adaptation_level: 0.5,
        };
        
        let result = inhibition.process(&input);
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inhibited_on.len(), 4);
        assert_eq!(response.inhibited_off.len(), 4);
    }

    #[test]
    fn test_inhibition_strength() {
        let inhibition = LateralInhibition::new().unwrap();
        assert_eq!(inhibition.inhibition_strength, 0.3);
    }

    #[test]
    fn test_inhibition_radius() {
        let inhibition = LateralInhibition::new().unwrap();
        assert_eq!(inhibition.inhibition_radius, 0.2);
    }

    #[test]
    fn test_contrast_enhancement() {
        let inhibition = LateralInhibition::new().unwrap();
        let on = vec![0.1, 0.5, 0.9];
        let off = vec![0.2, 0.4, 0.8];
        let result = inhibition.calculate_contrast_enhancement(&on, &off);
        assert!(result.is_ok());
        let contrast = result.unwrap();
        assert_eq!(contrast.len(), 3);
    }
}
