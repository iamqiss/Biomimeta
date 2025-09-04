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

//! Koniocellular Pathway Implementation
//! 
//! This module implements the koniocellular (K) pathway for blue-yellow and
//! auxiliary processing. K cells have intermediate properties and are involved
//! in color processing, particularly blue-yellow opponency.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use crate::retinal_processing::amacrine_networks::AmacrineResponse;

/// Koniocellular pathway for blue-yellow and auxiliary processing
pub struct KoniocellularPathway {
    blue_yellow_processing: BlueYellowProcessing,
    auxiliary_processing: AuxiliaryProcessing,
    intermediate_properties: IntermediateProperties,
    adaptation_state: KoniocellularAdaptation,
}

impl KoniocellularPathway {
    /// Creates new koniocellular pathway with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            blue_yellow_processing: BlueYellowProcessing::new()?,
            auxiliary_processing: AuxiliaryProcessing::new()?,
            intermediate_properties: IntermediateProperties::new()?,
            adaptation_state: KoniocellularAdaptation::default(),
        })
    }

    /// Processes amacrine response through koniocellular pathway
    pub fn process(&self, input: &AmacrineResponse) -> Result<KoniocellularResponse, AfiyahError> {
        // Apply blue-yellow processing
        let blue_yellow_response = self.blue_yellow_processing.process(&input.lateral_inhibition)?;
        
        // Apply auxiliary processing
        let auxiliary_response = self.auxiliary_processing.process(&input.temporal_filtering)?;
        
        // Apply intermediate properties
        let intermediate_response = self.intermediate_properties.process(&input.lateral_inhibition)?;
        
        // Combine responses
        let combined_response = self.combine_responses(&blue_yellow_response, &auxiliary_response, &intermediate_response)?;
        
        Ok(KoniocellularResponse {
            color_signals: combined_response,
            blue_yellow_signals: blue_yellow_response,
            auxiliary_signals: auxiliary_response,
            intermediate_properties: self.intermediate_properties.properties.clone(),
            adaptation_level: self.adaptation_state.current_level,
        })
    }

    /// Calibrates koniocellular pathway based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.blue_yellow_processing.calibrate(params)?;
        self.auxiliary_processing.calibrate(params)?;
        self.intermediate_properties.calibrate(params)?;
        Ok(())
    }

    fn combine_responses(&self, blue_yellow: &[f64], auxiliary: &[f64], intermediate: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut combined = Vec::with_capacity(blue_yellow.len());
        
        for (i, ((&by_val, &aux_val), &int_val)) in blue_yellow.iter().zip(auxiliary.iter()).zip(intermediate.iter()).enumerate() {
            // Weighted combination: 50% blue-yellow, 30% auxiliary, 20% intermediate
            let combined_val = (by_val * 0.5) + (aux_val * 0.3) + (int_val * 0.2);
            combined.push(combined_val.min(1.0).max(0.0));
        }
        
        Ok(combined)
    }
}

/// Blue-yellow processing for koniocellular pathway
#[derive(Debug, Clone)]
pub struct BlueYellowProcessing {
    blue_sensitivity: f64,
    yellow_sensitivity: f64,
    blue_yellow_opponency: f64,
    chromatic_adaptation: f64,
}

impl BlueYellowProcessing {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            blue_sensitivity: 0.6,  // Moderate blue sensitivity
            yellow_sensitivity: 0.7, // Moderate yellow sensitivity
            blue_yellow_opponency: 0.8, // Strong blue-yellow opponency
            chromatic_adaptation: 0.5, // Chromatic adaptation
        })
    }

    pub fn process(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut blue_yellow_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // Apply blue-yellow processing (simplified)
            let blue_response = signal * self.blue_sensitivity;
            let yellow_response = signal * self.yellow_sensitivity;
            
            // Blue-yellow opponency
            let opponency_response = (blue_response - yellow_response) * self.blue_yellow_opponency;
            blue_yellow_response.push(opponency_response.abs().min(1.0));
        }
        
        Ok(blue_yellow_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust sensitivities based on adaptation
        self.blue_sensitivity = 0.6 * params.adaptation_rate;
        self.yellow_sensitivity = 0.7 * params.adaptation_rate;
        Ok(())
    }
}

/// Auxiliary processing for koniocellular pathway
#[derive(Debug, Clone)]
pub struct AuxiliaryProcessing {
    auxiliary_sensitivity: f64,
    temporal_processing: f64,
    spatial_processing: f64,
    integration_capability: f64,
}

impl AuxiliaryProcessing {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            auxiliary_sensitivity: 0.5,  // Moderate auxiliary sensitivity
            temporal_processing: 0.4,    // Lower temporal processing than M cells
            spatial_processing: 0.6,     // Lower spatial processing than P cells
            integration_capability: 0.7, // Good integration capability
        })
    }

    pub fn process(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut auxiliary_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // Apply auxiliary processing
            let auxiliary_signal = signal * self.auxiliary_sensitivity * self.integration_capability;
            auxiliary_response.push(auxiliary_signal.min(1.0).max(0.0));
        }
        
        Ok(auxiliary_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust auxiliary sensitivity based on adaptation
        self.auxiliary_sensitivity = 0.5 * params.adaptation_rate;
        Ok(())
    }
}

/// Intermediate properties for koniocellular pathway
#[derive(Debug, Clone)]
pub struct IntermediateProperties {
    properties: IntermediateCellProperties,
    receptive_field_size: f64,
    temporal_resolution: f64,
    spatial_resolution: f64,
}

impl IntermediateProperties {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            properties: IntermediateCellProperties {
                size: 0.2,           // Intermediate size
                temporal_resolution: 30.0, // Intermediate temporal resolution
                spatial_resolution: 0.7,   // Intermediate spatial resolution
                contrast_sensitivity: 0.6, // Intermediate contrast sensitivity
            },
            receptive_field_size: 0.2,  // Intermediate receptive field size
            temporal_resolution: 30.0,   // 30 Hz temporal resolution
            spatial_resolution: 0.7,     // Intermediate spatial resolution
        })
    }

    pub fn process(&self, input: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut intermediate_response = Vec::with_capacity(input.len());
        
        for &signal in input {
            // Apply intermediate properties
            let intermediate_signal = signal * self.spatial_resolution * self.properties.contrast_sensitivity;
            intermediate_response.push(intermediate_signal.min(1.0).max(0.0));
        }
        
        Ok(intermediate_response)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust properties based on adaptation
        self.spatial_resolution = 0.7 * params.adaptation_rate;
        self.properties.contrast_sensitivity = 0.6 * params.adaptation_rate;
        Ok(())
    }
}

/// Intermediate cell properties
#[derive(Debug, Clone)]
pub struct IntermediateCellProperties {
    pub size: f64,
    pub temporal_resolution: f64,
    pub spatial_resolution: f64,
    pub contrast_sensitivity: f64,
}

/// Koniocellular adaptation state
#[derive(Debug, Clone)]
pub struct KoniocellularAdaptation {
    pub current_level: f64,
    pub adaptation_rate: f64,
    pub blue_yellow_adaptation: f64,
    pub auxiliary_adaptation: f64,
}

impl Default for KoniocellularAdaptation {
    fn default() -> Self {
        Self {
            current_level: 0.5,
            adaptation_rate: 0.1,
            blue_yellow_adaptation: 0.5,
            auxiliary_adaptation: 0.5,
        }
    }
}

/// Response from koniocellular pathway processing
#[derive(Debug, Clone)]
pub struct KoniocellularResponse {
    pub color_signals: Vec<f64>,
    pub blue_yellow_signals: Vec<f64>,
    pub auxiliary_signals: Vec<f64>,
    pub intermediate_properties: IntermediateCellProperties,
    pub adaptation_level: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_koniocellular_pathway_creation() {
        let pathway = KoniocellularPathway::new();
        assert!(pathway.is_ok());
    }

    #[test]
    fn test_blue_yellow_processing() {
        let blue_yellow = BlueYellowProcessing::new().unwrap();
        assert_eq!(blue_yellow.blue_sensitivity, 0.6);
        assert_eq!(blue_yellow.yellow_sensitivity, 0.7);
        assert_eq!(blue_yellow.blue_yellow_opponency, 0.8);
    }

    #[test]
    fn test_auxiliary_processing() {
        let auxiliary = AuxiliaryProcessing::new().unwrap();
        assert_eq!(auxiliary.auxiliary_sensitivity, 0.5);
        assert_eq!(auxiliary.temporal_processing, 0.4);
        assert_eq!(auxiliary.spatial_processing, 0.6);
    }

    #[test]
    fn test_intermediate_properties() {
        let intermediate = IntermediateProperties::new().unwrap();
        assert_eq!(intermediate.receptive_field_size, 0.2);
        assert_eq!(intermediate.temporal_resolution, 30.0);
        assert_eq!(intermediate.spatial_resolution, 0.7);
    }

    #[test]
    fn test_koniocellular_adaptation() {
        let adaptation = KoniocellularAdaptation::default();
        assert_eq!(adaptation.current_level, 0.5);
        assert_eq!(adaptation.adaptation_rate, 0.1);
    }
}
