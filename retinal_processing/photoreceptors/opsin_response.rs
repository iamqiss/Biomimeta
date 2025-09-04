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

//! Opsin Response Model Implementation
//! 
//! This module implements the opsin photopigment response model based on
//! biological research. Opsins are the light-sensitive proteins in photoreceptors
//! that initiate the phototransduction cascade.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;

/// Opsin response model for photopigment activation
pub struct OpsinResponseModel {
    rhodopsin_opsin: RhodopsinOpsin,
    cone_opsins: ConeOpsins,
    bleaching_model: BleachingModel,
    regeneration_model: RegenerationModel,
}

impl OpsinResponseModel {
    /// Creates new opsin response model with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            rhodopsin_opsin: RhodopsinOpsin::new()?,
            cone_opsins: ConeOpsins::new()?,
            bleaching_model: BleachingModel::new()?,
            regeneration_model: RegenerationModel::new()?,
        })
    }

    /// Processes rod and cone signals through opsin response
    pub fn process(&mut self, rod_signals: &[f64], cone_signals: &[f64]) -> Result<OpsinResponse, AfiyahError> {
        // Process rod signals through rhodopsin
        let rhodopsin_response = self.rhodopsin_opsin.process(rod_signals)?;
        
        // Process cone signals through cone opsins
        let cone_opsin_response = self.cone_opsins.process(cone_signals)?;
        
        // Apply bleaching model
        let bleached_response = self.bleaching_model.apply(&rhodopsin_response, &cone_opsin_response)?;
        
        // Apply regeneration model
        let regenerated_response = self.regeneration_model.apply(&bleached_response)?;
        
        Ok(regenerated_response)
    }

    /// Calibrates opsin response model based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.rhodopsin_opsin.calibrate(params)?;
        self.cone_opsins.calibrate(params)?;
        self.bleaching_model.calibrate(params)?;
        self.regeneration_model.calibrate(params)?;
        Ok(())
    }
}

/// Rhodopsin opsin for rod photoreceptors
#[derive(Debug, Clone)]
pub struct RhodopsinOpsin {
    absorption_peak: f64,        // Peak absorption wavelength (nm)
    quantum_efficiency: f64,     // Quantum efficiency of photon capture
    activation_energy: f64,      // Energy required for activation
    thermal_activation_rate: f64, // Rate of thermal activation
}

impl RhodopsinOpsin {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            absorption_peak: 500.0,      // Peak at 500nm (green-blue)
            quantum_efficiency: 0.67,    // 67% quantum efficiency
            activation_energy: 2.3e-19,  // Joules per photon
            thermal_activation_rate: 1e-6, // Thermal noise rate
        })
    }

    pub fn process(&self, rod_signals: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut opsin_responses = Vec::with_capacity(rod_signals.len());
        
        for &signal in rod_signals {
            // Apply quantum efficiency
            let quantum_response = signal * self.quantum_efficiency;
            
            // Apply thermal activation (noise)
            let thermal_noise = self.thermal_activation_rate * 0.1;
            let noisy_response = quantum_response + thermal_noise;
            
            // Ensure response is within biological limits
            opsin_responses.push(noisy_response.min(1.0).max(0.0));
        }
        
        Ok(opsin_responses)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Adjust quantum efficiency based on rod sensitivity
        self.quantum_efficiency = 0.67 * params.rod_sensitivity;
        Ok(())
    }
}

/// Cone opsins for color vision
#[derive(Debug, Clone)]
pub struct ConeOpsins {
    s_opsin: SOpsin,
    m_opsin: MOpsin,
    l_opsin: LOpsin,
}

impl ConeOpsins {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            s_opsin: SOpsin::new()?,
            m_opsin: MOpsin::new()?,
            l_opsin: LOpsin::new()?,
        })
    }

    pub fn process(&self, cone_signals: &[f64]) -> Result<Vec<f64>, AfiyahError> {
        let mut opsin_responses = Vec::with_capacity(cone_signals.len());
        
        for &signal in cone_signals {
            // Process through all cone opsin types
            let s_response = self.s_opsin.process(signal);
            let m_response = self.m_opsin.process(signal);
            let l_response = self.l_opsin.process(signal);
            
            // Combine responses (weighted by cone distribution)
            let combined_response = (s_response * 0.05) + (m_response * 0.4) + (l_response * 0.55);
            opsin_responses.push(combined_response);
        }
        
        Ok(opsin_responses)
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.s_opsin.calibrate(params)?;
        self.m_opsin.calibrate(params)?;
        self.l_opsin.calibrate(params)?;
        Ok(())
    }
}

/// S-opsin for short wavelength cones
#[derive(Debug, Clone)]
pub struct SOpsin {
    absorption_peak: f64,
    quantum_efficiency: f64,
}

impl SOpsin {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            absorption_peak: 420.0,      // Blue peak
            quantum_efficiency: 0.45,    // Lower efficiency than L/M opsins
        })
    }

    pub fn process(&self, signal: f64) -> f64 {
        signal * self.quantum_efficiency
    }

    pub fn calibrate(&mut self, _params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // S-opsin is relatively stable
        Ok(())
    }
}

/// M-opsin for medium wavelength cones
#[derive(Debug, Clone)]
pub struct MOpsin {
    absorption_peak: f64,
    quantum_efficiency: f64,
}

impl MOpsin {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            absorption_peak: 530.0,      // Green peak
            quantum_efficiency: 0.62,    // High efficiency
        })
    }

    pub fn process(&self, signal: f64) -> f64 {
        signal * self.quantum_efficiency
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.quantum_efficiency = 0.62 * params.cone_sensitivity;
        Ok(())
    }
}

/// L-opsin for long wavelength cones
#[derive(Debug, Clone)]
pub struct LOpsin {
    absorption_peak: f64,
    quantum_efficiency: f64,
}

impl LOpsin {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            absorption_peak: 560.0,      // Red peak
            quantum_efficiency: 0.65,    // Highest efficiency
        })
    }

    pub fn process(&self, signal: f64) -> f64 {
        signal * self.quantum_efficiency
    }

    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.quantum_efficiency = 0.65 * params.cone_sensitivity;
        Ok(())
    }
}

/// Bleaching model for opsin photopigments
#[derive(Debug, Clone)]
pub struct BleachingModel {
    bleaching_rate: f64,
    bleaching_threshold: f64,
    current_bleaching_level: f64,
}

impl BleachingModel {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            bleaching_rate: 0.1,
            bleaching_threshold: 0.8,
            current_bleaching_level: 0.0,
        })
    }

    pub fn apply(&mut self, rhodopsin_response: &[f64], cone_response: &[f64]) -> Result<BleachedResponse, AfiyahError> {
        // Calculate average response level
        let avg_rhodopsin = rhodopsin_response.iter().sum::<f64>() / rhodopsin_response.len() as f64;
        let avg_cone = cone_response.iter().sum::<f64>() / cone_response.len() as f64;
        let avg_response = (avg_rhodopsin + avg_cone) / 2.0;
        
        // Update bleaching level
        if avg_response > self.bleaching_threshold {
            self.current_bleaching_level = (self.current_bleaching_level + self.bleaching_rate).min(1.0);
        } else {
            self.current_bleaching_level = (self.current_bleaching_level - self.bleaching_rate * 0.1).max(0.0);
        }
        
        // Apply bleaching to responses
        let bleached_rhodopsin: Vec<f64> = rhodopsin_response.iter()
            .map(|&x| x * (1.0 - self.current_bleaching_level))
            .collect();
        
        let bleached_cone: Vec<f64> = cone_response.iter()
            .map(|&x| x * (1.0 - self.current_bleaching_level))
            .collect();
        
        Ok(BleachedResponse {
            rhodopsin_response: bleached_rhodopsin,
            cone_response: bleached_cone,
            bleaching_level: self.current_bleaching_level,
        })
    }

    pub fn calibrate(&mut self, _params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Bleaching model is relatively stable
        Ok(())
    }
}

/// Regeneration model for opsin photopigments
#[derive(Debug, Clone)]
pub struct RegenerationModel {
    regeneration_rate: f64,
    regeneration_threshold: f64,
}

impl RegenerationModel {
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            regeneration_rate: 0.05,
            regeneration_threshold: 0.3,
        })
    }

    pub fn apply(&self, bleached_response: &BleachedResponse) -> Result<OpsinResponse, AfiyahError> {
        // Apply regeneration based on bleaching level
        let regeneration_factor = if bleached_response.bleaching_level > self.regeneration_threshold {
            1.0 + self.regeneration_rate
        } else {
            1.0
        };
        
        let regenerated_rhodopsin: Vec<f64> = bleached_response.rhodopsin_response.iter()
            .map(|&x| (x * regeneration_factor).min(1.0))
            .collect();
        
        let regenerated_cone: Vec<f64> = bleached_response.cone_response.iter()
            .map(|&x| (x * regeneration_factor).min(1.0))
            .collect();
        
        Ok(OpsinResponse {
            rhodopsin_response: regenerated_rhodopsin,
            cone_response: regenerated_cone,
            bleaching_level: bleached_response.bleaching_level,
            regeneration_factor,
        })
    }

    pub fn calibrate(&mut self, _params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        // Regeneration model is relatively stable
        Ok(())
    }
}

/// Response from opsin processing
#[derive(Debug, Clone)]
pub struct OpsinResponse {
    pub rhodopsin_response: Vec<f64>,
    pub cone_response: Vec<f64>,
    pub bleaching_level: f64,
    pub regeneration_factor: f64,
}

/// Bleached response from bleaching model
#[derive(Debug, Clone)]
pub struct BleachedResponse {
    pub rhodopsin_response: Vec<f64>,
    pub cone_response: Vec<f64>,
    pub bleaching_level: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opsin_response_model_creation() {
        let model = OpsinResponseModel::new();
        assert!(model.is_ok());
    }

    #[test]
    fn test_rhodopsin_opsin() {
        let rhodopsin = RhodopsinOpsin::new().unwrap();
        assert_eq!(rhodopsin.absorption_peak, 500.0);
        assert_eq!(rhodopsin.quantum_efficiency, 0.67);
    }

    #[test]
    fn test_cone_opsins() {
        let cone_opsins = ConeOpsins::new().unwrap();
        let signals = vec![0.1, 0.5, 0.9];
        let result = cone_opsins.process(&signals);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bleaching_model() {
        let mut bleaching = BleachingModel::new().unwrap();
        let rhodopsin = vec![0.1, 0.5, 0.9];
        let cone = vec![0.2, 0.6, 0.8];
        let result = bleaching.apply(&rhodopsin, &cone);
        assert!(result.is_ok());
    }

    #[test]
    fn test_regeneration_model() {
        let regeneration = RegenerationModel::new().unwrap();
        let bleached = BleachedResponse {
            rhodopsin_response: vec![0.1, 0.5, 0.9],
            cone_response: vec![0.2, 0.6, 0.8],
            bleaching_level: 0.5,
        };
        let result = regeneration.apply(&bleached);
        assert!(result.is_ok());
    }
}
