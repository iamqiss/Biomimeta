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

//! Rod Photoreceptor Implementation
//! 
//! This module implements rod photoreceptors for low-light vision based on
//! biological research. Rods are responsible for scotopic (night) vision and
//! have high sensitivity but low temporal resolution.

use crate::AfiyahError;
use crate::retinal_processing::RetinalCalibrationParams;
use super::{SpatialSample, PhotoreceptorType};

/// Rod photoreceptor system for low-light vision
pub struct RodPhotoreceptors {
    sensitivity: f64,
    temporal_response: TemporalResponse,
    adaptation_curve: AdaptationCurve,
    noise_model: NoiseModel,
}

impl RodPhotoreceptors {
    /// Creates new rod photoreceptors with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            sensitivity: 1e6, // High sensitivity for low-light detection
            temporal_response: TemporalResponse::rod_default(),
            adaptation_curve: AdaptationCurve::rod_default(),
            noise_model: NoiseModel::rod_default(),
        })
    }

    /// Processes luminance data through rod photoreceptors
    pub fn process(&self, luminance_data: &[f64], spatial_samples: &[SpatialSample]) -> Result<Vec<f64>, AfiyahError> {
        if luminance_data.len() != spatial_samples.len() {
            return Err(AfiyahError::InputError(
                "Luminance data length must match spatial samples length".to_string()
            ));
        }

        let mut rod_signals = Vec::with_capacity(luminance_data.len());
        
        for (i, (&luminance, sample)) in luminance_data.iter().zip(spatial_samples.iter()).enumerate() {
            // Only process rod samples
            if matches!(sample.photoreceptor_type, PhotoreceptorType::Rod) {
                let rod_signal = self.process_single_rod(luminance, sample)?;
                rod_signals.push(rod_signal);
            } else {
                // For cone samples, use minimal rod contribution
                rod_signals.push(0.01);
            }
        }
        
        Ok(rod_signals)
    }

    /// Calibrates rod photoreceptors based on biological parameters
    pub fn calibrate(&mut self, params: &RetinalCalibrationParams) -> Result<(), AfiyahError> {
        self.sensitivity = params.rod_sensitivity;
        self.adaptation_curve.update_sensitivity(params.rod_sensitivity);
        Ok(())
    }

    fn process_single_rod(&self, luminance: f64, sample: &SpatialSample) -> Result<f64, AfiyahError> {
        // Apply sensitivity scaling based on spatial density
        let density_factor = sample.density / 150_000.0; // Normalize to peripheral density
        
        // Apply adaptation curve
        let adapted_luminance = self.adaptation_curve.apply(luminance);
        
        // Apply temporal response filtering
        let temporal_response = self.temporal_response.filter(adapted_luminance);
        
        // Apply noise model
        let noisy_response = self.noise_model.add_noise(temporal_response);
        
        // Scale by sensitivity and density
        let final_response = noisy_response * self.sensitivity * density_factor;
        
        // Ensure response is within biological limits
        Ok(final_response.min(1.0).max(0.0))
    }
}

/// Temporal response characteristics of rod photoreceptors
#[derive(Debug, Clone)]
pub struct TemporalResponse {
    pub time_constant: f64,      // Response time constant (ms)
    pub integration_time: f64,   // Temporal integration window (ms)
    pub cutoff_frequency: f64,   // High-frequency cutoff (Hz)
}

impl TemporalResponse {
    /// Creates default temporal response for rods
    pub fn rod_default() -> Self {
        Self {
            time_constant: 200.0,    // Rods are slower than cones
            integration_time: 100.0, // Longer integration for sensitivity
            cutoff_frequency: 5.0,   // Lower temporal resolution
        }
    }

    /// Applies temporal filtering to input signal
    pub fn filter(&self, input: f64) -> f64 {
        // Simple low-pass filter approximation
        let alpha = 1.0 / (1.0 + self.time_constant / 1000.0);
        input * alpha
    }
}

/// Adaptation curve for rod photoreceptors
#[derive(Debug, Clone)]
pub struct AdaptationCurve {
    pub dark_adapted_threshold: f64,
    pub light_adapted_threshold: f64,
    pub adaptation_rate: f64,
    pub current_threshold: f64,
}

impl AdaptationCurve {
    /// Creates default adaptation curve for rods
    pub fn rod_default() -> Self {
        Self {
            dark_adapted_threshold: 1e-6,  // Very sensitive in dark
            light_adapted_threshold: 1e-3, // Less sensitive in light
            adaptation_rate: 0.1,
            current_threshold: 1e-4,
        }
    }

    /// Applies adaptation to input luminance
    pub fn apply(&self, luminance: f64) -> f64 {
        if luminance > self.current_threshold {
            // Light adaptation
            (luminance - self.current_threshold) / (1.0 - self.current_threshold)
        } else {
            // Dark adaptation
            luminance / self.current_threshold
        }
    }

    /// Updates sensitivity based on calibration
    pub fn update_sensitivity(&mut self, sensitivity: f64) {
        self.dark_adapted_threshold = 1e-6 / sensitivity;
        self.light_adapted_threshold = 1e-3 / sensitivity;
    }
}

/// Noise model for rod photoreceptors
#[derive(Debug, Clone)]
pub struct NoiseModel {
    pub thermal_noise: f64,
    pub quantum_noise: f64,
    pub amplifier_noise: f64,
}

impl NoiseModel {
    /// Creates default noise model for rods
    pub fn rod_default() -> Self {
        Self {
            thermal_noise: 0.01,    // Thermal noise level
            quantum_noise: 0.005,   // Quantum noise level
            amplifier_noise: 0.002, // Amplifier noise level
        }
    }

    /// Adds noise to input signal
    pub fn add_noise(&self, signal: f64) -> f64 {
        // Simple noise model - in practice would use proper random number generation
        let total_noise = self.thermal_noise + self.quantum_noise + self.amplifier_noise;
        let noise_factor = 1.0 + (total_noise * 0.1); // Scale noise appropriately
        signal * noise_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rod_photoreceptors_creation() {
        let rods = RodPhotoreceptors::new();
        assert!(rods.is_ok());
    }

    #[test]
    fn test_rod_processing() {
        let rods = RodPhotoreceptors::new().unwrap();
        let luminance_data = vec![0.1, 0.5, 0.9];
        let spatial_samples = vec![
            SpatialSample { x: 0.0, y: 0.0, density: 150_000.0, photoreceptor_type: PhotoreceptorType::Rod },
            SpatialSample { x: 0.5, y: 0.5, density: 150_000.0, photoreceptor_type: PhotoreceptorType::Rod },
            SpatialSample { x: 1.0, y: 1.0, density: 150_000.0, photoreceptor_type: PhotoreceptorType::Rod },
        ];
        
        let result = rods.process(&luminance_data, &spatial_samples);
        assert!(result.is_ok());
        let signals = result.unwrap();
        assert_eq!(signals.len(), 3);
    }

    #[test]
    fn test_temporal_response() {
        let temporal = TemporalResponse::rod_default();
        assert_eq!(temporal.time_constant, 200.0);
        assert_eq!(temporal.cutoff_frequency, 5.0);
    }

    #[test]
    fn test_adaptation_curve() {
        let adaptation = AdaptationCurve::rod_default();
        assert_eq!(adaptation.dark_adapted_threshold, 1e-6);
        assert_eq!(adaptation.light_adapted_threshold, 1e-3);
    }
}
