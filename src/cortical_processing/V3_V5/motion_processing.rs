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

//! Motion Processing Module
//! 
//! This module implements motion processing algorithms inspired by V5/MT area
//! of the visual cortex, including motion energy detection and global motion integration.

use crate::AfiyahError;
use std::f64::consts::PI;

/// Motion energy detector
/// Models motion-sensitive neurons in V5/MT area
#[derive(Debug, Clone)]
pub struct MotionEnergyDetector {
    pub preferred_direction: f64,    // Preferred motion direction in radians
    pub preferred_speed: f64,        // Preferred motion speed (pixels/frame)
    pub spatial_frequency: f64,      // Preferred spatial frequency
    pub temporal_frequency: f64,     // Preferred temporal frequency
    pub adaptation_rate: f64,        // Real-time adaptation rate
    pub current_response: f64,       // Current response level
}

impl MotionEnergyDetector {
    pub fn new(direction: f64, speed: f64) -> Result<Self, AfiyahError> {
        if direction < 0.0 || direction > 2.0 * PI {
            return Err(AfiyahError::BiologicalValidation(
                "Direction must be between 0 and 2π".to_string()
            ));
        }
        
        Ok(Self {
            preferred_direction: direction,
            preferred_speed: speed,
            spatial_frequency: 2.0,  // Default spatial frequency
            temporal_frequency: 4.0,  // Default temporal frequency
            adaptation_rate: 0.1,
            current_response: 0.0,
        })
    }

    /// Computes motion energy response for a sequence of frames
    pub fn compute_response(&mut self, frame_sequence: &[Vec<f64>], width: u32, height: u32) -> Result<f64, AfiyahError> {
        if frame_sequence.len() < 2 {
            return Err(AfiyahError::InputError("Need at least 2 frames for motion detection".to_string()));
        }
        
        let mut energy_response = 0.0;
        
        // Compute motion energy using spatiotemporal filtering
        for t in 0..frame_sequence.len() - 1 {
            let current_frame = &frame_sequence[t];
            let next_frame = &frame_sequence[t + 1];
            
            let motion_energy = self.compute_frame_motion_energy(current_frame, next_frame, width, height)?;
            energy_response += motion_energy;
        }
        
        // Normalize by number of frame pairs
        energy_response /= (frame_sequence.len() - 1) as f64;
        
        // Apply real-time adaptation
        self.adapt_response(energy_response);
        
        Ok(energy_response)
    }

    /// Computes motion energy between two consecutive frames
    fn compute_frame_motion_energy(&self, frame1: &[f64], frame2: &[f64], width: u32, height: u32) -> Result<f64, AfiyahError> {
        let mut energy = 0.0;
        
        for y in 0..height {
            for x in 0..width {
                let pixel_index = (y * width + x) as usize;
                if pixel_index < frame1.len() && pixel_index < frame2.len() {
                    // Compute temporal derivative
                    let temporal_derivative = frame2[pixel_index] - frame1[pixel_index];
                    
                    // Apply directional filter
                    let directional_response = self.apply_directional_filter(x as f64, y as f64, temporal_derivative);
                    
                    energy += directional_response * directional_response;
                }
            }
        }
        
        Ok(energy.sqrt())
    }

    /// Applies directional filter based on preferred direction
    fn apply_directional_filter(&self, x: f64, y: f64, temporal_derivative: f64) -> f64 {
        let cos_theta = self.preferred_direction.cos();
        let sin_theta = self.preferred_direction.sin();
        
        // Project spatial coordinates onto preferred direction
        let projection = x * cos_theta + y * sin_theta;
        
        // Apply spatiotemporal filter
        let spatial_filter = (2.0 * PI * self.spatial_frequency * projection).cos();
        let temporal_filter = (2.0 * PI * self.temporal_frequency * temporal_derivative).cos();
        
        spatial_filter * temporal_filter * temporal_derivative
    }

    /// Applies real-time adaptation
    fn adapt_response(&mut self, new_response: f64) {
        let adaptation_factor = 1.0 - self.adaptation_rate * self.current_response.abs();
        self.current_response = new_response * adaptation_factor.max(0.1);
    }
}

/// Global motion integrator
/// Models global motion integration in V5/MT area
#[derive(Debug)]
pub struct GlobalMotionIntegrator {
    pub motion_detectors: Vec<MotionEnergyDetector>,
    pub integration_window: u32,     // Number of frames for integration
    pub adaptation_enabled: bool,
}

impl GlobalMotionIntegrator {
    pub fn new(num_directions: u32, preferred_speed: f64) -> Result<Self, AfiyahError> {
        let mut motion_detectors = Vec::new();
        
        for i in 0..num_directions {
            let direction = (i as f64 * 2.0 * PI) / (num_directions as f64);
            motion_detectors.push(MotionEnergyDetector::new(direction, preferred_speed)?);
        }
        
        Ok(Self {
            motion_detectors,
            integration_window: 5,  // Default 5-frame integration window
            adaptation_enabled: true,
        })
    }

    /// Integrates motion signals across multiple directions
    pub fn integrate_motion(&mut self, frame_sequence: &[Vec<f64>], width: u32, height: u32) -> Result<GlobalMotionResponse, AfiyahError> {
        let mut direction_responses = Vec::new();
        let mut total_energy = 0.0;
        
        // Compute responses for all directions
        for detector in &mut self.motion_detectors {
            let response = detector.compute_response(frame_sequence, width, height)?;
            direction_responses.push(response);
            total_energy += response;
        }
        
        // Find dominant motion direction
        let dominant_direction = self.find_dominant_direction(&direction_responses)?;
        
        // Compute motion coherence
        let motion_coherence = self.compute_motion_coherence(&direction_responses, dominant_direction);
        
        Ok(GlobalMotionResponse {
            direction_responses,
            dominant_direction,
            motion_coherence,
            total_energy,
        })
    }

    /// Finds the dominant motion direction
    fn find_dominant_direction(&self, responses: &[f64]) -> Result<f64, AfiyahError> {
        if responses.is_empty() {
            return Err(AfiyahError::InputError("No motion responses to analyze".to_string()));
        }
        
        let max_index = responses.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let direction = (max_index as f64 * 2.0 * PI) / (self.motion_detectors.len() as f64);
        Ok(direction)
    }

    /// Computes motion coherence across directions
    fn compute_motion_coherence(&self, responses: &[f64], dominant_direction: f64) -> f64 {
        if responses.is_empty() {
            return 0.0;
        }
        
        let max_response = responses.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_response == 0.0 {
            return 0.0;
        }
        
        // Compute weighted average around dominant direction
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for (i, &response) in responses.iter().enumerate() {
            let direction = (i as f64 * 2.0 * PI) / (self.motion_detectors.len() as f64);
            let direction_diff = (direction - dominant_direction).abs();
            let weight = (-direction_diff).exp(); // Gaussian weight
            
            weighted_sum += response * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Sets the integration window size
    pub fn set_integration_window(&mut self, window_size: u32) {
        self.integration_window = window_size;
    }

    /// Enables or disables real-time adaptation
    pub fn set_adaptation(&mut self, enabled: bool) {
        self.adaptation_enabled = enabled;
    }
}

/// Response from global motion integration
#[derive(Debug, Clone)]
pub struct GlobalMotionResponse {
    pub direction_responses: Vec<f64>,
    pub dominant_direction: f64,
    pub motion_coherence: f64,
    pub total_energy: f64,
}

/// Motion vector for tracking
#[derive(Debug, Clone)]
pub struct MotionVector {
    pub direction: f64,
    pub magnitude: f64,
    pub confidence: f64,
}

impl MotionVector {
    pub fn new(direction: f64, magnitude: f64, confidence: f64) -> Self {
        Self {
            direction,
            magnitude,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_energy_detector_creation() {
        let detector = MotionEnergyDetector::new(PI / 4.0, 2.0);
        assert!(detector.is_ok());
    }

    #[test]
    fn test_global_motion_integrator_creation() {
        let integrator = GlobalMotionIntegrator::new(8, 2.0);
        assert!(integrator.is_ok());
    }

    #[test]
    fn test_motion_vector_creation() {
        let vector = MotionVector::new(PI / 2.0, 5.0, 0.8);
        assert_eq!(vector.direction, PI / 2.0);
        assert_eq!(vector.magnitude, 5.0);
        assert_eq!(vector.confidence, 0.8);
    }
  }
