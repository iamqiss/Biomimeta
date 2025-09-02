//! Cortical Processing Module
//! 
//! This module implements cortical processing algorithms inspired by the visual cortex,
//! including V1 orientation filters, V5/MT motion processing, and real-time adaptation.

pub mod v1_orientation_filters;
pub mod motion_processing;
pub mod real_time_adaptation;

use crate::AfiyahError;
use crate::retinal_processing::RetinalOutput;
use v1_orientation_filters::{OrientationFilterBank, OrientationResponse};
use motion_processing::{GlobalMotionIntegrator, GlobalMotionResponse};
use real_time_adaptation::{RealTimeAdaptationController, AdaptationInput, AdaptationOutput, InputType};

/// Main cortical processor that orchestrates all cortical processing stages
pub struct CorticalProcessor {
    orientation_bank: OrientationFilterBank,
    motion_integrator: GlobalMotionIntegrator,
    adaptation_controller: RealTimeAdaptationController,
    processing_config: CorticalProcessingConfig,
}

impl CorticalProcessor {
    /// Creates a new cortical processor with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            orientation_bank: OrientationFilterBank::new(8, 2.0)?, // 8 orientations, 2.0 cycles/degree
            motion_integrator: GlobalMotionIntegrator::new(8, 2.0)?, // 8 directions, 2.0 pixels/frame
            adaptation_controller: RealTimeAdaptationController::new(),
            processing_config: CorticalProcessingConfig::default(),
        })
    }

    /// Processes retinal input through the complete cortical pipeline
    pub fn process(&mut self, input: &RetinalOutput) -> Result<crate::CorticalOutput, AfiyahError> {
        // Stage 1: V1 orientation processing
        let orientation_response = self.process_orientation(input)?;
        
        // Stage 2: V5/MT motion processing
        let motion_response = self.process_motion(input)?;
        
        // Stage 3: Real-time adaptation
        let adaptation_response = self.process_adaptation(input)?;
        
        // Stage 4: Generate cortical output
        let cortical_output = self.generate_cortical_output(&orientation_response, &motion_response, &adaptation_response)?;
        
        Ok(cortical_output)
    }

    /// Processes orientation information using V1 filters
    fn process_orientation(&mut self, input: &RetinalOutput) -> Result<OrientationResponse, AfiyahError> {
        // Convert retinal output to orientation input format
        let orientation_input = self.convert_to_orientation_input(input);
        
        // Process through orientation filter bank
        self.orientation_bank.process(&orientation_input, 64, 64) // Assuming 64x64 processing window
    }

    /// Processes motion information using V5/MT area
    fn process_motion(&mut self, input: &RetinalOutput) -> Result<GlobalMotionResponse, AfiyahError> {
        // Convert retinal output to motion input format
        let motion_input = self.convert_to_motion_input(input);
        
        // Process through motion integrator
        self.motion_integrator.integrate_motion(&motion_input, 64, 64)
    }

    /// Processes real-time adaptation
    fn process_adaptation(&mut self, input: &RetinalOutput) -> Result<AdaptationOutput, AfiyahError> {
        // Convert retinal output to adaptation input format
        let adaptation_input = self.convert_to_adaptation_input(input);
        
        // Process through adaptation controller
        self.adaptation_controller.adapt_to_input(&adaptation_input)
    }

    /// Converts retinal output to orientation input format
    fn convert_to_orientation_input(&self, input: &RetinalOutput) -> Vec<f64> {
        // Combine magnocellular and parvocellular streams for orientation processing
        let mut orientation_input = Vec::new();
        
        // Use magnocellular stream for motion-sensitive orientation detection
        orientation_input.extend_from_slice(&input.magnocellular_stream);
        
        // Use parvocellular stream for fine detail orientation detection
        orientation_input.extend_from_slice(&input.parvocellular_stream);
        
        orientation_input
    }

    /// Converts retinal output to motion input format
    fn convert_to_motion_input(&self, input: &RetinalOutput) -> Vec<Vec<f64>> {
        // Create frame sequence from retinal output
        // For now, create a simple 2-frame sequence
        let mut motion_input = Vec::new();
        
        // Frame 1: current magnocellular stream
        motion_input.push(input.magnocellular_stream.clone());
        
        // Frame 2: modified version for motion detection
        let mut frame2 = input.magnocellular_stream.clone();
        for value in &mut frame2 {
            *value = (*value * 0.9).max(0.0); // Simulate motion
        }
        motion_input.push(frame2);
        
        motion_input
    }

    /// Converts retinal output to adaptation input format
    fn convert_to_adaptation_input(&self, input: &RetinalOutput) -> AdaptationInput {
        use std::time::Instant;
        
        // Combine all retinal streams for adaptation analysis
        let mut characteristics = Vec::new();
        characteristics.extend_from_slice(&input.magnocellular_stream);
        characteristics.extend_from_slice(&input.parvocellular_stream);
        characteristics.extend_from_slice(&input.koniocellular_stream);
        
        AdaptationInput {
            characteristics,
            timestamp: Instant::now(),
            input_type: InputType::Combined,
        }
    }

    /// Generates final cortical output
    fn generate_cortical_output(
        &self,
        orientation_response: &OrientationResponse,
        motion_response: &GlobalMotionResponse,
        adaptation_response: &AdaptationOutput,
    ) -> Result<crate::CorticalOutput, AfiyahError> {
        // Convert orientation responses to orientation maps
        let orientation_maps = self.convert_to_orientation_maps(orientation_response);
        
        // Convert motion responses to motion vectors
        let motion_vectors = self.convert_to_motion_vectors(motion_response);
        
        // Create depth maps (placeholder for now)
        let depth_maps = vec![crate::DepthMap {
            depth_values: vec![0.5; 100],
            confidence: 0.8,
        }];
        
        // Create saliency map (placeholder for now)
        let saliency_map = crate::SaliencyMap::default();
        
        // Create temporal prediction (placeholder for now)
        let temporal_prediction = crate::TemporalPrediction::default();
        
        // Compute cortical compression ratio
        let cortical_compression = self.compute_cortical_compression(
            orientation_response,
            motion_response,
            adaptation_response,
        );
        
        Ok(crate::CorticalOutput {
            orientation_maps,
            motion_vectors,
            depth_maps,
            saliency_map,
            temporal_prediction,
            cortical_compression,
        })
    }

    /// Converts orientation response to orientation maps
    fn convert_to_orientation_maps(&self, response: &OrientationResponse) -> Vec<crate::OrientationMap> {
        let mut maps = Vec::new();
        
        for (i, &strength) in response.complex_responses.iter().enumerate() {
            let orientation = (i as f64 * std::f64::consts::PI) / response.complex_responses.len() as f64;
            
            maps.push(crate::OrientationMap {
                orientation,
                strength,
                data: response.complex_responses.clone(),
            });
        }
        
        maps
    }

    /// Converts motion response to motion vectors
    fn convert_to_motion_vectors(&self, response: &GlobalMotionResponse) -> Vec<crate::MotionVector> {
        let mut vectors = Vec::new();
        
        for (i, &magnitude) in response.direction_responses.iter().enumerate() {
            let direction = (i as f64 * 2.0 * std::f64::consts::PI) / response.direction_responses.len() as f64;
            let confidence = magnitude / response.total_energy.max(0.001);
            
            vectors.push(crate::MotionVector {
                direction,
                magnitude,
                confidence,
            });
        }
        
        vectors
    }

    /// Computes cortical compression ratio
    fn compute_cortical_compression(
        &self,
        orientation_response: &OrientationResponse,
        motion_response: &GlobalMotionResponse,
        adaptation_response: &AdaptationOutput,
    ) -> f64 {
        // Compute compression based on orientation energy and motion coherence
        let orientation_compression = 1.0 - (orientation_response.orientation_energy / 10.0).min(1.0);
        let motion_compression = 1.0 - (motion_response.motion_coherence / 5.0).min(1.0);
        let adaptation_compression = 1.0 - adaptation_response.adaptation_confidence;
        
        // Weighted average of compression factors
        (orientation_compression * 0.4 + motion_compression * 0.4 + adaptation_compression * 0.2).min(0.95)
    }

    /// Sets cortical processing configuration
    pub fn set_config(&mut self, config: CorticalProcessingConfig) {
        self.processing_config = config;
    }

    /// Enables or disables real-time adaptation
    pub fn set_adaptation_enabled(&mut self, enabled: bool) {
        self.adaptation_controller.set_adaptation_enabled(enabled);
    }
}

/// Configuration for cortical processing
#[derive(Debug, Clone)]
pub struct CorticalProcessingConfig {
    pub orientation_enabled: bool,
    pub motion_enabled: bool,
    pub adaptation_enabled: bool,
    pub processing_resolution: (u32, u32),
    pub adaptation_window_ms: u64,
}

impl Default for CorticalProcessingConfig {
    fn default() -> Self {
        Self {
            orientation_enabled: true,
            motion_enabled: true,
            adaptation_enabled: true,
            processing_resolution: (64, 64),
            adaptation_window_ms: 100,
        }
    }
}
