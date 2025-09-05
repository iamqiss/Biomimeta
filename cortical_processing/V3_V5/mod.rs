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

//! V3, V4, V5/MT, and MST Cortical Areas Module
//! 
//! This module implements the higher-order visual cortical areas including:
//! - V3: Motion and form processing
//! - V4: Color and shape processing  
//! - V5/MT: Motion processing and optic flow
//! - MST: Complex motion integration and heading direction
//! 
//! Based on neurophysiological studies of primate visual cortex.

use ndarray::{Array2, Array3, Array4};
use crate::AfiyahError;

// Re-export all sub-modules
pub mod motion_processing;
pub mod depth_integration;
pub mod object_recognition;
pub mod optic_flow;
pub mod heading_direction;

// Re-export the main types
pub use motion_processing::{MotionDetector, MotionField, MotionVector};
pub use depth_integration::{DepthProcessor, DepthMap, StereoDisparity};
pub use object_recognition::{ObjectDetector, ObjectFeatures};
pub use optic_flow::{OpticFlowProcessor, FlowField, FlowVector};
pub use heading_direction::{HeadingDetector, HeadingVector, NavigationField};

/// Configuration for V3-V5 processing
#[derive(Debug, Clone)]
pub struct V3V5Config {
    pub motion_sensitivity: f64,
    pub depth_resolution: usize,
    pub object_complexity: f64,
    pub flow_integration_window: usize,
    pub heading_accuracy: f64,
}

impl Default for V3V5Config {
    fn default() -> Self {
        Self {
            motion_sensitivity: 0.8,
            depth_resolution: 64,
            object_complexity: 0.7,
            flow_integration_window: 5,
            heading_accuracy: 0.9,
        }
    }
}

/// Output from V3-V5 processing
#[derive(Debug, Clone)]
pub struct V3V5Output {
    pub motion_field: MotionField,
    pub depth_map: DepthMap,
    pub object_features: ObjectFeatures,
    pub optic_flow: FlowField,
    pub heading_vector: HeadingVector,
    pub processing_confidence: f64,
}

/// Main V3-V5 processor that coordinates all higher-order visual processing
pub struct V3V5Processor {
    motion_detector: MotionDetector,
    depth_processor: DepthProcessor,
    object_detector: ObjectDetector,
    optic_flow_processor: OpticFlowProcessor,
    heading_detector: HeadingDetector,
    config: V3V5Config,
}

impl V3V5Processor {
    /// Creates a new V3-V5 processor with default configuration
    pub fn new() -> Result<Self, AfiyahError> {
        let config = V3V5Config::default();
        Self::with_config(config)
    }

    /// Creates a new V3-V5 processor with custom configuration
    pub fn with_config(config: V3V5Config) -> Result<Self, AfiyahError> {
        let motion_detector = MotionDetector::new()?;
        let depth_processor = DepthProcessor::new()?;
        let object_detector = ObjectDetector::new()?;
        let optic_flow_processor = OpticFlowProcessor::new()?;
        let heading_detector = HeadingDetector::new()?;

        Ok(Self {
            motion_detector,
            depth_processor,
            object_detector,
            optic_flow_processor,
            heading_detector,
            config,
        })
    }

    /// Processes input through the complete V3-V5 pipeline
    pub fn process(&mut self, input: &Array2<f64>, previous_frame: Option<&Array2<f64>>) -> Result<V3V5Output, AfiyahError> {
        // Stage 1: Motion detection and processing
        let motion_field = if let Some(prev) = previous_frame {
            self.motion_detector.detect_motion(input, prev)?
        } else {
            MotionField::default()
        };

        // Stage 2: Depth processing and stereo integration
        let depth_map = self.depth_processor.process_depth(input)?;

        // Stage 3: Object recognition and feature extraction
        let object_features = self.object_detector.extract_features(input)?;

        // Stage 4: Optic flow computation
        let optic_flow = if let Some(prev) = previous_frame {
            self.optic_flow_processor.compute_flow(input, prev)?
        } else {
            FlowField::default()
        };

        // Stage 5: Heading direction estimation
        let heading_vector = self.heading_detector.estimate_heading(&motion_field, &optic_flow)?;

        // Calculate overall processing confidence
        let processing_confidence = self.calculate_processing_confidence(&motion_field, &object_features)?;

        Ok(V3V5Output {
            motion_field,
            depth_map,
            object_features,
            optic_flow,
            heading_vector,
            processing_confidence,
        })
    }

    /// Updates the V3-V5 configuration
    pub fn update_config(&mut self, config: V3V5Config) -> Result<(), AfiyahError> {
        self.config = config;
        // Recreate components with new configuration
        self.motion_detector = MotionDetector::new()?;
        self.depth_processor = DepthProcessor::new()?;
        self.object_detector = ObjectDetector::new()?;
        self.optic_flow_processor = OpticFlowProcessor::new()?;
        self.heading_detector = HeadingDetector::new()?;
        Ok(())
    }

    /// Gets current V3-V5 configuration
    pub fn get_config(&self) -> &V3V5Config {
        &self.config
    }

    fn calculate_processing_confidence(&self, motion_field: &MotionField, object_features: &ObjectFeatures) -> Result<f64, AfiyahError> {
        // Calculate confidence based on motion coherence and object feature strength
        let motion_coherence = motion_field.calculate_coherence()?;
        let feature_strength = object_features.calculate_strength()?;
        
        let confidence = (motion_coherence + feature_strength) / 2.0;
        Ok(confidence.min(1.0).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v3v5_processor_creation() {
        let processor = V3V5Processor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_v3v5_processor_with_config() {
        let config = V3V5Config {
            motion_sensitivity: 0.9,
            depth_resolution: 128,
            object_complexity: 0.8,
            flow_integration_window: 7,
            heading_accuracy: 0.95,
        };
        let processor = V3V5Processor::with_config(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_v3v5_processing() {
        let mut processor = V3V5Processor::new().unwrap();
        
        // Create mock input
        let input = Array2::ones((64, 64));
        let previous = Array2::zeros((64, 64));
        
        let result = processor.process(&input, Some(&previous));
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(output.processing_confidence >= 0.0 && output.processing_confidence <= 1.0);
    }
}