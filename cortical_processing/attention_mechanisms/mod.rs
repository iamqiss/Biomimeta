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

//! Attention Mechanisms Module
//! 
//! This module implements sophisticated attention mechanisms based on
//! neurophysiological studies of visual attention and eye movement control.
//! 
//! Biological Basis:
//! - Posner & Petersen (1990): Attention networks
//! - Corbetta & Shulman (2002): Dorsal and ventral attention systems
//! - Itti & Koch (2001): Computational models of visual attention
//! - Theeuwes (2010): Top-down vs bottom-up attention

use ndarray::{Array2, Array3};
use crate::AfiyahError;

// Re-export all sub-modules
pub mod foveal_prioritization;
pub mod saccade_prediction;
pub mod saliency_mapping;
pub mod multi_scale_attention;
pub mod temporal_attention;
pub mod cross_modal_attention;

// Re-export the main types
pub use foveal_prioritization::{FovealProcessor, FovealMap, FovealRegion};
pub use saccade_prediction::{SaccadePredictor, SaccadeVector, SaccadeTarget};
pub use saliency_mapping::{SaliencyMap, SaliencyProcessor, SaliencyRegion};
pub use multi_scale_attention::{MultiScaleProcessor, MultiScaleMap, MultiScaleConfig, AttentionRegion};
pub use temporal_attention::{TemporalAttentionProcessor, TemporalAttentionOutput, TemporalAttentionConfig, TemporalAttentionState};
pub use cross_modal_attention::{CrossModalProcessor, CrossModalOutput, CrossModalConfig, AudioFeatures, HapticFeatures};

/// Configuration for attention mechanisms
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub foveal_radius: f64,
    pub saccade_threshold: f64,
    pub saliency_weight: f64,
    pub attention_decay: f64,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            foveal_radius: 2.0, // degrees
            saccade_threshold: 0.7,
            saliency_weight: 0.8,
            attention_decay: 0.95,
        }
    }
}

/// Output from attention processing
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    pub foveal_map: FovealMap,
    pub saccade_targets: Vec<SaccadeTarget>,
    pub saliency_map: SaliencyMap,
    pub attention_weights: Array2<f64>,
    pub processing_confidence: f64,
}

/// Main attention processor that coordinates all attention mechanisms
pub struct AttentionProcessor {
    foveal_processor: FovealProcessor,
    saccade_predictor: SaccadePredictor,
    saliency_processor: SaliencyProcessor,
    multi_scale_processor: MultiScaleProcessor,
    temporal_processor: TemporalAttentionProcessor,
    cross_modal_processor: CrossModalProcessor,
    config: AttentionConfig,
}

impl AttentionProcessor {
    /// Creates a new attention processor with default configuration
    pub fn new() -> Result<Self, AfiyahError> {
        let config = AttentionConfig::default();
        Self::with_config(config)
    }

    /// Creates a new attention processor with custom configuration
    pub fn with_config(config: AttentionConfig) -> Result<Self, AfiyahError> {
        let foveal_processor = FovealProcessor::new()?;
        let saccade_predictor = SaccadePredictor::new()?;
        let saliency_processor = SaliencyProcessor::new()?;
        let multi_scale_processor = MultiScaleProcessor::new()?;
        let temporal_processor = TemporalAttentionProcessor::new()?;
        let cross_modal_processor = CrossModalProcessor::new()?;

        Ok(Self {
            foveal_processor,
            saccade_predictor,
            saliency_processor,
            multi_scale_processor,
            temporal_processor,
            cross_modal_processor,
            config,
        })
    }

    /// Processes attention mechanisms for visual input
    pub fn process_attention(&mut self, input: &Array2<f64>, previous_attention: Option<&AttentionOutput>) -> Result<AttentionOutput, AfiyahError> {
        // Process foveal prioritization
        let foveal_map = self.foveal_processor.process_foveal_prioritization(input)?;

        // Process saliency mapping
        let saliency_map = self.saliency_processor.compute_saliency(input)?;

        // Process multi-scale attention
        let multi_scale_map = self.multi_scale_processor.process_multi_scale_attention(input, None)?;

        // Process temporal attention
        let temporal_output = self.temporal_processor.process_temporal_attention(input)?;

        // Process cross-modal attention (without audio/haptic for now)
        let cross_modal_output = self.cross_modal_processor.process_cross_modal_attention(input, None, None)?;

        // Predict saccade targets
        let saccade_targets = self.saccade_predictor.predict_saccades(&foveal_map, &saliency_map)?;

        // Calculate integrated attention weights
        let attention_weights = self.calculate_integrated_attention_weights(
            &foveal_map, 
            &saliency_map, 
            &multi_scale_map, 
            &temporal_output, 
            &cross_modal_output
        )?;

        // Calculate processing confidence
        let processing_confidence = self.calculate_enhanced_processing_confidence(
            &foveal_map, 
            &saliency_map, 
            &multi_scale_map, 
            &temporal_output, 
            &cross_modal_output
        )?;

        Ok(AttentionOutput {
            foveal_map,
            saccade_targets,
            saliency_map,
            attention_weights,
            processing_confidence,
        })
    }

    /// Updates the attention configuration
    pub fn update_config(&mut self, config: AttentionConfig) -> Result<(), AfiyahError> {
        self.config = config;
        // Recreate components with new configuration
        self.foveal_processor = FovealProcessor::new()?;
        self.saccade_predictor = SaccadePredictor::new()?;
        self.saliency_processor = SaliencyProcessor::new()?;
        self.multi_scale_processor = MultiScaleProcessor::new()?;
        self.temporal_processor = TemporalAttentionProcessor::new()?;
        self.cross_modal_processor = CrossModalProcessor::new()?;
        Ok(())
    }

    /// Gets current attention configuration
    pub fn get_config(&self) -> &AttentionConfig {
        &self.config
    }

    fn calculate_integrated_attention_weights(&self, 
                                            foveal_map: &FovealMap, 
                                            saliency_map: &SaliencyMap,
                                            multi_scale_map: &MultiScaleMap,
                                            temporal_output: &TemporalAttentionOutput,
                                            cross_modal_output: &CrossModalOutput) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = foveal_map.weights.dim();
        let mut attention_weights = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let foveal_weight = foveal_map.weights[[i, j]];
                let saliency_weight = saliency_map.weights[[i, j]];
                let multi_scale_weight = multi_scale_map.integrated_map[[i, j]];
                let temporal_weight = temporal_output.current_attention[[i, j]];
                let cross_modal_weight = cross_modal_output.integrated_attention[[i, j]];
                
                // Weighted combination of all attention mechanisms
                let combined_weight = (
                    foveal_weight * 0.25 +
                    saliency_weight * 0.25 +
                    multi_scale_weight * 0.2 +
                    temporal_weight * 0.15 +
                    cross_modal_weight * 0.15
                );
                
                attention_weights[[i, j]] = combined_weight.min(1.0).max(0.0);
            }
        }

        Ok(attention_weights)
    }

    fn calculate_enhanced_processing_confidence(&self, 
                                              foveal_map: &FovealMap, 
                                              saliency_map: &SaliencyMap,
                                              multi_scale_map: &MultiScaleMap,
                                              temporal_output: &TemporalAttentionOutput,
                                              cross_modal_output: &CrossModalOutput) -> Result<f64, AfiyahError> {
        // Calculate confidence based on all attention mechanisms
        let foveal_strength = foveal_map.calculate_strength()?;
        let saliency_strength = saliency_map.calculate_strength()?;
        let multi_scale_strength = multi_scale_map.attention_regions.len() as f64 / 100.0; // Normalize by expected regions
        let temporal_consistency = temporal_output.temporal_consistency;
        let cross_modal_correlation = cross_modal_output.cross_modal_correlation;
        
        // Weighted combination of confidence measures
        let confidence = (
            foveal_strength * 0.3 +
            saliency_strength * 0.3 +
            multi_scale_strength * 0.2 +
            temporal_consistency * 0.1 +
            cross_modal_correlation * 0.1
        );
        
        Ok(confidence.min(1.0).max(0.0))
    }

    // Legacy methods for backward compatibility
    fn calculate_attention_weights(&self, foveal_map: &FovealMap, saliency_map: &SaliencyMap) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = foveal_map.weights.dim();
        let mut attention_weights = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let foveal_weight = foveal_map.weights[[i, j]];
                let saliency_weight = saliency_map.weights[[i, j]];
                
                // Combine foveal and saliency weights
                let combined_weight = foveal_weight * self.config.saliency_weight + 
                                    saliency_weight * (1.0 - self.config.saliency_weight);
                
                attention_weights[[i, j]] = combined_weight;
            }
        }

        Ok(attention_weights)
    }

    fn calculate_processing_confidence(&self, foveal_map: &FovealMap, saliency_map: &SaliencyMap) -> Result<f64, AfiyahError> {
        // Calculate confidence based on foveal and saliency map strength
        let foveal_strength = foveal_map.calculate_strength()?;
        let saliency_strength = saliency_map.calculate_strength()?;
        
        let confidence = (foveal_strength + saliency_strength) / 2.0;
        Ok(confidence.min(1.0).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.foveal_radius, 2.0);
        assert_eq!(config.saccade_threshold, 0.7);
    }

    #[test]
    fn test_attention_processor_creation() {
        let processor = AttentionProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_attention_processor_with_config() {
        let config = AttentionConfig {
            foveal_radius: 3.0,
            saccade_threshold: 0.8,
            saliency_weight: 0.9,
            attention_decay: 0.9,
        };
        let processor = AttentionProcessor::with_config(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_attention_processing() {
        let mut processor = AttentionProcessor::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = processor.process_attention(&input, None);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(output.processing_confidence >= 0.0 && output.processing_confidence <= 1.0);
    }
}