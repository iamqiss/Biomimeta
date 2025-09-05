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

//! Multi-Modal Integration Module
//! 
//! This module implements sophisticated multi-modal integration capabilities
//! based on biological cross-modal processing and attention mechanisms.
//! 
//! Biological Basis:
//! - Cross-modal attention and binding
//! - Audio-visual correlation for enhanced compression
//! - Multi-sensory integration principles
//! - Synesthetic processing for improved efficiency

use ndarray::Array2;
use crate::AfiyahError;

// Re-export all sub-modules
pub mod audio_visual_correlation;
pub mod cross_modal_attention;

// Re-export the main types
pub use audio_visual_correlation::{AudioVisualCorrelator, CorrelationMap, AudioVisualFeatures};
pub use cross_modal_attention::{CrossModalAttention, AttentionWeights, ModalIntegration};

/// Multi-modal integration processor
pub struct MultiModalProcessor {
    audio_visual_correlator: AudioVisualCorrelator,
    cross_modal_attention: CrossModalAttention,
    integration_params: IntegrationParams,
}

/// Integration parameters for multi-modal processing
#[derive(Debug, Clone)]
pub struct IntegrationParams {
    pub audio_weight: f64,
    pub visual_weight: f64,
    pub correlation_threshold: f64,
    pub attention_strength: f64,
    pub integration_window: usize,
}

impl Default for IntegrationParams {
    fn default() -> Self {
        Self {
            audio_weight: 0.3,
            visual_weight: 0.7,
            correlation_threshold: 0.5,
            attention_strength: 0.8,
            integration_window: 10,
        }
    }
}

impl MultiModalProcessor {
    /// Creates a new multi-modal processor
    pub fn new() -> Result<Self, AfiyahError> {
        let audio_visual_correlator = AudioVisualCorrelator::new()?;
        let cross_modal_attention = CrossModalAttention::new()?;
        let integration_params = IntegrationParams::default();

        Ok(Self {
            audio_visual_correlator,
            cross_modal_attention,
            integration_params,
        })
    }

    /// Processes multi-modal input for enhanced compression
    pub fn process_multi_modal(&mut self, visual_input: &Array2<f64>, audio_input: &[f64]) -> Result<Array2<f64>, AfiyahError> {
        // Extract audio-visual correlations
        let correlation_map = self.audio_visual_correlator.correlate(visual_input, audio_input)?;

        // Apply cross-modal attention
        let attention_weights = self.cross_modal_attention.compute_attention(visual_input, audio_input)?;

        // Integrate multi-modal information
        let integrated_output = self.integrate_modalities(visual_input, &correlation_map, &attention_weights)?;

        Ok(integrated_output)
    }

    fn integrate_modalities(&self, visual_input: &Array2<f64>, correlation_map: &CorrelationMap, attention_weights: &AttentionWeights) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = visual_input.dim();
        let mut integrated = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let visual_value = visual_input[[i, j]];
                let correlation_value = correlation_map.get_correlation(i, j)?;
                let attention_value = attention_weights.get_attention(i, j)?;

                // Integrate visual, correlation, and attention information
                let integrated_value = visual_value * self.integration_params.visual_weight +
                                    correlation_value * self.integration_params.audio_weight +
                                    attention_value * self.integration_params.attention_strength;

                integrated[[i, j]] = integrated_value.clamp(0.0, 1.0);
            }
        }

        Ok(integrated)
    }

    /// Updates integration parameters
    pub fn update_params(&mut self, params: IntegrationParams) {
        self.integration_params = params;
    }

    /// Gets current integration parameters
    pub fn get_params(&self) -> &IntegrationParams {
        &self.integration_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_params_default() {
        let params = IntegrationParams::default();
        assert_eq!(params.audio_weight, 0.3);
        assert_eq!(params.visual_weight, 0.7);
        assert_eq!(params.correlation_threshold, 0.5);
    }

    #[test]
    fn test_multi_modal_processor_creation() {
        let processor = MultiModalProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_multi_modal_processing() {
        let mut processor = MultiModalProcessor::new().unwrap();
        let visual_input = Array2::ones((32, 32));
        let audio_input = vec![0.5; 100];
        
        let result = processor.process_multi_modal(&visual_input, &audio_input);
        assert!(result.is_ok());
        
        let integrated_output = result.unwrap();
        assert_eq!(integrated_output.dim(), (32, 32));
    }

    #[test]
    fn test_parameter_update() {
        let mut processor = MultiModalProcessor::new().unwrap();
        let params = IntegrationParams {
            audio_weight: 0.4,
            visual_weight: 0.6,
            correlation_threshold: 0.6,
            attention_strength: 0.9,
            integration_window: 15,
        };
        
        processor.update_params(params);
        assert_eq!(processor.get_params().audio_weight, 0.4);
        assert_eq!(processor.get_params().visual_weight, 0.6);
    }
}