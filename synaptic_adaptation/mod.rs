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

//! Synaptic Adaptation Module
//! 
//! This module implements sophisticated synaptic adaptation mechanisms based on
//! neurophysiological studies of synaptic plasticity and learning.
//! 
//! Biological Basis:
//! - Hebb (1949): Hebbian learning and synaptic plasticity
//! - Bliss & Lomo (1973): Long-term potentiation (LTP)
//! - Turrigiano (1999): Homeostatic plasticity
//! - Kandel (2001): Synaptic plasticity and memory

use ndarray::Array2;
use crate::AfiyahError;

// Re-export all sub-modules
pub mod hebbian_learning;
pub mod homeostatic_plasticity;
pub mod neuromodulation;
pub mod habituation_response;

// Re-export the main types
pub use hebbian_learning::{HebbianLearner, SynapticWeight, LearningRule};
pub use homeostatic_plasticity::{HomeostaticPlasticity, PlasticityState, AdaptationMechanism};
pub use neuromodulation::{Neuromodulator, NeuromodulatorySignal, ModulationType};
pub use habituation_response::{HabituationProcessor, HabituationState, AdaptationResponse};

/// Configuration for synaptic adaptation
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    pub learning_rate: f64,
    pub decay_rate: f64,
    pub plasticity_threshold: f64,
    pub adaptation_window: usize,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            decay_rate: 0.001,
            plasticity_threshold: 0.5,
            adaptation_window: 100,
        }
    }
}

/// Output from synaptic adaptation
#[derive(Debug, Clone)]
pub struct AdaptationOutput {
    pub synaptic_weights: Array2<f64>,
    pub learning_rate: f64,
    pub plasticity_state: PlasticityState,
    pub adaptation_level: f64,
    pub processing_confidence: f64,
}

/// Main synaptic adaptation processor
pub struct SynapticAdaptation {
    hebbian_learner: HebbianLearner,
    homeostatic_plasticity: HomeostaticPlasticity,
    neuromodulator: Neuromodulator,
    habituation_processor: HabituationProcessor,
    config: AdaptationConfig,
}

impl SynapticAdaptation {
    /// Creates a new synaptic adaptation processor
    pub fn new() -> Result<Self, AfiyahError> {
        let config = AdaptationConfig::default();
        Self::with_config(config)
    }

    /// Creates a new synaptic adaptation processor with custom configuration
    pub fn with_config(config: AdaptationConfig) -> Result<Self, AfiyahError> {
        let hebbian_learner = HebbianLearner::new()?;
        let homeostatic_plasticity = HomeostaticPlasticity::new()?;
        let neuromodulator = Neuromodulator::new()?;
        let habituation_processor = HabituationProcessor::new()?;

        Ok(Self {
            hebbian_learner,
            homeostatic_plasticity,
            neuromodulator,
            habituation_processor,
            config,
        })
    }

    /// Processes synaptic adaptation for input
    pub fn adapt(&mut self, input: &Array2<f64>) -> Result<AdaptationOutput, AfiyahError> {
        // Apply Hebbian learning
        let hebbian_output = self.hebbian_learner.learn(input)?;

        // Apply homeostatic plasticity
        let plasticity_output = self.homeostatic_plasticity.adapt(&hebbian_output)?;

        // Apply neuromodulation
        let neuromodulatory_output = self.neuromodulator.modulate(&plasticity_output)?;

        // Apply habituation
        let habituation_output = self.habituation_processor.process(&neuromodulatory_output)?;

        // Calculate overall adaptation level
        let adaptation_level = self.calculate_adaptation_level(&habituation_output)?;

        // Calculate processing confidence
        let processing_confidence = self.calculate_processing_confidence(&habituation_output)?;

        Ok(AdaptationOutput {
            synaptic_weights: habituation_output,
            learning_rate: self.config.learning_rate,
            plasticity_state: plasticity_output,
            adaptation_level,
            processing_confidence,
        })
    }

    /// Updates the adaptation configuration
    pub fn update_config(&mut self, config: AdaptationConfig) -> Result<(), AfiyahError> {
        self.config = config;
        // Recreate components with new configuration
        self.hebbian_learner = HebbianLearner::new()?;
        self.homeostatic_plasticity = HomeostaticPlasticity::new()?;
        self.neuromodulator = Neuromodulator::new()?;
        self.habituation_processor = HabituationProcessor::new()?;
        Ok(())
    }

    /// Gets current adaptation configuration
    pub fn get_config(&self) -> &AdaptationConfig {
        &self.config
    }

    fn calculate_adaptation_level(&self, weights: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut total_weight = 0.0;
        let mut count = 0;

        for weight in weights.iter() {
            total_weight += weight.abs();
            count += 1;
        }

        if count > 0 {
            Ok(total_weight / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_processing_confidence(&self, weights: &Array2<f64>) -> Result<f64, AfiyahError> {
        // Calculate confidence based on weight stability and magnitude
        let adaptation_level = self.calculate_adaptation_level(weights)?;
        let confidence = adaptation_level.min(1.0).max(0.0);
        Ok(confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptation_config_default() {
        let config = AdaptationConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.decay_rate, 0.001);
    }

    #[test]
    fn test_synaptic_adaptation_creation() {
        let adaptation = SynapticAdaptation::new();
        assert!(adaptation.is_ok());
    }

    #[test]
    fn test_synaptic_adaptation_with_config() {
        let config = AdaptationConfig {
            learning_rate: 0.02,
            decay_rate: 0.002,
            plasticity_threshold: 0.6,
            adaptation_window: 150,
        };
        let adaptation = SynapticAdaptation::with_config(config);
        assert!(adaptation.is_ok());
    }

    #[test]
    fn test_adaptation_processing() {
        let mut adaptation = SynapticAdaptation::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = adaptation.adapt(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(output.adaptation_level >= 0.0);
        assert!(output.processing_confidence >= 0.0 && output.processing_confidence <= 1.0);
    }
}