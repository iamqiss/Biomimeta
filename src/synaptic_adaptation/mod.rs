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
//! This module implements synaptic adaptation mechanisms including Hebbian learning,
//! homeostatic plasticity, neuromodulation, and habituation response based on
//! biological research on neural plasticity and adaptation.

pub mod hebbian_learning;
pub mod homeostatic_plasticity;
pub mod neuromodulation;
pub mod habituation_response;

use crate::AfiyahError;
use crate::cortical_processing::CorticalOutput;

/// Main synaptic adaptation processor that orchestrates all adaptation mechanisms
pub struct SynapticAdaptation {
    hebbian_learning: hebbian_learning::HebbianLearning,
    homeostatic_plasticity: homeostatic_plasticity::HomeostaticPlasticity,
    neuromodulation: neuromodulation::Neuromodulation,
    habituation: habituation_response::HabituationResponse,
    adaptation_state: AdaptationState,
}

impl SynapticAdaptation {
    /// Creates a new synaptic adaptation processor with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            hebbian_learning: hebbian_learning::HebbianLearning::new()?,
            homeostatic_plasticity: homeostatic_plasticity::HomeostaticPlasticity::new()?,
            neuromodulation: neuromodulation::Neuromodulation::new()?,
            habituation: habituation_response::HabituationResponse::new()?,
            adaptation_state: AdaptationState::default(),
        })
    }

    /// Adapts cortical output using synaptic adaptation mechanisms
    pub fn adapt(&mut self, cortical_output: &CorticalOutput) -> Result<AdaptedOutput, AfiyahError> {
        // Stage 1: Hebbian learning (strengthen co-active connections)
        let hebbian_output = self.hebbian_learning.adapt(cortical_output)?;
        
        // Stage 2: Homeostatic plasticity (maintain network stability)
        let homeostatic_output = self.homeostatic_plasticity.adapt(&hebbian_output)?;
        
        // Stage 3: Neuromodulation (dopamine-like adaptive weighting)
        let neuromodulated_output = self.neuromodulation.adapt(&homeostatic_output)?;
        
        // Stage 4: Habituation response (reduce encoding overhead for repetitive content)
        let habituated_output = self.habituation.adapt(&neuromodulated_output)?;
        
        // Update adaptation state
        self.update_adaptation_state(&habituated_output)?;
        
        Ok(AdaptedOutput {
            adapted_features: habituated_output.adapted_features,
            synaptic_weights: habituated_output.synaptic_weights,
            adaptation_level: self.adaptation_state.current_level,
            learning_rate: self.adaptation_state.learning_rate,
            plasticity_index: self.calculate_plasticity_index(&habituated_output),
        })
    }

    /// Trains synaptic adaptation mechanisms
    pub fn train(&mut self, training_data: &[crate::VisualInput]) -> Result<(), AfiyahError> {
        // Train Hebbian learning
        self.hebbian_learning.train(training_data)?;
        
        // Train homeostatic plasticity
        self.homeostatic_plasticity.train(training_data)?;
        
        // Train neuromodulation
        self.neuromodulation.train(training_data)?;
        
        // Train habituation response
        self.habituation.train(training_data)?;
        
        Ok(())
    }

    fn update_adaptation_state(&mut self, output: &AdaptedOutput) -> Result<(), AfiyahError> {
        // Update adaptation level based on plasticity index
        self.adaptation_state.current_level = (self.adaptation_state.current_level * 0.9 + 
                                             output.plasticity_index * 0.1).min(1.0);
        
        // Update learning rate based on adaptation success
        if output.plasticity_index > 0.5 {
            self.adaptation_state.learning_rate = (self.adaptation_state.learning_rate * 1.01).min(0.1);
        } else {
            self.adaptation_state.learning_rate = (self.adaptation_state.learning_rate * 0.99).max(0.001);
        }
        
        Ok(())
    }

    fn calculate_plasticity_index(&self, output: &AdaptedOutput) -> f64 {
        // Calculate plasticity index based on synaptic weight changes
        let weight_variance = self.calculate_weight_variance(&output.synaptic_weights);
        let adaptation_strength = output.adaptation_level;
        
        (weight_variance * adaptation_strength).min(1.0)
    }

    fn calculate_weight_variance(&self, weights: &[f64]) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }
        
        let mean_weight = weights.iter().sum::<f64>() / weights.len() as f64;
        let variance = weights.iter()
            .map(|&w| (w - mean_weight).powi(2))
            .sum::<f64>() / weights.len() as f64;
        
        variance.sqrt()
    }
}

/// Output from Hebbian learning
#[derive(Debug, Clone)]
pub struct HebbianOutput {
    pub strengthened_connections: Vec<Connection>,
    pub weakened_connections: Vec<Connection>,
    pub learning_rate: f64,
    pub co_activity_strength: f64,
}

/// Output from homeostatic plasticity
#[derive(Debug, Clone)]
pub struct HomeostaticOutput {
    pub stabilized_weights: Vec<f64>,
    pub homeostatic_error: f64,
    pub target_activity: f64,
    pub current_activity: f64,
}

/// Output from neuromodulation
#[derive(Debug, Clone)]
pub struct NeuromodulatedOutput {
    pub modulated_weights: Vec<f64>,
    pub neuromodulator_level: f64,
    pub adaptation_signal: f64,
    pub reward_signal: f64,
}

/// Output from habituation response
#[derive(Debug, Clone)]
pub struct HabituatedOutput {
    pub adapted_features: Vec<f64>,
    pub synaptic_weights: Vec<f64>,
    pub habituation_level: f64,
    pub repetition_count: u32,
}

/// Final adapted output
#[derive(Debug, Clone)]
pub struct AdaptedOutput {
    pub adapted_features: Vec<f64>,
    pub synaptic_weights: Vec<f64>,
    pub adaptation_level: f64,
    pub learning_rate: f64,
    pub plasticity_index: f64,
}

/// Synaptic connection structure
#[derive(Debug, Clone)]
pub struct Connection {
    pub from_neuron: usize,
    pub to_neuron: usize,
    pub weight: f64,
    pub strength: f64,
    pub last_activity: f64,
}

/// Adaptation state
#[derive(Debug, Clone)]
pub struct AdaptationState {
    pub current_level: f64,
    pub learning_rate: f64,
    pub plasticity_threshold: f64,
    pub adaptation_rate: f64,
    pub stability_factor: f64,
}

impl Default for AdaptationState {
    fn default() -> Self {
        Self {
            current_level: 0.5,
            learning_rate: 0.01,
            plasticity_threshold: 0.3,
            adaptation_rate: 0.1,
            stability_factor: 0.8,
        }
    }
}

/// Synaptic weight update rule
#[derive(Debug, Clone)]
pub struct WeightUpdateRule {
    pub learning_rate: f64,
    pub decay_rate: f64,
    pub threshold: f64,
    pub max_weight: f64,
    pub min_weight: f64,
}

impl WeightUpdateRule {
    /// Creates a new weight update rule
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            decay_rate: 0.001,
            threshold: 0.1,
            max_weight: 1.0,
            min_weight: -1.0,
        }
    }

    /// Updates synaptic weight based on activity
    pub fn update_weight(&self, current_weight: f64, pre_activity: f64, post_activity: f64) -> f64 {
        // Hebbian learning rule: Δw = η * pre_activity * post_activity
        let weight_change = self.learning_rate * pre_activity * post_activity;
        
        // Apply decay
        let decayed_weight = current_weight * (1.0 - self.decay_rate);
        
        // Update weight with bounds
        let new_weight = decayed_weight + weight_change;
        new_weight.max(self.min_weight).min(self.max_weight)
    }
}

/// Plasticity analyzer
pub struct PlasticityAnalyzer {
    pub analysis_window: usize,
    pub plasticity_threshold: f64,
    pub stability_threshold: f64,
}

impl PlasticityAnalyzer {
    /// Creates a new plasticity analyzer
    pub fn new() -> Self {
        Self {
            analysis_window: 100,
            plasticity_threshold: 0.1,
            stability_threshold: 0.05,
        }
    }

    /// Analyzes plasticity in weight changes
    pub fn analyze_plasticity(&self, weight_history: &[f64]) -> Result<PlasticityAnalysis, AfiyahError> {
        if weight_history.len() < 2 {
            return Ok(PlasticityAnalysis {
                plasticity_index: 0.0,
                stability_index: 1.0,
                adaptation_rate: 0.0,
                is_plastic: false,
                is_stable: true,
            });
        }

        let recent_weights = if weight_history.len() > self.analysis_window {
            &weight_history[weight_history.len() - self.analysis_window..]
        } else {
            weight_history
        };

        // Calculate weight variance
        let mean_weight = recent_weights.iter().sum::<f64>() / recent_weights.len() as f64;
        let variance = recent_weights.iter()
            .map(|&w| (w - mean_weight).powi(2))
            .sum::<f64>() / recent_weights.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate plasticity index
        let plasticity_index = (std_dev / (mean_weight.abs() + 1e-6)).min(1.0);
        
        // Calculate stability index
        let stability_index = if std_dev < self.stability_threshold { 1.0 } else { 0.0 };
        
        // Calculate adaptation rate
        let weight_changes: Vec<f64> = recent_weights.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        let adaptation_rate = if !weight_changes.is_empty() {
            weight_changes.iter().sum::<f64>() / weight_changes.len() as f64
        } else {
            0.0
        };

        Ok(PlasticityAnalysis {
            plasticity_index,
            stability_index,
            adaptation_rate,
            is_plastic: plasticity_index > self.plasticity_threshold,
            is_stable: stability_index > 0.5,
        })
    }
}

/// Plasticity analysis result
#[derive(Debug, Clone)]
pub struct PlasticityAnalysis {
    pub plasticity_index: f64,
    pub stability_index: f64,
    pub adaptation_rate: f64,
    pub is_plastic: bool,
    pub is_stable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synaptic_adaptation_creation() {
        let adaptation = SynapticAdaptation::new();
        assert!(adaptation.is_ok());
    }

    #[test]
    fn test_adaptation_state_defaults() {
        let state = AdaptationState::default();
        assert_eq!(state.current_level, 0.5);
        assert_eq!(state.learning_rate, 0.01);
    }

    #[test]
    fn test_weight_update_rule() {
        let rule = WeightUpdateRule::new();
        let new_weight = rule.update_weight(0.5, 0.8, 0.6);
        assert!(new_weight >= rule.min_weight && new_weight <= rule.max_weight);
    }

    #[test]
    fn test_plasticity_analyzer() {
        let analyzer = PlasticityAnalyzer::new();
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let analysis = analyzer.analyze_plasticity(&weights);
        assert!(analysis.is_ok());
    }
}