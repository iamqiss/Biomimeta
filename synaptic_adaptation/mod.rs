//! Synaptic Adaptation Module
//! 
//! This module implements the synaptic adaptation mechanisms that enable the visual system
//! to learn and adapt to changing visual patterns, mimicking the plasticity of biological synapses.

use ndarray::{Array2, Array3};
use crate::AfiyahError;
use crate::cortical_processing::V1::V1Output;

// Re-export all sub-modules
pub mod hebbian_learning;
pub mod homeostatic_plasticity;
pub mod neuromodulation;
pub mod habituation_response;

// Re-export the main types
pub use hebbian_learning::HebbianLearner;
pub use homeostatic_plasticity::HomeostaticController;
pub use neuromodulation::Neuromodulator;
pub use habituation_response::HabituationController;

/// Configuration for synaptic adaptation parameters
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    pub hebbian_learning_rate: f64,
    pub homeostatic_target: f64,
    pub neuromodulation_strength: f64,
    pub habituation_rate: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            hebbian_learning_rate: 0.01,
            homeostatic_target: 0.5,
            neuromodulation_strength: 0.1,
            habituation_rate: 0.05,
        }
    }
}

/// Output from synaptic adaptation processing
#[derive(Debug, Clone)]
pub struct AdaptationOutput {
    pub adapted_weights: Array3<f64>,
    pub plasticity_map: Array2<f64>,
    pub adaptation_metrics: AdaptationMetrics,
}

/// Metrics for tracking adaptation performance
#[derive(Debug, Clone)]
pub struct AdaptationMetrics {
    pub hebbian_activity: f64,
    pub homeostatic_error: f64,
    pub dopamine_level: f64,
    pub habituation_level: f64,
}

/// Main synaptic adaptation system that coordinates all adaptation mechanisms
pub struct SynapticAdaptation {
    hebbian_learner: HebbianLearner,
    homeostatic_controller: HomeostaticController,
    neuromodulator: Neuromodulator,
    habituation_controller: HabituationController,
    config: AdaptationConfig,
}

impl SynapticAdaptation {
    /// Creates a new synaptic adaptation system with default configuration
    pub fn new() -> Result<Self, AfiyahError> {
        let config = AdaptationConfig::default();
        Self::with_config(config)
    }

    /// Creates a new synaptic adaptation system with custom configuration
    pub fn with_config(config: AdaptationConfig) -> Result<Self, AfiyahError> {
        let hebbian_learner = HebbianLearner::new(config.hebbian_learning_rate)?;
        let homeostatic_controller = HomeostaticController::new(config.homeostatic_target)?;
        let neuromodulator = Neuromodulator::new(config.neuromodulation_strength)?;
        let habituation_controller = HabituationController::new(config.habituation_rate)?;

        Ok(Self {
            hebbian_learner,
            homeostatic_controller,
            neuromodulator,
            habituation_controller,
            config,
        })
    }

    /// Processes input through the complete synaptic adaptation pipeline
    pub fn adapt(&mut self, input: &V1Output) -> Result<AdaptationOutput, AfiyahError> {
        // Extract activity patterns from V1 output
        let activity_patterns = &input.simple_cell_responses;
        
        // Apply Hebbian learning
        let hebbian_weights = self.hebbian_learner.update_weights(activity_patterns)?;
        
        // Apply homeostatic plasticity
        let homeostatic_weights = self.homeostatic_controller.adjust_weights(&hebbian_weights, activity_patterns)?;
        
        // Apply neuromodulation
        let neuromodulated_weights = self.neuromodulator.modulate_weights(&homeostatic_weights, activity_patterns)?;
        
        // Apply habituation
        let final_weights = self.habituation_controller.apply_habituation(&neuromodulated_weights, activity_patterns)?;
        
        // Generate plasticity map
        let plasticity_map = self.generate_plasticity_map(&final_weights)?;
        
        // Collect adaptation metrics
        let adaptation_metrics = AdaptationMetrics {
            hebbian_activity: self.hebbian_learner.get_activity_level(),
            homeostatic_error: self.homeostatic_controller.get_error(),
            dopamine_level: self.neuromodulator.get_dopamine_level(),
            habituation_level: self.habituation_controller.get_habituation_level(),
        };

        Ok(AdaptationOutput {
            adapted_weights: final_weights,
            plasticity_map,
            adaptation_metrics,
        })
    }

    /// Generates a spatial map of synaptic plasticity
    fn generate_plasticity_map(&self, weights: &Array3<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (orientations, spatial_freqs, spatial_size) = weights.dim();
        let size = (spatial_size as f64).sqrt() as usize;
        
        if size * size != spatial_size {
            return Err(AfiyahError::InputError { 
                message: "Spatial size must be a perfect square for plasticity mapping".to_string() 
            });
        }

        let mut plasticity_map = Array2::zeros((size, size));
        
        for h in 0..size {
            for w in 0..size {
                let spatial_idx = h * size + w;
                let mut plasticity = 0.0;
                
                for o in 0..orientations {
                    for s in 0..spatial_freqs {
                        plasticity += weights[[o, s, spatial_idx]].abs();
                    }
                }
                
                plasticity_map[[h, w]] = plasticity / (orientations * spatial_freqs) as f64;
            }
        }
        
        Ok(plasticity_map)
    }

    /// Updates the adaptation configuration
    pub fn update_config(&mut self, config: AdaptationConfig) -> Result<(), AfiyahError> {
        self.config = config;
        // Recreate components with new configuration
        self.hebbian_learner = HebbianLearner::new(self.config.hebbian_learning_rate)?;
        self.homeostatic_controller = HomeostaticController::new(self.config.homeostatic_target)?;
        self.neuromodulator = Neuromodulator::new(self.config.neuromodulation_strength)?;
        self.habituation_controller = HabituationController::new(self.config.habituation_rate)?;
        Ok(())
    }

    /// Gets current adaptation metrics
    pub fn get_metrics(&self) -> AdaptationMetrics {
        AdaptationMetrics {
            hebbian_activity: self.hebbian_learner.get_activity_level(),
            homeostatic_error: self.homeostatic_controller.get_error(),
            dopamine_level: self.neuromodulator.get_dopamine_level(),
            habituation_level: self.habituation_controller.get_habituation_level(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cortical_processing::V1::V1Output;

    #[test]
    fn test_synaptic_adaptation_creation() {
        let adaptation = SynapticAdaptation::new();
        assert!(adaptation.is_ok());
    }

    #[test]
    fn test_adaptation_with_config() {
        let config = AdaptationConfig {
            hebbian_learning_rate: 0.02,
            homeostatic_target: 0.6,
            neuromodulation_strength: 0.15,
            habituation_rate: 0.08,
        };
        let adaptation = SynapticAdaptation::with_config(config);
        assert!(adaptation.is_ok());
    }

    #[test]
    fn test_adaptation_processing() {
        let mut adaptation = SynapticAdaptation::new().unwrap();
        
        // Create mock V1 output
        let mock_v1_output = V1Output {
            simple_responses: Array3::ones((8, 4, 64)),
            complex_responses: Array3::ones((8, 4, 64)),
            orientation_responses: Array3::ones((8, 4, 64)),
            edge_responses: Array3::ones((8, 4, 64)),
        };
        
        let result = adaptation.adapt(&mock_v1_output);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.adapted_weights.dim(), (8, 4, 64));
        assert_eq!(output.plasticity_map.dim(), (8, 8));
    }
}