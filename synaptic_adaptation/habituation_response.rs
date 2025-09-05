//! Habituation Response Module

use ndarray::Array2;
use crate::AfiyahError;

/// Habituation state representing the current state of habituation
#[derive(Debug, Clone)]
pub struct HabituationState {
    pub habituation_level: f64,
    pub adaptation_rate: f64,
    pub recovery_rate: f64,
    pub threshold: f64,
}

impl HabituationState {
    pub fn new() -> Self {
        Self {
            habituation_level: 0.0,
            adaptation_rate: 0.1,
            recovery_rate: 0.05,
            threshold: 0.5,
        }
    }
}

/// Adaptation response from habituation processing
#[derive(Debug, Clone)]
pub struct AdaptationResponse {
    pub habituation_factor: f64,
    pub adaptation_strength: f64,
    pub recovery_time: f64,
    pub processing_efficiency: f64,
}

/// Habituation processor implementing biological habituation mechanisms
pub struct HabituationProcessor {
    habituation_state: HabituationState,
    adaptation_threshold: f64,
    recovery_threshold: f64,
    efficiency_factor: f64,
}

impl HabituationProcessor {
    /// Creates a new habituation processor
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            habituation_state: HabituationState::new(),
            adaptation_threshold: 0.3,
            recovery_threshold: 0.7,
            efficiency_factor: 0.8,
        })
    }

    /// Processes habituation for input weights
    pub fn process(&mut self, weights: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let mut adapted_weights = weights.clone();
        
        // Calculate habituation level based on input
        let habituation_level = self.calculate_habituation_level(weights)?;
        
        // Update habituation state
        self.update_habituation_state(habituation_level)?;
        
        // Apply habituation adaptation
        self.apply_habituation_adaptation(&mut adapted_weights)?;
        
        Ok(adapted_weights)
    }

    fn calculate_habituation_level(&self, weights: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut total_activity = 0.0;
        let mut count = 0;

        for weight in weights.iter() {
            total_activity += weight.abs();
            count += 1;
        }

        if count > 0 {
            let mean_activity = total_activity / count as f64;
            Ok(mean_activity.clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }

    fn update_habituation_state(&mut self, current_activity: f64) -> Result<(), AfiyahError> {
        // Update habituation level based on current activity
        if current_activity > self.adaptation_threshold {
            // Increase habituation (repetitive input)
            self.habituation_state.habituation_level = 
                (self.habituation_state.habituation_level + self.habituation_state.adaptation_rate)
                .min(1.0);
        } else if current_activity < self.recovery_threshold {
            // Decrease habituation (novel input)
            self.habituation_state.habituation_level = 
                (self.habituation_state.habituation_level - self.habituation_state.recovery_rate)
                .max(0.0);
        }

        Ok(())
    }

    fn apply_habituation_adaptation(&self, weights: &mut Array2<f64>) -> Result<(), AfiyahError> {
        let habituation_factor = 1.0 - self.habituation_state.habituation_level;
        
        // Apply habituation by reducing weight magnitudes
        for weight in weights.iter_mut() {
            *weight *= habituation_factor;
        }
        
        Ok(())
    }

    /// Calculates adaptation response
    pub fn calculate_adaptation_response(&self) -> Result<AdaptationResponse, AfiyahError> {
        let habituation_factor = 1.0 - self.habituation_state.habituation_level;
        let adaptation_strength = self.habituation_state.habituation_level;
        let recovery_time = 1.0 / self.habituation_state.recovery_rate;
        let processing_efficiency = habituation_factor * self.efficiency_factor;

        Ok(AdaptationResponse {
            habituation_factor,
            adaptation_strength,
            recovery_time,
            processing_efficiency,
        })
    }

    /// Resets habituation state
    pub fn reset_habituation(&mut self) {
        self.habituation_state.habituation_level = 0.0;
    }

    /// Updates the adaptation threshold
    pub fn set_adaptation_threshold(&mut self, threshold: f64) {
        self.adaptation_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Updates the recovery threshold
    pub fn set_recovery_threshold(&mut self, threshold: f64) {
        self.recovery_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Updates the efficiency factor
    pub fn set_efficiency_factor(&mut self, factor: f64) {
        self.efficiency_factor = factor.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_habituation_state_creation() {
        let state = HabituationState::new();
        assert_eq!(state.habituation_level, 0.0);
        assert_eq!(state.adaptation_rate, 0.1);
    }

    #[test]
    fn test_habituation_processor_creation() {
        let processor = HabituationProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_habituation_processing() {
        let mut processor = HabituationProcessor::new().unwrap();
        let weights = Array2::ones((32, 32));
        
        let result = processor.process(&weights);
        assert!(result.is_ok());
        
        let adapted_weights = result.unwrap();
        assert_eq!(adapted_weights.dim(), (32, 32));
    }

    #[test]
    fn test_adaptation_response() {
        let processor = HabituationProcessor::new().unwrap();
        let response = processor.calculate_adaptation_response().unwrap();
        
        assert!(response.habituation_factor >= 0.0 && response.habituation_factor <= 1.0);
        assert!(response.adaptation_strength >= 0.0 && response.adaptation_strength <= 1.0);
        assert!(response.processing_efficiency >= 0.0 && response.processing_efficiency <= 1.0);
    }

    #[test]
    fn test_habituation_reset() {
        let mut processor = HabituationProcessor::new().unwrap();
        processor.reset_habituation();
        assert_eq!(processor.habituation_state.habituation_level, 0.0);
    }
}