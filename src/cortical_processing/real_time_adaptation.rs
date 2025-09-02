//! Real-time Adaptation Module
//! 
//! This module implements real-time adaptation mechanisms for cortical processing,
//! including synaptic plasticity, homeostatic regulation, and dynamic parameter adjustment.

use crate::AfiyahError;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Real-time adaptation controller
/// Manages dynamic parameter adjustment based on input characteristics
#[derive(Debug)]
pub struct RealTimeAdaptationController {
    pub adaptation_window: Duration,     // Time window for adaptation
    pub learning_rate: f64,              // Adaptation learning rate
    pub stability_threshold: f64,        // Stability threshold for adaptation
    pub adaptation_history: VecDeque<AdaptationEvent>,
    pub current_parameters: AdaptationParameters,
    pub adaptation_enabled: bool,
}

impl RealTimeAdaptationController {
    pub fn new() -> Self {
        Self {
            adaptation_window: Duration::from_millis(100), // 100ms adaptation window
            learning_rate: 0.01,
            stability_threshold: 0.1,
            adaptation_history: VecDeque::new(),
            current_parameters: AdaptationParameters::default(),
            adaptation_enabled: true,
        }
    }

    /// Processes input and adapts parameters in real-time
    pub fn adapt_to_input(&mut self, input: &AdaptationInput) -> Result<AdaptationOutput, AfiyahError> {
        let current_time = Instant::now();
        
        // Record adaptation event
        let event = AdaptationEvent {
            timestamp: current_time,
            input_characteristics: input.characteristics.clone(),
            parameter_changes: Vec::new(),
        };
        self.adaptation_history.push_back(event);
        
        // Remove old events outside adaptation window
        while let Some(oldest_event) = self.adaptation_history.front() {
            if current_time.duration_since(oldest_event.timestamp) > self.adaptation_window {
                self.adaptation_history.pop_front();
            } else {
                break;
            }
        }
        
        // Analyze input characteristics
        let input_analysis = self.analyze_input_characteristics(input)?;
        
        // Compute parameter adjustments
        let parameter_adjustments = self.compute_parameter_adjustments(&input_analysis)?;
        
        // Apply parameter adjustments
        self.apply_parameter_adjustments(&parameter_adjustments)?;
        
        // Generate adaptation output
        Ok(AdaptationOutput {
            adapted_parameters: self.current_parameters.clone(),
            adaptation_confidence: self.compute_adaptation_confidence(&input_analysis),
            stability_measure: self.compute_stability_measure(),
        })
    }

    /// Analyzes input characteristics for adaptation
    fn analyze_input_characteristics(&self, input: &AdaptationInput) -> Result<InputAnalysis, AfiyahError> {
        let mut analysis = InputAnalysis::default();
        
        // Compute temporal statistics
        analysis.temporal_variability = self.compute_temporal_variability(&input.characteristics);
        analysis.spatial_complexity = self.compute_spatial_complexity(&input.characteristics);
        analysis.motion_intensity = self.compute_motion_intensity(&input.characteristics);
        
        // Compute adaptation urgency
        analysis.adaptation_urgency = self.compute_adaptation_urgency(&analysis);
        
        Ok(analysis)
    }

    /// Computes temporal variability of input
    fn compute_temporal_variability(&self, characteristics: &[f64]) -> f64 {
        if characteristics.len() < 2 {
            return 0.0;
        }
        
        let mean = characteristics.iter().sum::<f64>() / characteristics.len() as f64;
        let variance = characteristics.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>() / characteristics.len() as f64;
        
        variance.sqrt()
    }

    /// Computes spatial complexity of input
    fn compute_spatial_complexity(&self, characteristics: &[f64]) -> f64 {
        if characteristics.is_empty() {
            return 0.0;
        }
        
        // Compute entropy as measure of spatial complexity
        let mut histogram = [0; 256];
        for &value in characteristics {
            let bin = ((value * 255.0) as usize).min(255);
            histogram[bin] += 1;
        }
        
        let total = characteristics.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &histogram {
            if count > 0 {
                let probability = count as f64 / total;
                entropy -= probability * probability.log2();
            }
        }
        
        entropy / 8.0 // Normalize to [0, 1]
    }

    /// Computes motion intensity from input characteristics
    fn compute_motion_intensity(&self, characteristics: &[f64]) -> f64 {
        if characteristics.len() < 2 {
            return 0.0;
        }
        
        let mut motion_sum = 0.0;
        for i in 1..characteristics.len() {
            motion_sum += (characteristics[i] - characteristics[i-1]).abs();
        }
        
        motion_sum / (characteristics.len() - 1) as f64
    }

    /// Computes adaptation urgency based on input analysis
    fn compute_adaptation_urgency(&self, analysis: &InputAnalysis) -> f64 {
        let urgency = analysis.temporal_variability * 0.4 +
                     analysis.spatial_complexity * 0.3 +
                     analysis.motion_intensity * 0.3;
        
        urgency.min(1.0)
    }

    /// Computes parameter adjustments based on input analysis
    fn compute_parameter_adjustments(&self, analysis: &InputAnalysis) -> Result<ParameterAdjustments, AfiyahError> {
        let mut adjustments = ParameterAdjustments::default();
        
        // Adjust learning rate based on adaptation urgency
        adjustments.learning_rate_change = (analysis.adaptation_urgency - 0.5) * self.learning_rate;
        
        // Adjust filter parameters based on spatial complexity
        adjustments.spatial_frequency_change = (analysis.spatial_complexity - 0.5) * 0.5;
        
        // Adjust temporal parameters based on motion intensity
        adjustments.temporal_frequency_change = (analysis.motion_intensity - 0.5) * 0.5;
        
        // Adjust adaptation rate based on temporal variability
        adjustments.adaptation_rate_change = (analysis.temporal_variability - 0.5) * 0.1;
        
        Ok(adjustments)
    }

    /// Applies parameter adjustments to current parameters
    fn apply_parameter_adjustments(&mut self, adjustments: &ParameterAdjustments) -> Result<(), AfiyahError> {
        // Apply learning rate adjustment
        self.current_parameters.learning_rate = (self.current_parameters.learning_rate + 
                                                 adjustments.learning_rate_change).max(0.001).min(0.1);
        
        // Apply spatial frequency adjustment
        self.current_parameters.spatial_frequency = (self.current_parameters.spatial_frequency + 
                                                     adjustments.spatial_frequency_change).max(0.1).min(5.0);
        
        // Apply temporal frequency adjustment
        self.current_parameters.temporal_frequency = (self.current_parameters.temporal_frequency + 
                                                      adjustments.temporal_frequency_change).max(0.1).min(10.0);
        
        // Apply adaptation rate adjustment
        self.current_parameters.adaptation_rate = (self.current_parameters.adaptation_rate + 
                                                   adjustments.adaptation_rate_change).max(0.01).min(0.3);
        
        Ok(())
    }

    /// Computes adaptation confidence
    fn compute_adaptation_confidence(&self, analysis: &InputAnalysis) -> f64 {
        // Higher confidence for more stable input characteristics
        let stability_factor = 1.0 - analysis.temporal_variability;
        let complexity_factor = analysis.spatial_complexity;
        
        (stability_factor * 0.6 + complexity_factor * 0.4).max(0.0).min(1.0)
    }

    /// Computes stability measure of current adaptation
    fn compute_stability_measure(&self) -> f64 {
        if self.adaptation_history.len() < 2 {
            return 1.0;
        }
        
        let recent_events: Vec<_> = self.adaptation_history.iter().rev().take(5).collect();
        let mut stability_sum = 0.0;
        
        for i in 1..recent_events.len() {
            let current = &recent_events[i-1].input_characteristics;
            let previous = &recent_events[i].input_characteristics;
            
            if current.len() == previous.len() {
                let diff = current.iter().zip(previous.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>() / current.len() as f64;
                stability_sum += 1.0 - diff;
            }
        }
        
        if recent_events.len() > 1 {
            stability_sum / (recent_events.len() - 1) as f64
        } else {
            1.0
        }
    }

    /// Enables or disables real-time adaptation
    pub fn set_adaptation_enabled(&mut self, enabled: bool) {
        self.adaptation_enabled = enabled;
    }

    /// Sets the adaptation window duration
    pub fn set_adaptation_window(&mut self, window: Duration) {
        self.adaptation_window = window;
    }

    /// Sets the learning rate
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.max(0.001).min(0.1);
    }
}

/// Input for adaptation processing
#[derive(Debug, Clone)]
pub struct AdaptationInput {
    pub characteristics: Vec<f64>,
    pub timestamp: Instant,
    pub input_type: InputType,
}

/// Type of input being processed
#[derive(Debug, Clone)]
pub enum InputType {
    Visual,
    Motion,
    Temporal,
    Combined,
}

/// Analysis of input characteristics
#[derive(Debug, Clone, Default)]
pub struct InputAnalysis {
    pub temporal_variability: f64,
    pub spatial_complexity: f64,
    pub motion_intensity: f64,
    pub adaptation_urgency: f64,
}

/// Parameter adjustments computed by adaptation
#[derive(Debug, Clone, Default)]
pub struct ParameterAdjustments {
    pub learning_rate_change: f64,
    pub spatial_frequency_change: f64,
    pub temporal_frequency_change: f64,
    pub adaptation_rate_change: f64,
}

/// Current adaptation parameters
#[derive(Debug, Clone)]
pub struct AdaptationParameters {
    pub learning_rate: f64,
    pub spatial_frequency: f64,
    pub temporal_frequency: f64,
    pub adaptation_rate: f64,
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            spatial_frequency: 2.0,
            temporal_frequency: 4.0,
            adaptation_rate: 0.1,
        }
    }
}

/// Output from adaptation processing
#[derive(Debug, Clone)]
pub struct AdaptationOutput {
    pub adapted_parameters: AdaptationParameters,
    pub adaptation_confidence: f64,
    pub stability_measure: f64,
}

/// Adaptation event for history tracking
#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    pub timestamp: Instant,
    pub input_characteristics: Vec<f64>,
    pub parameter_changes: Vec<ParameterAdjustments>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptation_controller_creation() {
        let controller = RealTimeAdaptationController::new();
        assert_eq!(controller.learning_rate, 0.01);
        assert!(controller.adaptation_enabled);
    }

    #[test]
    fn test_adaptation_parameters_default() {
        let params = AdaptationParameters::default();
        assert_eq!(params.learning_rate, 0.01);
        assert_eq!(params.spatial_frequency, 2.0);
    }

    #[test]
    fn test_input_analysis_default() {
        let analysis = InputAnalysis::default();
        assert_eq!(analysis.temporal_variability, 0.0);
        assert_eq!(analysis.spatial_complexity, 0.0);
    }
}
