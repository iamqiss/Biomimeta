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

//! Hebbian Learning Implementation
//! 
//! This module implements Hebbian learning mechanisms based on the principle
//! "neurons that fire together, wire together." It strengthens connections
//! between co-active neurons and weakens connections between non-co-active neurons.

use crate::AfiyahError;
use crate::cortical_processing::CorticalOutput;
use super::{HebbianOutput, Connection, WeightUpdateRule};

/// Hebbian learning processor for synaptic weight adaptation
pub struct HebbianLearning {
    connections: Vec<Connection>,
    weight_update_rule: WeightUpdateRule,
    learning_state: HebbianLearningState,
    co_activity_threshold: f64,
    learning_rate: f64,
}

impl HebbianLearning {
    /// Creates a new Hebbian learning processor with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        let mut connections = Vec::new();
        
        // Initialize random connections (simplified)
        for i in 0..100 {
            for j in 0..100 {
                if i != j {
                    connections.push(Connection {
                        from_neuron: i,
                        to_neuron: j,
                        weight: (rand::random::<f64>() - 0.5) * 0.1, // Small random weights
                        strength: 0.0,
                        last_activity: 0.0,
                    });
                }
            }
        }
        
        Ok(Self {
            connections,
            weight_update_rule: WeightUpdateRule::new(),
            learning_state: HebbianLearningState::default(),
            co_activity_threshold: 0.3,
            learning_rate: 0.01,
        })
    }

    /// Adapts cortical output using Hebbian learning
    pub fn adapt(&mut self, cortical_output: &CorticalOutput) -> Result<HebbianOutput, AfiyahError> {
        // Extract neural activities from cortical output
        let activities = self.extract_activities(cortical_output)?;
        
        // Update connections based on co-activity
        let mut strengthened_connections = Vec::new();
        let mut weakened_connections = Vec::new();
        
        for connection in &mut self.connections {
            let pre_activity = activities.get(connection.from_neuron).copied().unwrap_or(0.0);
            let post_activity = activities.get(connection.to_neuron).copied().unwrap_or(0.0);
            
            // Calculate co-activity strength
            let co_activity = pre_activity * post_activity;
            
            // Update weight based on Hebbian rule
            let old_weight = connection.weight;
            connection.weight = self.weight_update_rule.update_weight(
                old_weight,
                pre_activity,
                post_activity,
            );
            
            // Update connection strength
            connection.strength = co_activity;
            connection.last_activity = co_activity;
            
            // Categorize connections
            if co_activity > self.co_activity_threshold {
                strengthened_connections.push(connection.clone());
            } else if co_activity < -self.co_activity_threshold {
                weakened_connections.push(connection.clone());
            }
        }
        
        // Update learning state
        self.update_learning_state(&activities)?;
        
        Ok(HebbianOutput {
            strengthened_connections,
            weakened_connections,
            learning_rate: self.learning_rate,
            co_activity_strength: self.calculate_co_activity_strength(&activities),
        })
    }

    /// Trains Hebbian learning on visual input data
    pub fn train(&mut self, training_data: &[crate::VisualInput]) -> Result<(), AfiyahError> {
        for input in training_data {
            // Extract activities from visual input
            let activities = self.extract_activities_from_input(input)?;
            
            // Apply Hebbian learning
            self.apply_hebbian_learning(&activities)?;
        }
        
        // Update learning parameters based on training
        self.update_learning_parameters()?;
        
        Ok(())
    }

    fn extract_activities(&self, cortical_output: &CorticalOutput) -> Result<Vec<f64>, AfiyahError> {
        // Combine all cortical features into activity vector
        let mut activities = Vec::new();
        
        // Add V1 features
        for orientation_map in &cortical_output.v1_features {
            activities.push(orientation_map.strength);
        }
        
        // Add V2 features
        for texture_map in &cortical_output.v2_features {
            activities.push(texture_map.strength);
        }
        
        // Add V3-V5 features
        for motion_vector in &cortical_output.v3_v5_features {
            activities.push(motion_vector.confidence);
        }
        
        // Add attention features
        for attention_map in &cortical_output.attention_maps {
            activities.push(attention_map.attention_strength);
        }
        
        // Add temporal features
        for prediction in &cortical_output.temporal_predictions {
            activities.push(prediction.prediction_confidence);
        }
        
        // Add feedback features
        activities.extend(&cortical_output.feedback_modulation);
        
        Ok(activities)
    }

    fn extract_activities_from_input(&self, input: &crate::VisualInput) -> Result<Vec<f64>, AfiyahError> {
        // Extract activities from visual input
        let mut activities = Vec::new();
        
        // Add luminance activities
        activities.extend(&input.luminance_data);
        
        // Add chromatic activities
        for chromatic in &input.chromatic_data {
            activities.push(chromatic.red);
            activities.push(chromatic.green);
            activities.push(chromatic.blue);
        }
        
        // Add metadata activities
        activities.push(input.metadata.quality_hint);
        if let Some(attention_region) = &input.metadata.attention_region {
            activities.push(attention_region.priority);
        }
        
        Ok(activities)
    }

    fn apply_hebbian_learning(&mut self, activities: &[f64]) -> Result<(), AfiyahError> {
        // Apply Hebbian learning rule to all connections
        for connection in &mut self.connections {
            let pre_activity = activities.get(connection.from_neuron).copied().unwrap_or(0.0);
            let post_activity = activities.get(connection.to_neuron).copied().unwrap_or(0.0);
            
            // Calculate weight change based on Hebbian rule
            let weight_change = self.learning_rate * pre_activity * post_activity;
            
            // Update weight
            connection.weight = (connection.weight + weight_change)
                .max(self.weight_update_rule.min_weight)
                .min(self.weight_update_rule.max_weight);
            
            // Update strength
            connection.strength = pre_activity * post_activity;
            connection.last_activity = connection.strength;
        }
        
        Ok(())
    }

    fn update_learning_state(&mut self, activities: &[f64]) -> Result<(), AfiyahError> {
        // Update learning state based on activities
        let avg_activity = activities.iter().sum::<f64>() / activities.len() as f64;
        let max_activity = activities.iter().fold(0.0, |a, &b| a.max(b));
        
        self.learning_state.avg_activity = avg_activity;
        self.learning_state.max_activity = max_activity;
        self.learning_state.learning_rate = self.calculate_adaptive_learning_rate(avg_activity);
        
        Ok(())
    }

    fn calculate_adaptive_learning_rate(&self, avg_activity: f64) -> f64 {
        // Adaptive learning rate based on activity level
        if avg_activity > 0.5 {
            self.learning_rate * 1.1 // Increase learning rate for high activity
        } else if avg_activity < 0.1 {
            self.learning_rate * 0.9 // Decrease learning rate for low activity
        } else {
            self.learning_rate
        }
    }

    fn calculate_co_activity_strength(&self, activities: &[f64]) -> f64 {
        // Calculate overall co-activity strength
        let mut co_activity_sum = 0.0;
        let mut connection_count = 0;
        
        for connection in &self.connections {
            let pre_activity = activities.get(connection.from_neuron).copied().unwrap_or(0.0);
            let post_activity = activities.get(connection.to_neuron).copied().unwrap_or(0.0);
            
            co_activity_sum += pre_activity * post_activity;
            connection_count += 1;
        }
        
        if connection_count > 0 {
            co_activity_sum / connection_count as f64
        } else {
            0.0
        }
    }

    fn update_learning_parameters(&mut self) -> Result<(), AfiyahError> {
        // Update learning parameters based on training progress
        self.learning_rate = self.learning_state.learning_rate;
        
        // Adjust co-activity threshold based on learning state
        if self.learning_state.avg_activity > 0.5 {
            self.co_activity_threshold *= 1.05; // Increase threshold for high activity
        } else {
            self.co_activity_threshold *= 0.95; // Decrease threshold for low activity
        }
        
        // Ensure threshold stays within reasonable bounds
        self.co_activity_threshold = self.co_activity_threshold.max(0.1).min(0.9);
        
        Ok(())
    }
}

/// Hebbian learning state
#[derive(Debug, Clone)]
pub struct HebbianLearningState {
    pub avg_activity: f64,
    pub max_activity: f64,
    pub learning_rate: f64,
    pub co_activity_strength: f64,
    pub learning_progress: f64,
}

impl Default for HebbianLearningState {
    fn default() -> Self {
        Self {
            avg_activity: 0.0,
            max_activity: 0.0,
            learning_rate: 0.01,
            co_activity_strength: 0.0,
            learning_progress: 0.0,
        }
    }
}

/// Hebbian learning rule variants
#[derive(Debug, Clone)]
pub enum HebbianRule {
    /// Standard Hebbian rule: Δw = η * pre * post
    Standard,
    /// Oja's rule: Δw = η * pre * (post - w * pre)
    Oja,
    /// Bienenstock-Cooper-Munro (BCM) rule: Δw = η * pre * post * (post - θ)
    BCM,
    /// Spike-timing dependent plasticity (STDP)
    STDP,
}

impl HebbianRule {
    /// Applies the specified Hebbian rule
    pub fn apply(&self, pre_activity: f64, post_activity: f64, current_weight: f64, learning_rate: f64) -> f64 {
        match self {
            HebbianRule::Standard => learning_rate * pre_activity * post_activity,
            HebbianRule::Oja => learning_rate * pre_activity * (post_activity - current_weight * pre_activity),
            HebbianRule::BCM => {
                let theta = 0.5; // Threshold
                learning_rate * pre_activity * post_activity * (post_activity - theta)
            }
            HebbianRule::STDP => {
                // Simplified STDP implementation
                let time_diff = post_activity - pre_activity;
                if time_diff > 0.0 {
                    learning_rate * pre_activity * post_activity * (-time_diff).exp()
                } else {
                    -learning_rate * pre_activity * post_activity * time_diff.exp()
                }
            }
        }
    }
}

/// Co-activity analyzer
pub struct CoActivityAnalyzer {
    pub analysis_window: usize,
    pub correlation_threshold: f64,
}

impl CoActivityAnalyzer {
    /// Creates a new co-activity analyzer
    pub fn new() -> Self {
        Self {
            analysis_window: 50,
            correlation_threshold: 0.3,
        }
    }

    /// Analyzes co-activity between neurons
    pub fn analyze_co_activity(&self, activities: &[f64], neuron_pairs: &[(usize, usize)]) -> Result<Vec<f64>, AfiyahError> {
        let mut co_activities = Vec::new();
        
        for &(neuron1, neuron2) in neuron_pairs {
            let activity1 = activities.get(neuron1).copied().unwrap_or(0.0);
            let activity2 = activities.get(neuron2).copied().unwrap_or(0.0);
            
            let co_activity = activity1 * activity2;
            co_activities.push(co_activity);
        }
        
        Ok(co_activities)
    }

    /// Calculates correlation between neuron activities
    pub fn calculate_correlation(&self, activities1: &[f64], activities2: &[f64]) -> Result<f64, AfiyahError> {
        if activities1.len() != activities2.len() || activities1.is_empty() {
            return Ok(0.0);
        }
        
        let mean1 = activities1.iter().sum::<f64>() / activities1.len() as f64;
        let mean2 = activities2.iter().sum::<f64>() / activities2.len() as f64;
        
        let numerator: f64 = activities1.iter().zip(activities2.iter())
            .map(|(a1, a2)| (a1 - mean1) * (a2 - mean2))
            .sum();
        
        let denominator1: f64 = activities1.iter()
            .map(|a| (a - mean1).powi(2))
            .sum::<f64>().sqrt();
        
        let denominator2: f64 = activities2.iter()
            .map(|a| (a - mean2).powi(2))
            .sum::<f64>().sqrt();
        
        if denominator1 > 0.0 && denominator2 > 0.0 {
            Ok(numerator / (denominator1 * denominator2))
        } else {
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hebbian_learning_creation() {
        let learning = HebbianLearning::new();
        assert!(learning.is_ok());
    }

    #[test]
    fn test_hebbian_learning_state_defaults() {
        let state = HebbianLearningState::default();
        assert_eq!(state.avg_activity, 0.0);
        assert_eq!(state.learning_rate, 0.01);
    }

    #[test]
    fn test_hebbian_rule_standard() {
        let rule = HebbianRule::Standard;
        let weight_change = rule.apply(0.8, 0.6, 0.5, 0.01);
        assert_eq!(weight_change, 0.01 * 0.8 * 0.6);
    }

    #[test]
    fn test_co_activity_analyzer() {
        let analyzer = CoActivityAnalyzer::new();
        let activities = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let neuron_pairs = vec![(0, 1), (2, 3)];
        let co_activities = analyzer.analyze_co_activity(&activities, &neuron_pairs);
        assert!(co_activities.is_ok());
    }
}