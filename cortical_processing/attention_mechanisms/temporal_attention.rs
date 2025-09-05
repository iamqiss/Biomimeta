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

//! Temporal Attention Mechanisms
//! 
//! Implements sophisticated temporal attention processing based on
//! neurophysiological studies of temporal dynamics in visual attention.
//! 
//! Biological Basis:
//! - Posner & Petersen (1990): Attention networks and temporal dynamics
//! - Corbetta & Shulman (2002): Temporal attention networks
//! - Nobre & Kastner (2014): Temporal attention mechanisms
//! - Theeuwes (2010): Temporal aspects of attention

use ndarray::{Array2, Array3, Array4};
use std::collections::VecDeque;
use crate::AfiyahError;

/// Temporal attention configuration
#[derive(Debug, Clone)]
pub struct TemporalAttentionConfig {
    pub temporal_window: usize,
    pub attention_decay: f64,
    pub prediction_horizon: usize,
    pub motion_sensitivity: f64,
    pub temporal_integration: f64,
    pub adaptation_rate: f64,
    pub habituation_threshold: f64,
    pub novelty_weight: f64,
}

impl Default for TemporalAttentionConfig {
    fn default() -> Self {
        Self {
            temporal_window: 10, // 10 frames of temporal context
            attention_decay: 0.95, // Exponential decay of attention
            prediction_horizon: 5, // Predict 5 frames ahead
            motion_sensitivity: 0.8, // Sensitivity to motion changes
            temporal_integration: 0.7, // Integration across time
            adaptation_rate: 0.1, // Rate of temporal adaptation
            habituation_threshold: 0.3, // Threshold for habituation
            novelty_weight: 0.6, // Weight for novelty detection
        }
    }
}

/// Temporal attention state
#[derive(Debug, Clone)]
pub struct TemporalAttentionState {
    pub attention_history: VecDeque<Array2<f64>>,
    pub motion_vectors: VecDeque<Array2<(f64, f64)>>,
    pub novelty_scores: VecDeque<f64>,
    pub adaptation_weights: Array2<f64>,
    pub prediction_errors: VecDeque<f64>,
    pub temporal_consistency: f64,
}

/// Temporal attention output
#[derive(Debug, Clone)]
pub struct TemporalAttentionOutput {
    pub current_attention: Array2<f64>,
    pub predicted_attention: Array2<f64>,
    pub motion_prediction: Array2<(f64, f64)>,
    pub novelty_map: Array2<f64>,
    pub temporal_consistency: f64,
    pub adaptation_level: f64,
    pub prediction_confidence: f64,
}

/// Temporal attention processor
pub struct TemporalAttentionProcessor {
    config: TemporalAttentionConfig,
    state: TemporalAttentionState,
    motion_detector: MotionDetector,
    novelty_detector: NoveltyDetector,
    temporal_predictor: TemporalPredictor,
}

/// Motion detection for temporal attention
struct MotionDetector {
    optical_flow_threshold: f64,
    motion_magnitude_threshold: f64,
    temporal_smoothing: f64,
}

impl MotionDetector {
    fn new() -> Self {
        Self {
            optical_flow_threshold: 0.1,
            motion_magnitude_threshold: 0.05,
            temporal_smoothing: 0.8,
        }
    }

    fn detect_motion(&self, current: &Array2<f64>, previous: &Array2<f64>) -> Result<Array2<(f64, f64)>, AfiyahError> {
        let (height, width) = current.dim();
        let mut motion_vectors = Array2::from_elem((height, width), (0.0, 0.0));
        
        // Simple optical flow using gradient-based method
        for i in 1..height-1 {
            for j in 1..width-1 {
                // Calculate temporal gradient
                let dt = current[[i, j]] - previous[[i, j]];
                
                // Calculate spatial gradients
                let dx = (current[[i, j+1]] - current[[i, j-1]]) / 2.0;
                let dy = (current[[i+1, j]] - current[[i-1, j]]) / 2.0;
                
                // Calculate motion vector (Lucas-Kanade approximation)
                let magnitude_sq = dx * dx + dy * dy;
                if magnitude_sq > self.motion_magnitude_threshold {
                    let vx = -dt * dx / (magnitude_sq + 1e-6);
                    let vy = -dt * dy / (magnitude_sq + 1e-6);
                    
                    // Apply threshold
                    if vx.abs() > self.optical_flow_threshold || vy.abs() > self.optical_flow_threshold {
                        motion_vectors[[i, j]] = (vx, vy);
                    }
                }
            }
        }
        
        Ok(motion_vectors)
    }
}

/// Novelty detection for temporal attention
struct NoveltyDetector {
    novelty_threshold: f64,
    adaptation_rate: f64,
    memory_decay: f64,
}

impl NoveltyDetector {
    fn new() -> Self {
        Self {
            novelty_threshold: 0.2,
            adaptation_rate: 0.05,
            memory_decay: 0.95,
        }
    }

    fn detect_novelty(&self, current: &Array2<f64>, memory: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut novelty_sum = 0.0;
        let mut count = 0;
        
        for i in 0..current.len() {
            let current_val = current.as_slice().unwrap()[i];
            let memory_val = memory.as_slice().unwrap()[i];
            let novelty = (current_val - memory_val).abs();
            novelty_sum += novelty;
            count += 1;
        }
        
        let avg_novelty = if count > 0 { novelty_sum / count as f64 } else { 0.0 };
        Ok(avg_novelty)
    }
}

/// Temporal prediction for attention
struct TemporalPredictor {
    prediction_horizon: usize,
    temporal_smoothing: f64,
    motion_extrapolation: f64,
}

impl TemporalPredictor {
    fn new(prediction_horizon: usize) -> Self {
        Self {
            prediction_horizon,
            temporal_smoothing: 0.7,
            motion_extrapolation: 0.8,
        }
    }

    fn predict_attention(&self, 
                        current_attention: &Array2<f64>, 
                        motion_vectors: &Array2<(f64, f64)>,
                        attention_history: &VecDeque<Array2<f64>>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = current_attention.dim();
        let mut predicted = Array2::zeros((height, width));
        
        // Predict based on motion vectors
        for i in 0..height {
            for j in 0..width {
                let (vx, vy) = motion_vectors[[i, j]];
                
                // Extrapolate attention based on motion
                let future_i = (i as f64 + vy * self.prediction_horizon as f64) as usize;
                let future_j = (j as f64 + vx * self.prediction_horizon as f64) as usize;
                
                if future_i < height && future_j < width {
                    predicted[[future_i, future_j]] += current_attention[[i, j]] * self.motion_extrapolation;
                }
            }
        }
        
        // Apply temporal smoothing based on history
        if !attention_history.is_empty() {
            let recent_attention = attention_history.back().unwrap();
            for i in 0..height {
                for j in 0..width {
                    predicted[[i, j]] = predicted[[i, j]] * (1.0 - self.temporal_smoothing) + 
                                      recent_attention[[i, j]] * self.temporal_smoothing;
                }
            }
        }
        
        Ok(predicted)
    }
}

impl TemporalAttentionProcessor {
    /// Creates a new temporal attention processor
    pub fn new() -> Result<Self, AfiyahError> {
        let config = TemporalAttentionConfig::default();
        Self::with_config(config)
    }

    /// Creates a new temporal attention processor with custom configuration
    pub fn with_config(config: TemporalAttentionConfig) -> Result<Self, AfiyahError> {
        let (height, width) = (64, 64); // Default size, will be updated on first frame
        
        let state = TemporalAttentionState {
            attention_history: VecDeque::with_capacity(config.temporal_window),
            motion_vectors: VecDeque::with_capacity(config.temporal_window),
            novelty_scores: VecDeque::with_capacity(config.temporal_window),
            adaptation_weights: Array2::ones((height, width)),
            prediction_errors: VecDeque::with_capacity(config.prediction_horizon),
            temporal_consistency: 1.0,
        };

        let motion_detector = MotionDetector::new();
        let novelty_detector = NoveltyDetector::new();
        let temporal_predictor = TemporalPredictor::new(config.prediction_horizon);

        Ok(Self {
            config,
            state,
            motion_detector,
            novelty_detector,
            temporal_predictor,
        })
    }

    /// Processes temporal attention for a new frame
    pub fn process_temporal_attention(&mut self, current_frame: &Array2<f64>) -> Result<TemporalAttentionOutput, AfiyahError> {
        // Initialize state if this is the first frame
        if self.state.attention_history.is_empty() {
            self.initialize_state(current_frame)?;
        }

        // Detect motion if we have previous frame
        let motion_vectors = if let Some(previous_frame) = self.state.attention_history.back() {
            self.motion_detector.detect_motion(current_frame, previous_frame)?
        } else {
            Array2::from_elem(current_frame.dim(), (0.0, 0.0))
        };

        // Detect novelty
        let memory = self.compute_temporal_memory()?;
        let novelty_score = self.novelty_detector.detect_novelty(current_frame, &memory)?;

        // Update adaptation weights
        self.update_adaptation_weights(current_frame, &novelty_score)?;

        // Compute current attention
        let current_attention = self.compute_current_attention(current_frame, &motion_vectors, &novelty_score)?;

        // Predict future attention
        let predicted_attention = self.temporal_predictor.predict_attention(
            &current_attention, 
            &motion_vectors, 
            &self.state.attention_history
        )?;

        // Update state
        self.update_state(&current_attention, &motion_vectors, novelty_score)?;

        // Calculate temporal consistency
        let temporal_consistency = self.calculate_temporal_consistency()?;

        // Calculate prediction confidence
        let prediction_confidence = self.calculate_prediction_confidence()?;

        // Calculate adaptation level
        let adaptation_level = self.calculate_adaptation_level()?;

        // Create novelty map
        let novelty_map = self.create_novelty_map(current_frame, &memory)?;

        Ok(TemporalAttentionOutput {
            current_attention,
            predicted_attention,
            motion_prediction: motion_vectors,
            novelty_map,
            temporal_consistency,
            adaptation_level,
            prediction_confidence,
        })
    }

    /// Updates the temporal attention configuration
    pub fn update_config(&mut self, config: TemporalAttentionConfig) -> Result<(), AfiyahError> {
        self.config = config.clone();
        
        // Update temporal predictor
        self.temporal_predictor = TemporalPredictor::new(config.prediction_horizon);
        
        // Resize state if needed
        self.resize_state()?;
        
        Ok(())
    }

    /// Gets current configuration
    pub fn get_config(&self) -> &TemporalAttentionConfig {
        &self.config
    }

    /// Gets current state
    pub fn get_state(&self) -> &TemporalAttentionState {
        &self.state
    }

    fn initialize_state(&mut self, frame: &Array2<f64>) -> Result<(), AfiyahError> {
        let (height, width) = frame.dim();
        self.state.adaptation_weights = Array2::ones((height, width));
        self.state.temporal_consistency = 1.0;
        Ok(())
    }

    fn compute_temporal_memory(&self) -> Result<Array2<f64>, AfiyahError> {
        if self.state.attention_history.is_empty() {
            return Err(AfiyahError::InvalidState("No attention history available".to_string()));
        }

        let (height, width) = self.state.attention_history[0].dim();
        let mut memory = Array2::zeros((height, width));
        let mut weight_sum = 0.0;

        for (i, attention_map) in self.state.attention_history.iter().enumerate() {
            let weight = self.config.attention_decay.powi(i as i32);
            for j in 0..height {
                for k in 0..width {
                    memory[[j, k]] += attention_map[[j, k]] * weight;
                }
            }
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            for j in 0..height {
                for k in 0..width {
                    memory[[j, k]] /= weight_sum;
                }
            }
        }

        Ok(memory)
    }

    fn update_adaptation_weights(&mut self, current_frame: &Array2<f64>, novelty_score: &f64) -> Result<(), AfiyahError> {
        let adaptation_rate = self.config.adaptation_rate * (1.0 + *novelty_score);
        
        for i in 0..self.state.adaptation_weights.len() {
            let current_weight = self.state.adaptation_weights.as_slice().unwrap()[i];
            let frame_val = current_frame.as_slice().unwrap()[i];
            
            // Update weight based on novelty and current frame
            let new_weight = current_weight + adaptation_rate * (frame_val - current_weight);
            self.state.adaptation_weights.as_slice_mut().unwrap()[i] = new_weight.max(0.0).min(1.0);
        }
        
        Ok(())
    }

    fn compute_current_attention(&self, current_frame: &Array2<f64>, motion_vectors: &Array2<(f64, f64)>, novelty_score: &f64) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = current_frame.dim();
        let mut attention = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let frame_val = current_frame[[i, j]];
                let (vx, vy) = motion_vectors[[i, j]];
                let adaptation_weight = self.state.adaptation_weights[[i, j]];
                
                // Calculate attention based on frame value, motion, and adaptation
                let motion_magnitude = (vx * vx + vy * vy).sqrt();
                let motion_attention = motion_magnitude * self.config.motion_sensitivity;
                let novelty_attention = *novelty_score * self.config.novelty_weight;
                
                attention[[i, j]] = (frame_val * adaptation_weight + motion_attention + novelty_attention).min(1.0);
            }
        }

        Ok(attention)
    }

    fn update_state(&mut self, current_attention: &Array2<f64>, motion_vectors: &Array2<(f64, f64)>, novelty_score: f64) -> Result<(), AfiyahError> {
        // Add current attention to history
        self.state.attention_history.push_back(current_attention.clone());
        if self.state.attention_history.len() > self.config.temporal_window {
            self.state.attention_history.pop_front();
        }

        // Add motion vectors to history
        self.state.motion_vectors.push_back(motion_vectors.clone());
        if self.state.motion_vectors.len() > self.config.temporal_window {
            self.state.motion_vectors.pop_front();
        }

        // Add novelty score to history
        self.state.novelty_scores.push_back(novelty_score);
        if self.state.novelty_scores.len() > self.config.temporal_window {
            self.state.novelty_scores.pop_front();
        }

        Ok(())
    }

    fn calculate_temporal_consistency(&self) -> Result<f64, AfiyahError> {
        if self.state.attention_history.len() < 2 {
            return Ok(1.0);
        }

        let mut consistency_sum = 0.0;
        let mut count = 0;

        for i in 1..self.state.attention_history.len() {
            let current = &self.state.attention_history[i];
            let previous = &self.state.attention_history[i-1];
            
            let mut correlation_sum = 0.0;
            let mut norm_sum = 0.0;
            
            for j in 0..current.len() {
                let curr_val = current.as_slice().unwrap()[j];
                let prev_val = previous.as_slice().unwrap()[j];
                correlation_sum += curr_val * prev_val;
                norm_sum += curr_val * curr_val + prev_val * prev_val;
            }
            
            if norm_sum > 0.0 {
                let correlation = 2.0 * correlation_sum / norm_sum;
                consistency_sum += correlation;
                count += 1;
            }
        }

        Ok(if count > 0 { consistency_sum / count as f64 } else { 1.0 })
    }

    fn calculate_prediction_confidence(&self) -> Result<f64, AfiyahError> {
        if self.state.prediction_errors.is_empty() {
            return Ok(1.0);
        }

        let avg_error = self.state.prediction_errors.iter().sum::<f64>() / self.state.prediction_errors.len() as f64;
        let confidence = (-avg_error).exp().min(1.0);
        Ok(confidence)
    }

    fn calculate_adaptation_level(&self) -> Result<f64, AfiyahError> {
        let avg_weight = self.state.adaptation_weights.iter().sum::<f64>() / self.state.adaptation_weights.len() as f64;
        Ok(avg_weight)
    }

    fn create_novelty_map(&self, current_frame: &Array2<f64>, memory: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = current_frame.dim();
        let mut novelty_map = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let current_val = current_frame[[i, j]];
                let memory_val = memory[[i, j]];
                let novelty = (current_val - memory_val).abs();
                novelty_map[[i, j]] = novelty;
            }
        }

        Ok(novelty_map)
    }

    fn resize_state(&mut self) -> Result<(), AfiyahError> {
        // This would resize the state if the frame dimensions change
        // For now, we'll keep it simple and just clear the state
        self.state.attention_history.clear();
        self.state.motion_vectors.clear();
        self.state.novelty_scores.clear();
        self.state.prediction_errors.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_attention_config_default() {
        let config = TemporalAttentionConfig::default();
        assert_eq!(config.temporal_window, 10);
        assert_eq!(config.prediction_horizon, 5);
    }

    #[test]
    fn test_temporal_attention_processor_creation() {
        let processor = TemporalAttentionProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_temporal_attention_processing() {
        let mut processor = TemporalAttentionProcessor::new().unwrap();
        let frame = Array2::ones((32, 32));
        
        let result = processor.process_temporal_attention(&frame);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(output.temporal_consistency >= 0.0 && output.temporal_consistency <= 1.0);
    }

    #[test]
    fn test_motion_detection() {
        let detector = MotionDetector::new();
        let current = Array2::ones((16, 16));
        let previous = Array2::zeros((16, 16));
        
        let result = detector.detect_motion(&current, &previous);
        assert!(result.is_ok());
    }
}