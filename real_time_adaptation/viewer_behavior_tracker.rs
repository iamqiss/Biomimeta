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

//! Viewer Behavior Tracker for Real-Time Adaptation
//! 
//! Tracks and analyzes viewer behavior patterns to inform adaptive compression decisions.
//! 
//! Biological Basis:
//! - Yarbus (1967): Eye movements and vision - Eye movement patterns during viewing
//! - Henderson (2003): Human gaze control during real-world scene perception
//! - Itti & Koch (2001): Computational modelling of visual attention
//! - Theeuwes (2010): Top-down and bottom-up control of visual selection

use ndarray::{Array1, Array2};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use crate::AfiyahError;

/// Viewer behavior data
#[derive(Debug, Clone)]
pub struct ViewerBehavior {
    pub gaze_position: (f64, f64),
    pub saccade_frequency: f64,
    pub fixation_duration: f64,
    pub attention_level: f64,
    pub pupil_dilation: f64,
    pub blink_rate: f64,
    pub head_movement: f64,
    pub engagement_score: f64,
    pub consistency_score: f64,
    pub cognitive_load: f64,
}

/// Behavior metrics for analysis
#[derive(Debug, Clone)]
pub struct BehaviorMetrics {
    pub average_saccade_frequency: f64,
    pub average_fixation_duration: f64,
    pub attention_variance: f64,
    pub engagement_trend: f64,
    pub cognitive_load_trend: f64,
    pub behavior_stability: f64,
    pub adaptation_speed: f64,
    pub preference_consistency: f64,
}

/// Behavior tracker configuration
#[derive(Debug, Clone)]
pub struct BehaviorTrackerConfig {
    pub tracking_window: Duration,
    pub saccade_threshold: f64,
    pub fixation_threshold: f64,
    pub attention_decay: f64,
    pub engagement_weight: f64,
    pub consistency_weight: f64,
    pub cognitive_load_weight: f64,
    pub adaptation_rate: f64,
}

impl Default for BehaviorTrackerConfig {
    fn default() -> Self {
        Self {
            tracking_window: Duration::from_secs(10),
            saccade_threshold: 0.1,
            fixation_threshold: 0.05,
            attention_decay: 0.95,
            engagement_weight: 0.4,
            consistency_weight: 0.3,
            cognitive_load_weight: 0.3,
            adaptation_rate: 0.1,
        }
    }
}

/// Viewer behavior tracker
pub struct ViewerBehaviorTracker {
    config: BehaviorTrackerConfig,
    behavior_history: VecDeque<ViewerBehavior>,
    metrics_history: VecDeque<BehaviorMetrics>,
    current_behavior: ViewerBehavior,
    last_update: Instant,
    fixation_start: Option<Instant>,
    saccade_count: u32,
    blink_count: u32,
    attention_accumulator: f64,
    engagement_accumulator: f64,
    cognitive_load_accumulator: f64,
}

impl ViewerBehaviorTracker {
    /// Creates a new viewer behavior tracker
    pub fn new() -> Result<Self, AfiyahError> {
        let config = BehaviorTrackerConfig::default();
        Self::with_config(config)
    }

    /// Creates a new viewer behavior tracker with custom configuration
    pub fn with_config(config: BehaviorTrackerConfig) -> Result<Self, AfiyahError> {
        let current_behavior = ViewerBehavior {
            gaze_position: (0.5, 0.5), // Center of screen
            saccade_frequency: 0.0,
            fixation_duration: 0.0,
            attention_level: 0.5,
            pupil_dilation: 0.5,
            blink_rate: 0.0,
            head_movement: 0.0,
            engagement_score: 0.5,
            consistency_score: 1.0,
            cognitive_load: 0.5,
        };

        Ok(Self {
            config: config.clone(),
            behavior_history: VecDeque::with_capacity(1000),
            metrics_history: VecDeque::with_capacity(100),
            current_behavior,
            last_update: Instant::now(),
            fixation_start: None,
            saccade_count: 0,
            blink_count: 0,
            attention_accumulator: 0.5,
            engagement_accumulator: 0.5,
            cognitive_load_accumulator: 0.5,
        })
    }

    /// Updates viewer behavior with new data
    pub fn update_behavior(&mut self, behavior: &ViewerBehavior) -> Result<(), AfiyahError> {
        let now = Instant::now();
        let time_delta = now.duration_since(self.last_update).as_secs_f64();

        // Update saccade detection
        self.update_saccade_detection(behavior, time_delta)?;

        // Update fixation tracking
        self.update_fixation_tracking(behavior, time_delta)?;

        // Update attention level
        self.update_attention_level(behavior, time_delta)?;

        // Update engagement score
        self.update_engagement_score(behavior, time_delta)?;

        // Update cognitive load
        self.update_cognitive_load(behavior, time_delta)?;

        // Update consistency score
        self.update_consistency_score(behavior)?;

        // Update current behavior
        self.current_behavior = behavior.clone();

        // Add to history
        self.behavior_history.push_back(behavior.clone());
        if self.behavior_history.len() > 1000 {
            self.behavior_history.pop_front();
        }

        // Update metrics
        self.update_metrics()?;

        self.last_update = now;
        Ok(())
    }

    /// Gets current viewer behavior
    pub fn get_current_behavior(&self) -> Result<ViewerBehavior, AfiyahError> {
        Ok(self.current_behavior.clone())
    }

    /// Gets behavior metrics
    pub fn get_behavior_metrics(&self) -> Result<BehaviorMetrics, AfiyahError> {
        if self.metrics_history.is_empty() {
            return Err(AfiyahError::InvalidState("No behavior metrics available".to_string()));
        }
        Ok(self.metrics_history.back().unwrap().clone())
    }

    /// Predicts future behavior
    pub fn predict_behavior(&self, prediction_horizon: Duration) -> Result<ViewerBehavior, AfiyahError> {
        if self.behavior_history.len() < 2 {
            return Ok(self.current_behavior.clone());
        }

        let recent_behaviors: Vec<&ViewerBehavior> = self.behavior_history.iter().rev().take(10).collect();
        let prediction_time = prediction_horizon.as_secs_f64();

        // Predict gaze position using linear extrapolation
        let gaze_prediction = self.predict_gaze_position(&recent_behaviors, prediction_time)?;

        // Predict saccade frequency
        let saccade_prediction = self.predict_saccade_frequency(&recent_behaviors, prediction_time)?;

        // Predict attention level
        let attention_prediction = self.predict_attention_level(&recent_behaviors, prediction_time)?;

        // Predict engagement score
        let engagement_prediction = self.predict_engagement_score(&recent_behaviors, prediction_time)?;

        Ok(ViewerBehavior {
            gaze_position: gaze_prediction,
            saccade_frequency: saccade_prediction,
            fixation_duration: self.current_behavior.fixation_duration,
            attention_level: attention_prediction,
            pupil_dilation: self.current_behavior.pupil_dilation,
            blink_rate: self.current_behavior.blink_rate,
            head_movement: self.current_behavior.head_movement,
            engagement_score: engagement_prediction,
            consistency_score: self.current_behavior.consistency_score,
            cognitive_load: self.current_behavior.cognitive_load,
        })
    }

    /// Updates the tracker configuration
    pub fn update_config(&mut self, tracking_window: Duration) -> Result<(), AfiyahError> {
        self.config.tracking_window = tracking_window;
        Ok(())
    }

    /// Resets the tracker
    pub fn reset(&mut self) -> Result<(), AfiyahError> {
        self.behavior_history.clear();
        self.metrics_history.clear();
        self.saccade_count = 0;
        self.blink_count = 0;
        self.attention_accumulator = 0.5;
        self.engagement_accumulator = 0.5;
        self.cognitive_load_accumulator = 0.5;
        self.fixation_start = None;
        self.last_update = Instant::now();
        Ok(())
    }

    /// Gets current configuration
    pub fn get_config(&self) -> &BehaviorTrackerConfig {
        &self.config
    }

    fn update_saccade_detection(&mut self, behavior: &ViewerBehavior, time_delta: f64) -> Result<(), AfiyahError> {
        // Calculate gaze movement distance
        let gaze_movement = self.calculate_gaze_movement(&self.current_behavior.gaze_position, &behavior.gaze_position)?;

        // Detect saccade if movement exceeds threshold
        if gaze_movement > self.config.saccade_threshold {
            self.saccade_count += 1;
            self.fixation_start = Some(Instant::now());
        }

        // Update saccade frequency
        let saccade_frequency = if time_delta > 0.0 {
            self.saccade_count as f64 / time_delta
        } else {
            0.0
        };

        self.current_behavior.saccade_frequency = saccade_frequency;
        Ok(())
    }

    fn update_fixation_tracking(&mut self, behavior: &ViewerBehavior, time_delta: f64) -> Result<(), AfiyahError> {
        // Calculate gaze movement distance
        let gaze_movement = self.calculate_gaze_movement(&self.current_behavior.gaze_position, &behavior.gaze_position)?;

        // Update fixation duration
        if gaze_movement < self.config.fixation_threshold {
            if let Some(start) = self.fixation_start {
                let fixation_duration = Instant::now().duration_since(start).as_secs_f64();
                self.current_behavior.fixation_duration = fixation_duration;
            } else {
                self.fixation_start = Some(Instant::now());
            }
        } else {
            self.fixation_start = None;
            self.current_behavior.fixation_duration = 0.0;
        }

        Ok(())
    }

    fn update_attention_level(&mut self, behavior: &ViewerBehavior, time_delta: f64) -> Result<(), AfiyahError> {
        // Update attention accumulator with exponential decay
        let attention_input = behavior.attention_level;
        self.attention_accumulator = self.attention_accumulator * self.config.attention_decay + 
                                   attention_input * (1.0 - self.config.attention_decay);

        self.current_behavior.attention_level = self.attention_accumulator;
        Ok(())
    }

    fn update_engagement_score(&mut self, behavior: &ViewerBehavior, time_delta: f64) -> Result<(), AfiyahError> {
        // Calculate engagement based on multiple factors
        let gaze_stability = 1.0 - behavior.saccade_frequency;
        let attention_consistency = 1.0 - (behavior.attention_level - self.current_behavior.attention_level).abs();
        let fixation_quality = behavior.fixation_duration / 1.0; // Normalize to 1 second

        let engagement = (
            gaze_stability * 0.4 +
            attention_consistency * 0.3 +
            fixation_quality * 0.3
        ).min(1.0).max(0.0);

        // Update engagement accumulator
        self.engagement_accumulator = self.engagement_accumulator * self.config.attention_decay + 
                                    engagement * (1.0 - self.config.attention_decay);

        self.current_behavior.engagement_score = self.engagement_accumulator;
        Ok(())
    }

    fn update_cognitive_load(&mut self, behavior: &ViewerBehavior, time_delta: f64) -> Result<(), AfiyahError> {
        // Calculate cognitive load based on multiple indicators
        let saccade_load = behavior.saccade_frequency * 0.3;
        let attention_load = (1.0 - behavior.attention_level) * 0.4;
        let pupil_load = (behavior.pupil_dilation - 0.5).abs() * 0.3;

        let cognitive_load = (saccade_load + attention_load + pupil_load).min(1.0).max(0.0);

        // Update cognitive load accumulator
        self.cognitive_load_accumulator = self.cognitive_load_accumulator * self.config.attention_decay + 
                                        cognitive_load * (1.0 - self.config.attention_decay);

        self.current_behavior.cognitive_load = self.cognitive_load_accumulator;
        Ok(())
    }

    fn update_consistency_score(&mut self, behavior: &ViewerBehavior) -> Result<(), AfiyahError> {
        if self.behavior_history.len() < 2 {
            self.current_behavior.consistency_score = 1.0;
            return Ok(());
        }

        // Calculate consistency based on recent behavior patterns
        let recent_behaviors: Vec<&ViewerBehavior> = self.behavior_history.iter().rev().take(5).collect();
        
        let mut consistency_sum = 0.0;
        let mut count = 0;

        for i in 1..recent_behaviors.len() {
            let current = &recent_behaviors[i-1];
            let previous = &recent_behaviors[i];

            // Calculate consistency for different aspects
            let gaze_consistency = 1.0 - self.calculate_gaze_movement(&current.gaze_position, &previous.gaze_position)?;
            let attention_consistency = 1.0 - (current.attention_level - previous.attention_level).abs();
            let engagement_consistency = 1.0 - (current.engagement_score - previous.engagement_score).abs();

            let aspect_consistency = (gaze_consistency + attention_consistency + engagement_consistency) / 3.0;
            consistency_sum += aspect_consistency;
            count += 1;
        }

        let consistency_score = if count > 0 { consistency_sum / count as f64 } else { 1.0 };
        self.current_behavior.consistency_score = consistency_score.max(0.0).min(1.0);

        Ok(())
    }

    fn update_metrics(&mut self) -> Result<(), AfiyahError> {
        if self.behavior_history.len() < 2 {
            return Ok(());
        }

        let recent_behaviors: Vec<&ViewerBehavior> = self.behavior_history.iter().rev().take(50).collect();

        // Calculate average saccade frequency
        let average_saccade_frequency = recent_behaviors.iter()
            .map(|b| b.saccade_frequency)
            .sum::<f64>() / recent_behaviors.len() as f64;

        // Calculate average fixation duration
        let average_fixation_duration = recent_behaviors.iter()
            .map(|b| b.fixation_duration)
            .sum::<f64>() / recent_behaviors.len() as f64;

        // Calculate attention variance
        let attention_values: Vec<f64> = recent_behaviors.iter().map(|b| b.attention_level).collect();
        let attention_mean = attention_values.iter().sum::<f64>() / attention_values.len() as f64;
        let attention_variance = attention_values.iter()
            .map(|&x| (x - attention_mean).powi(2))
            .sum::<f64>() / attention_values.len() as f64;

        // Calculate engagement trend
        let engagement_trend = self.calculate_trend(&recent_behaviors.iter().map(|b| b.engagement_score).collect::<Vec<f64>>())?;

        // Calculate cognitive load trend
        let cognitive_load_trend = self.calculate_trend(&recent_behaviors.iter().map(|b| b.cognitive_load).collect::<Vec<f64>>())?;

        // Calculate behavior stability
        let behavior_stability = recent_behaviors.iter()
            .map(|b| b.consistency_score)
            .sum::<f64>() / recent_behaviors.len() as f64;

        // Calculate adaptation speed
        let adaptation_speed = self.calculate_adaptation_speed(&recent_behaviors)?;

        // Calculate preference consistency
        let preference_consistency = self.calculate_preference_consistency(&recent_behaviors)?;

        let metrics = BehaviorMetrics {
            average_saccade_frequency,
            average_fixation_duration,
            attention_variance,
            engagement_trend,
            cognitive_load_trend,
            behavior_stability,
            adaptation_speed,
            preference_consistency,
        };

        self.metrics_history.push_back(metrics);
        if self.metrics_history.len() > 100 {
            self.metrics_history.pop_front();
        }

        Ok(())
    }

    fn calculate_gaze_movement(&self, pos1: &(f64, f64), pos2: &(f64, f64)) -> Result<f64, AfiyahError> {
        let dx = pos2.0 - pos1.0;
        let dy = pos2.1 - pos1.1;
        Ok((dx * dx + dy * dy).sqrt())
    }

    fn predict_gaze_position(&self, behaviors: &[&ViewerBehavior], prediction_time: f64) -> Result<(f64, f64), AfiyahError> {
        if behaviors.len() < 2 {
            return Ok(self.current_behavior.gaze_position);
        }

        // Simple linear extrapolation
        let recent = behaviors[0];
        let previous = behaviors[1];

        let dx = recent.gaze_position.0 - previous.gaze_position.0;
        let dy = recent.gaze_position.1 - previous.gaze_position.1;

        let predicted_x = recent.gaze_position.0 + dx * prediction_time;
        let predicted_y = recent.gaze_position.1 + dy * prediction_time;

        // Clamp to valid range [0, 1]
        Ok((
            predicted_x.max(0.0).min(1.0),
            predicted_y.max(0.0).min(1.0)
        ))
    }

    fn predict_saccade_frequency(&self, behaviors: &[&ViewerBehavior], prediction_time: f64) -> Result<f64, AfiyahError> {
        if behaviors.is_empty() {
            return Ok(0.0);
        }

        // Use exponential decay for saccade frequency prediction
        let current_frequency = behaviors[0].saccade_frequency;
        let decay_factor = (-prediction_time * 0.5).exp();
        
        Ok(current_frequency * decay_factor)
    }

    fn predict_attention_level(&self, behaviors: &[&ViewerBehavior], prediction_time: f64) -> Result<f64, AfiyahError> {
        if behaviors.is_empty() {
            return Ok(0.5);
        }

        // Use trend analysis for attention prediction
        let trend = self.calculate_trend(&behaviors.iter().map(|b| b.attention_level).collect::<Vec<f64>>())?;
        let current_attention = behaviors[0].attention_level;
        
        let predicted_attention = current_attention + trend * prediction_time;
        Ok(predicted_attention.max(0.0).min(1.0))
    }

    fn predict_engagement_score(&self, behaviors: &[&ViewerBehavior], prediction_time: f64) -> Result<f64, AfiyahError> {
        if behaviors.is_empty() {
            return Ok(0.5);
        }

        // Use trend analysis for engagement prediction
        let trend = self.calculate_trend(&behaviors.iter().map(|b| b.engagement_score).collect::<Vec<f64>>())?;
        let current_engagement = behaviors[0].engagement_score;
        
        let predicted_engagement = current_engagement + trend * prediction_time;
        Ok(predicted_engagement.max(0.0).min(1.0))
    }

    fn calculate_trend(&self, values: &[f64]) -> Result<f64, AfiyahError> {
        if values.len() < 2 {
            return Ok(0.0);
        }

        let n = values.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        Ok(slope)
    }

    fn calculate_adaptation_speed(&self, behaviors: &[&ViewerBehavior]) -> Result<f64, AfiyahError> {
        if behaviors.len() < 3 {
            return Ok(0.0);
        }

        let mut adaptation_sum = 0.0;
        let mut count = 0;

        for i in 2..behaviors.len() {
            let current = behaviors[i-2];
            let middle = behaviors[i-1];
            let recent = behaviors[i];

            // Calculate rate of change in attention level
            let change1 = (middle.attention_level - current.attention_level).abs();
            let change2 = (recent.attention_level - middle.attention_level).abs();
            
            let adaptation_rate = (change1 + change2) / 2.0;
            adaptation_sum += adaptation_rate;
            count += 1;
        }

        Ok(if count > 0 { adaptation_sum / count as f64 } else { 0.0 })
    }

    fn calculate_preference_consistency(&self, behaviors: &[&ViewerBehavior]) -> Result<f64, AfiyahError> {
        if behaviors.is_empty() {
            return Ok(1.0);
        }

        // Calculate consistency of gaze patterns
        let mut consistency_sum = 0.0;
        let mut count = 0;

        for i in 1..behaviors.len() {
            let current = behaviors[i-1];
            let previous = behaviors[i];

            // Calculate consistency based on gaze position stability
            let gaze_consistency = 1.0 - self.calculate_gaze_movement(&current.gaze_position, &previous.gaze_position)?;
            consistency_sum += gaze_consistency;
            count += 1;
        }

        Ok(if count > 0 { consistency_sum / count as f64 } else { 1.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_tracker_creation() {
        let tracker = ViewerBehaviorTracker::new();
        assert!(tracker.is_ok());
    }

    #[test]
    fn test_behavior_update() {
        let mut tracker = ViewerBehaviorTracker::new().unwrap();
        let behavior = ViewerBehavior {
            gaze_position: (0.5, 0.5),
            saccade_frequency: 0.1,
            fixation_duration: 0.5,
            attention_level: 0.8,
            pupil_dilation: 0.6,
            blink_rate: 0.05,
            head_movement: 0.1,
            engagement_score: 0.7,
            consistency_score: 0.9,
            cognitive_load: 0.3,
        };

        let result = tracker.update_behavior(&behavior);
        assert!(result.is_ok());
    }

    #[test]
    fn test_behavior_prediction() {
        let tracker = ViewerBehaviorTracker::new().unwrap();
        let prediction_horizon = Duration::from_millis(100);
        
        let result = tracker.predict_behavior(prediction_horizon);
        assert!(result.is_ok());
    }
}