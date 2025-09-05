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

//! Adaptation Controller for Real-Time Adaptation
//! 
//! Controls the adaptation strategy based on content complexity, viewer behavior, and system performance.
//! 
//! Biological Basis:
//! - Kandel et al. (2013): Principles of Neural Science - Adaptive mechanisms
//! - Turrigiano (2008): The self-tuning neuron: Homeostatic plasticity
//! - Abraham & Bear (1996): Metaplasticity: The plasticity of synaptic plasticity
//! - Bienenstock et al. (1982): Theory for the development of neuron selectivity

use crate::AfiyahError;
use super::content_analyzer::ContentComplexity;
use super::viewer_behavior_tracker::ViewerBehavior;
use super::performance_monitor::PerformanceMetrics;
use super::performance_monitor::SystemHealth;

/// Adaptation strategy types
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationStrategy {
    Conservative,
    Balanced,
    Aggressive,
    Emergency,
    Stable,
}

/// Adaptation state
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationState {
    Initializing,
    Stable,
    Adapting,
    Optimal,
    Emergency,
}

/// Adaptation controller configuration
#[derive(Debug, Clone)]
pub struct AdaptationControllerConfig {
    pub adaptation_rate: f64,
    pub stability_threshold: f64,
    pub emergency_threshold: f64,
    pub content_weight: f64,
    pub behavior_weight: f64,
    pub performance_weight: f64,
    pub strategy_confidence_threshold: f64,
    pub adaptation_delay: f64,
}

impl Default for AdaptationControllerConfig {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.1,
            stability_threshold: 0.05,
            emergency_threshold: 0.8,
            content_weight: 0.4,
            behavior_weight: 0.3,
            performance_weight: 0.3,
            strategy_confidence_threshold: 0.7,
            adaptation_delay: 0.1,
        }
    }
}

/// Adaptation controller
pub struct AdaptationController {
    config: AdaptationControllerConfig,
    current_strategy: AdaptationStrategy,
    strategy_history: Vec<AdaptationStrategy>,
    adaptation_timer: f64,
    last_adaptation: f64,
    strategy_confidence: f64,
    adaptation_urgency: f64,
}

impl AdaptationController {
    /// Creates a new adaptation controller
    pub fn new() -> Result<Self, AfiyahError> {
        let config = AdaptationControllerConfig::default();
        Self::with_config(config)
    }

    /// Creates a new adaptation controller with custom configuration
    pub fn with_config(config: AdaptationControllerConfig) -> Result<Self, AfiyahError> {
        Ok(Self {
            config: config.clone(),
            current_strategy: AdaptationStrategy::Stable,
            strategy_history: Vec::new(),
            adaptation_timer: 0.0,
            last_adaptation: 0.0,
            strategy_confidence: 1.0,
            adaptation_urgency: 0.0,
        })
    }

    /// Determines the optimal adaptation strategy
    pub fn determine_strategy(&mut self, 
                            content: &ContentComplexity,
                            behavior: &ViewerBehavior,
                            performance: &PerformanceMetrics,
                            health: &SystemHealth) -> Result<AdaptationStrategy, AfiyahError> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Calculate adaptation urgency
        self.adaptation_urgency = self.calculate_adaptation_urgency(content, behavior, performance, health)?;

        // Check if adaptation is needed
        if !self.should_adapt(current_time)? {
            return Ok(self.current_strategy.clone());
        }

        // Determine new strategy based on current conditions
        let new_strategy = self.select_adaptation_strategy(content, behavior, performance, health)?;

        // Calculate strategy confidence
        self.strategy_confidence = self.calculate_strategy_confidence(content, behavior, performance, health, &new_strategy)?;

        // Update strategy if confidence is sufficient
        if self.strategy_confidence >= self.config.strategy_confidence_threshold {
            self.current_strategy = new_strategy.clone();
            self.strategy_history.push(new_strategy.clone());
            if self.strategy_history.len() > 100 {
                self.strategy_history.remove(0);
            }
            self.last_adaptation = current_time;
        }

        Ok(self.current_strategy.clone())
    }

    /// Gets current adaptation strategy
    pub fn get_current_strategy(&self) -> &AdaptationStrategy {
        &self.current_strategy
    }

    /// Gets strategy confidence
    pub fn get_strategy_confidence(&self) -> f64 {
        self.strategy_confidence
    }

    /// Gets adaptation urgency
    pub fn get_adaptation_urgency(&self) -> f64 {
        self.adaptation_urgency
    }

    /// Updates the controller configuration
    pub fn update_config(&mut self, adaptation_rate: f64, stability_threshold: f64) -> Result<(), AfiyahError> {
        self.config.adaptation_rate = adaptation_rate;
        self.config.stability_threshold = stability_threshold;
        Ok(())
    }

    /// Resets the controller
    pub fn reset(&mut self) -> Result<(), AfiyahError> {
        self.current_strategy = AdaptationStrategy::Stable;
        self.strategy_history.clear();
        self.adaptation_timer = 0.0;
        self.last_adaptation = 0.0;
        self.strategy_confidence = 1.0;
        self.adaptation_urgency = 0.0;
        Ok(())
    }

    /// Gets current configuration
    pub fn get_config(&self) -> &AdaptationControllerConfig {
        &self.config
    }

    fn calculate_adaptation_urgency(&self, 
                                  content: &ContentComplexity,
                                  behavior: &ViewerBehavior,
                                  performance: &PerformanceMetrics,
                                  health: &SystemHealth) -> Result<f64, AfiyahError> {
        // Calculate urgency based on multiple factors
        let content_urgency = self.calculate_content_urgency(content)?;
        let behavior_urgency = self.calculate_behavior_urgency(behavior)?;
        let performance_urgency = self.calculate_performance_urgency(performance)?;
        let health_urgency = self.calculate_health_urgency(health)?;

        // Weighted combination
        let urgency = (
            content_urgency * self.config.content_weight +
            behavior_urgency * self.config.behavior_weight +
            performance_urgency * self.config.performance_weight +
            health_urgency * 0.1 // Health is critical but less frequent
        );

        Ok(urgency.min(1.0).max(0.0))
    }

    fn calculate_content_urgency(&self, content: &ContentComplexity) -> Result<f64, AfiyahError> {
        // High urgency for high complexity or high variance
        let complexity_urgency = content.overall_complexity;
        let variance_urgency = content.variance;
        let stability_urgency = 1.0 - content.stability;

        let urgency = (complexity_urgency + variance_urgency + stability_urgency) / 3.0;
        Ok(urgency.min(1.0).max(0.0))
    }

    fn calculate_behavior_urgency(&self, behavior: &ViewerBehavior) -> Result<f64, AfiyahError> {
        // High urgency for high saccade frequency, low attention, or low consistency
        let saccade_urgency = behavior.saccade_frequency;
        let attention_urgency = 1.0 - behavior.attention_level;
        let consistency_urgency = 1.0 - behavior.consistency_score;
        let engagement_urgency = 1.0 - behavior.engagement_score;

        let urgency = (saccade_urgency + attention_urgency + consistency_urgency + engagement_urgency) / 4.0;
        Ok(urgency.min(1.0).max(0.0))
    }

    fn calculate_performance_urgency(&self, performance: &PerformanceMetrics) -> Result<f64, AfiyahError> {
        // High urgency for low efficiency or high variance
        let efficiency_urgency = 1.0 - performance.efficiency;
        let variance_urgency = performance.variance;
        let latency_urgency = performance.latency / 1000.0; // Normalize to seconds

        let urgency = (efficiency_urgency + variance_urgency + latency_urgency) / 3.0;
        Ok(urgency.min(1.0).max(0.0))
    }

    fn calculate_health_urgency(&self, health: &SystemHealth) -> Result<f64, AfiyahError> {
        match health {
            SystemHealth::Critical => Ok(1.0),
            SystemHealth::Degraded => Ok(0.7),
            SystemHealth::Good => Ok(0.3),
            SystemHealth::Optimal => Ok(0.0),
        }
    }

    fn should_adapt(&self, current_time: f64) -> Result<bool, AfiyahError> {
        // Check if enough time has passed since last adaptation
        let time_since_last = current_time - self.last_adaptation;
        if time_since_last < self.config.adaptation_delay {
            return Ok(false);
        }

        // Check if urgency exceeds threshold
        if self.adaptation_urgency > self.config.stability_threshold {
            return Ok(true);
        }

        // Check if strategy has been stable for too long
        if self.strategy_history.len() > 10 {
            let recent_strategies = &self.strategy_history[self.strategy_history.len()-10..];
            let all_same = recent_strategies.iter().all(|s| s == &self.current_strategy);
            if all_same && time_since_last > 10.0 { // 10 seconds
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn select_adaptation_strategy(&self, 
                                content: &ContentComplexity,
                                behavior: &ViewerBehavior,
                                performance: &PerformanceMetrics,
                                health: &SystemHealth) -> Result<AdaptationStrategy, AfiyahError> {
        // Emergency strategy for critical conditions
        if self.adaptation_urgency > self.config.emergency_threshold || 
           matches!(health, SystemHealth::Critical) {
            return Ok(AdaptationStrategy::Emergency);
        }

        // Determine strategy based on content complexity and viewer behavior
        let content_score = self.calculate_content_score(content)?;
        let behavior_score = self.calculate_behavior_score(behavior)?;
        let performance_score = self.calculate_performance_score(performance)?;

        let combined_score = (
            content_score * self.config.content_weight +
            behavior_score * self.config.behavior_weight +
            performance_score * self.config.performance_weight
        );

        // Select strategy based on combined score
        let strategy = if combined_score > 0.8 {
            AdaptationStrategy::Aggressive
        } else if combined_score > 0.6 {
            AdaptationStrategy::Balanced
        } else if combined_score > 0.3 {
            AdaptationStrategy::Conservative
        } else {
            AdaptationStrategy::Stable
        };

        Ok(strategy)
    }

    fn calculate_content_score(&self, content: &ContentComplexity) -> Result<f64, AfiyahError> {
        // Higher score for more complex content that needs adaptation
        let complexity_score = content.overall_complexity;
        let variance_score = content.variance;
        let difficulty_score = content.compression_difficulty;

        let score = (complexity_score + variance_score + difficulty_score) / 3.0;
        Ok(score.min(1.0).max(0.0))
    }

    fn calculate_behavior_score(&self, behavior: &ViewerBehavior) -> Result<f64, AfiyahError> {
        // Higher score for behaviors that indicate need for adaptation
        let saccade_score = behavior.saccade_frequency;
        let attention_score = 1.0 - behavior.attention_level;
        let engagement_score = 1.0 - behavior.engagement_score;
        let cognitive_score = behavior.cognitive_load;

        let score = (saccade_score + attention_score + engagement_score + cognitive_score) / 4.0;
        Ok(score.min(1.0).max(0.0))
    }

    fn calculate_performance_score(&self, performance: &PerformanceMetrics) -> Result<f64, AfiyahError> {
        // Higher score for poor performance that needs adaptation
        let efficiency_score = 1.0 - performance.efficiency;
        let variance_score = performance.variance;
        let latency_score = performance.latency / 1000.0; // Normalize

        let score = (efficiency_score + variance_score + latency_score) / 3.0;
        Ok(score.min(1.0).max(0.0))
    }

    fn calculate_strategy_confidence(&self, 
                                   content: &ContentComplexity,
                                   behavior: &ViewerBehavior,
                                   performance: &PerformanceMetrics,
                                   health: &SystemHealth,
                                   strategy: &AdaptationStrategy) -> Result<f64, AfiyahError> {
        // Calculate confidence based on consistency of indicators
        let content_consistency = 1.0 - content.variance;
        let behavior_consistency = behavior.consistency_score;
        let performance_consistency = 1.0 - performance.variance;

        let base_confidence = (content_consistency + behavior_consistency + performance_consistency) / 3.0;

        // Adjust confidence based on strategy appropriateness
        let strategy_confidence = match strategy {
            AdaptationStrategy::Emergency => {
                if self.adaptation_urgency > 0.8 { 1.0 } else { 0.3 }
            },
            AdaptationStrategy::Aggressive => {
                if self.adaptation_urgency > 0.6 { 0.9 } else { 0.5 }
            },
            AdaptationStrategy::Balanced => {
                if 0.3 <= self.adaptation_urgency && self.adaptation_urgency <= 0.7 { 0.9 } else { 0.6 }
            },
            AdaptationStrategy::Conservative => {
                if self.adaptation_urgency < 0.4 { 0.9 } else { 0.5 }
            },
            AdaptationStrategy::Stable => {
                if self.adaptation_urgency < 0.2 { 1.0 } else { 0.3 }
            },
        };

        let final_confidence = (base_confidence + strategy_confidence) / 2.0;
        Ok(final_confidence.min(1.0).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::content_analyzer::ContentComplexity;
    use super::super::viewer_behavior_tracker::ViewerBehavior;
    use super::super::performance_monitor::{PerformanceMetrics, SystemHealth};

    #[test]
    fn test_adaptation_controller_creation() {
        let controller = AdaptationController::new();
        assert!(controller.is_ok());
    }

    #[test]
    fn test_strategy_determination() {
        let mut controller = AdaptationController::new().unwrap();
        
        let content = ContentComplexity {
            overall_complexity: 0.8,
            spatial_complexity: 0.7,
            temporal_complexity: 0.6,
            perceptual_complexity: 0.8,
            compression_difficulty: 0.7,
            biological_processing_load: 0.6,
            variance: 0.3,
            stability: 0.7,
        };

        let behavior = ViewerBehavior {
            gaze_position: (0.5, 0.5),
            saccade_frequency: 0.3,
            fixation_duration: 0.2,
            attention_level: 0.6,
            pupil_dilation: 0.7,
            blink_rate: 0.1,
            head_movement: 0.2,
            engagement_score: 0.5,
            consistency_score: 0.8,
            cognitive_load: 0.6,
        };

        let performance = PerformanceMetrics {
            efficiency: 0.7,
            latency: 50.0,
            memory_usage: 0.6,
            cpu_usage: 0.5,
            variance: 0.2,
            throughput: 100.0,
        };

        let health = SystemHealth::Good;

        let result = controller.determine_strategy(&content, &behavior, &performance, &health);
        assert!(result.is_ok());
    }

    #[test]
    fn test_urgency_calculation() {
        let controller = AdaptationController::new().unwrap();
        
        let content = ContentComplexity {
            overall_complexity: 0.9,
            spatial_complexity: 0.8,
            temporal_complexity: 0.7,
            perceptual_complexity: 0.9,
            compression_difficulty: 0.8,
            biological_processing_load: 0.7,
            variance: 0.5,
            stability: 0.5,
        };

        let behavior = ViewerBehavior {
            gaze_position: (0.5, 0.5),
            saccade_frequency: 0.8,
            fixation_duration: 0.1,
            attention_level: 0.3,
            pupil_dilation: 0.8,
            blink_rate: 0.2,
            head_movement: 0.5,
            engagement_score: 0.2,
            consistency_score: 0.3,
            cognitive_load: 0.8,
        };

        let performance = PerformanceMetrics {
            efficiency: 0.3,
            latency: 200.0,
            memory_usage: 0.9,
            cpu_usage: 0.8,
            variance: 0.6,
            throughput: 30.0,
        };

        let health = SystemHealth::Critical;

        let urgency = controller.calculate_adaptation_urgency(&content, &behavior, &performance, &health).unwrap();
        assert!(urgency > 0.5); // Should be high urgency
    }
}