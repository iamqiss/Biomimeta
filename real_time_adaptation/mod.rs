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

//! Real-Time Biological Adaptation System
//! 
//! Implements sophisticated real-time adaptation mechanisms that dynamically
//! adjust compression parameters based on content characteristics and viewer behavior.
//! 
//! Biological Basis:
//! - Kandel et al. (2013): Principles of Neural Science - Plasticity and adaptation
//! - Bear et al. (2016): Neuroscience - Synaptic plasticity and learning
//! - Turrigiano (2008): The self-tuning neuron: Homeostatic plasticity
//! - Abraham & Bear (1996): Metaplasticity: The plasticity of synaptic plasticity

use ndarray::{Array1, Array2, Array3};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use crate::AfiyahError;

// Re-export all sub-modules
pub mod content_analyzer;
pub mod viewer_behavior_tracker;
pub mod adaptation_controller;
pub mod parameter_optimizer;
pub mod performance_monitor;

// Re-export the main types
pub use content_analyzer::{ContentAnalyzer, ContentFeatures, ContentComplexity};
pub use viewer_behavior_tracker::{ViewerBehaviorTracker, ViewerBehavior, BehaviorMetrics};
pub use adaptation_controller::{AdaptationController, AdaptationStrategy, AdaptationState};
pub use parameter_optimizer::{ParameterOptimizer, CompressionParameters, OptimizationTarget};
pub use performance_monitor::{PerformanceMonitor, PerformanceMetrics, SystemHealth};

/// Real-time adaptation configuration
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    pub adaptation_rate: f64,
    pub stability_threshold: f64,
    pub learning_rate: f64,
    pub memory_decay: f64,
    pub prediction_horizon: Duration,
    pub content_analysis_window: Duration,
    pub behavior_tracking_window: Duration,
    pub performance_monitoring_interval: Duration,
    pub emergency_threshold: f64,
    pub optimization_aggressiveness: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.1,
            stability_threshold: 0.05,
            learning_rate: 0.01,
            memory_decay: 0.95,
            prediction_horizon: Duration::from_millis(500),
            content_analysis_window: Duration::from_millis(100),
            behavior_tracking_window: Duration::from_secs(10),
            performance_monitoring_interval: Duration::from_millis(50),
            emergency_threshold: 0.8,
            optimization_aggressiveness: 0.7,
        }
    }
}

/// Real-time adaptation output
#[derive(Debug, Clone)]
pub struct AdaptationOutput {
    pub optimized_parameters: CompressionParameters,
    pub adaptation_confidence: f64,
    pub content_complexity: ContentComplexity,
    pub viewer_behavior: ViewerBehavior,
    pub performance_metrics: PerformanceMetrics,
    pub adaptation_recommendations: Vec<AdaptationRecommendation>,
    pub system_health: SystemHealth,
}

/// Adaptation recommendation
#[derive(Debug, Clone)]
pub struct AdaptationRecommendation {
    pub parameter: String,
    pub current_value: f64,
    pub recommended_value: f64,
    pub confidence: f64,
    pub urgency: AdaptationUrgency,
    pub biological_rationale: String,
}

/// Urgency level for adaptation recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationUrgency {
    Low,
    Medium,
    High,
    Critical,
}

/// Main real-time adaptation processor
pub struct RealTimeAdaptationProcessor {
    content_analyzer: ContentAnalyzer,
    viewer_behavior_tracker: ViewerBehaviorTracker,
    adaptation_controller: AdaptationController,
    parameter_optimizer: ParameterOptimizer,
    performance_monitor: PerformanceMonitor,
    config: AdaptationConfig,
    adaptation_history: VecDeque<AdaptationOutput>,
    last_adaptation: Instant,
    system_state: AdaptationState,
}

impl RealTimeAdaptationProcessor {
    /// Creates a new real-time adaptation processor
    pub fn new() -> Result<Self, AfiyahError> {
        let config = AdaptationConfig::default();
        Self::with_config(config)
    }

    /// Creates a new real-time adaptation processor with custom configuration
    pub fn with_config(config: AdaptationConfig) -> Result<Self, AfiyahError> {
        let content_analyzer = ContentAnalyzer::new()?;
        let viewer_behavior_tracker = ViewerBehaviorTracker::new()?;
        let adaptation_controller = AdaptationController::new()?;
        let parameter_optimizer = ParameterOptimizer::new()?;
        let performance_monitor = PerformanceMonitor::new()?;

        Ok(Self {
            content_analyzer,
            viewer_behavior_tracker,
            adaptation_controller,
            parameter_optimizer,
            performance_monitor,
            config: config.clone(),
            adaptation_history: VecDeque::with_capacity(100),
            last_adaptation: Instant::now(),
            system_state: AdaptationState::Initializing,
        })
    }

    /// Processes real-time adaptation for current frame
    pub fn process_adaptation(&mut self, 
                             current_frame: &Array2<f64>,
                             viewer_input: Option<&ViewerBehavior>,
                             current_parameters: &CompressionParameters) -> Result<AdaptationOutput, AfiyahError> {
        let start_time = Instant::now();

        // Analyze content complexity
        let content_features = self.content_analyzer.analyze_content(current_frame)?;
        let content_complexity = self.content_analyzer.calculate_complexity(&content_features)?;

        // Track viewer behavior
        let viewer_behavior = if let Some(input) = viewer_input {
            self.viewer_behavior_tracker.update_behavior(input)?;
            input.clone()
        } else {
            self.viewer_behavior_tracker.get_current_behavior()?
        };

        // Monitor system performance
        let performance_metrics = self.performance_monitor.collect_metrics()?;
        let system_health = self.performance_monitor.assess_health(&performance_metrics)?;

        // Determine adaptation strategy
        let adaptation_strategy = self.adaptation_controller.determine_strategy(
            &content_complexity,
            &viewer_behavior,
            &performance_metrics,
            &system_health
        )?;

        // Optimize parameters
        let optimized_parameters = self.parameter_optimizer.optimize_parameters(
            current_parameters,
            &content_complexity,
            &viewer_behavior,
            &adaptation_strategy,
            &performance_metrics
        )?;

        // Generate adaptation recommendations
        let adaptation_recommendations = self.generate_adaptation_recommendations(
            current_parameters,
            &optimized_parameters,
            &content_complexity,
            &viewer_behavior,
            &performance_metrics
        )?;

        // Calculate adaptation confidence
        let adaptation_confidence = self.calculate_adaptation_confidence(
            &content_complexity,
            &viewer_behavior,
            &performance_metrics,
            &adaptation_strategy
        )?;

        // Update system state
        self.update_system_state(&adaptation_strategy, &system_health)?;

        // Create adaptation output
        let output = AdaptationOutput {
            optimized_parameters,
            adaptation_confidence,
            content_complexity,
            viewer_behavior,
            performance_metrics,
            adaptation_recommendations,
            system_health,
        };

        // Update adaptation history
        self.update_adaptation_history(&output)?;

        // Update last adaptation time
        self.last_adaptation = start_time;

        Ok(output)
    }

    /// Updates the adaptation configuration
    pub fn update_config(&mut self, config: AdaptationConfig) -> Result<(), AfiyahError> {
        self.config = config.clone();
        
        // Update component configurations
        self.content_analyzer.update_config(config.content_analysis_window)?;
        self.viewer_behavior_tracker.update_config(config.behavior_tracking_window)?;
        self.adaptation_controller.update_config(config.adaptation_rate, config.stability_threshold)?;
        self.parameter_optimizer.update_config(config.learning_rate, config.optimization_aggressiveness)?;
        self.performance_monitor.update_config(config.performance_monitoring_interval)?;
        
        Ok(())
    }

    /// Gets current adaptation configuration
    pub fn get_config(&self) -> &AdaptationConfig {
        &self.config
    }

    /// Gets current system state
    pub fn get_system_state(&self) -> &AdaptationState {
        &self.system_state
    }

    /// Gets adaptation history
    pub fn get_adaptation_history(&self) -> &VecDeque<AdaptationOutput> {
        &self.adaptation_history
    }

    /// Resets the adaptation system
    pub fn reset(&mut self) -> Result<(), AfiyahError> {
        self.adaptation_history.clear();
        self.last_adaptation = Instant::now();
        self.system_state = AdaptationState::Initializing;
        
        // Reset all components
        self.content_analyzer.reset()?;
        self.viewer_behavior_tracker.reset()?;
        self.adaptation_controller.reset()?;
        self.parameter_optimizer.reset()?;
        self.performance_monitor.reset()?;
        
        Ok(())
    }

    fn generate_adaptation_recommendations(&self,
                                         current: &CompressionParameters,
                                         optimized: &CompressionParameters,
                                         content: &ContentComplexity,
                                         behavior: &ViewerBehavior,
                                         performance: &PerformanceMetrics) -> Result<Vec<AdaptationRecommendation>, AfiyahError> {
        let mut recommendations = Vec::new();

        // Analyze parameter differences
        let parameter_diffs = self.calculate_parameter_differences(current, optimized)?;

        for (param_name, (current_val, optimized_val, diff)) in parameter_diffs {
            let confidence = self.calculate_parameter_confidence(&param_name, content, behavior, performance)?;
            let urgency = self.determine_parameter_urgency(&param_name, diff, performance)?;
            let biological_rationale = self.generate_biological_rationale(&param_name, content, behavior)?;

            if diff.abs() > self.config.stability_threshold {
                recommendations.push(AdaptationRecommendation {
                    parameter: param_name,
                    current_value: current_val,
                    recommended_value: optimized_val,
                    confidence,
                    urgency,
                    biological_rationale,
                });
            }
        }

        // Sort by urgency and confidence
        recommendations.sort_by(|a, b| {
            let urgency_order = match (&a.urgency, &b.urgency) {
                (AdaptationUrgency::Critical, _) => std::cmp::Ordering::Less,
                (_, AdaptationUrgency::Critical) => std::cmp::Ordering::Greater,
                (AdaptationUrgency::High, _) => std::cmp::Ordering::Less,
                (_, AdaptationUrgency::High) => std::cmp::Ordering::Greater,
                (AdaptationUrgency::Medium, _) => std::cmp::Ordering::Less,
                (_, AdaptationUrgency::Medium) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            };
            
            if urgency_order == std::cmp::Ordering::Equal {
                b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                urgency_order
            }
        });

        Ok(recommendations)
    }

    fn calculate_parameter_differences(&self, current: &CompressionParameters, optimized: &CompressionParameters) -> Result<Vec<(String, (f64, f64, f64))>, AfiyahError> {
        let mut differences = Vec::new();

        // Compare all parameters
        differences.push(("compression_ratio".to_string(), (current.compression_ratio, optimized.compression_ratio, optimized.compression_ratio - current.compression_ratio)));
        differences.push(("quality_threshold".to_string(), (current.quality_threshold, optimized.quality_threshold, optimized.quality_threshold - current.quality_threshold)));
        differences.push(("temporal_window".to_string(), (current.temporal_window, optimized.temporal_window, optimized.temporal_window - current.temporal_window)));
        differences.push(("spatial_resolution".to_string(), (current.spatial_resolution, optimized.spatial_resolution, optimized.spatial_resolution - current.spatial_resolution)));
        differences.push(("attention_weight".to_string(), (current.attention_weight, optimized.attention_weight, optimized.attention_weight - current.attention_weight)));

        Ok(differences)
    }

    fn calculate_parameter_confidence(&self, param_name: &str, content: &ContentComplexity, behavior: &ViewerBehavior, performance: &PerformanceMetrics) -> Result<f64, AfiyahError> {
        // Calculate confidence based on content stability, behavior consistency, and performance metrics
        let content_stability = 1.0 - content.variance;
        let behavior_consistency = behavior.consistency_score;
        let performance_stability = 1.0 - performance.variance;

        let base_confidence = (content_stability + behavior_consistency + performance_stability) / 3.0;

        // Adjust based on parameter type
        let parameter_confidence = match param_name {
            "compression_ratio" => base_confidence * 0.9, // High confidence for compression ratio
            "quality_threshold" => base_confidence * 0.8, // Medium-high confidence for quality
            "temporal_window" => base_confidence * 0.7,   // Medium confidence for temporal
            "spatial_resolution" => base_confidence * 0.8, // Medium-high confidence for spatial
            "attention_weight" => base_confidence * 0.6,  // Lower confidence for attention
            _ => base_confidence * 0.5, // Default confidence
        };

        Ok(parameter_confidence.min(1.0).max(0.0))
    }

    fn determine_parameter_urgency(&self, param_name: &str, diff: f64, performance: &PerformanceMetrics) -> Result<AdaptationUrgency, AfiyahError> {
        let diff_magnitude = diff.abs();
        let performance_pressure = 1.0 - performance.efficiency;

        let urgency_score = diff_magnitude * 0.7 + performance_pressure * 0.3;

        let urgency = match param_name {
            "compression_ratio" if urgency_score > 0.7 => AdaptationUrgency::Critical,
            "quality_threshold" if urgency_score > 0.6 => AdaptationUrgency::High,
            "temporal_window" if urgency_score > 0.5 => AdaptationUrgency::Medium,
            "spatial_resolution" if urgency_score > 0.6 => AdaptationUrgency::High,
            "attention_weight" if urgency_score > 0.4 => AdaptationUrgency::Medium,
            _ if urgency_score > 0.8 => AdaptationUrgency::Critical,
            _ if urgency_score > 0.6 => AdaptationUrgency::High,
            _ if urgency_score > 0.4 => AdaptationUrgency::Medium,
            _ => AdaptationUrgency::Low,
        };

        Ok(urgency)
    }

    fn generate_biological_rationale(&self, param_name: &str, content: &ContentComplexity, behavior: &ViewerBehavior) -> Result<String, AfiyahError> {
        let rationale = match param_name {
            "compression_ratio" => {
                if content.overall_complexity > 0.7 {
                    "High content complexity requires increased compression to maintain biological processing efficiency".to_string()
                } else {
                    "Low content complexity allows for higher quality preservation".to_string()
                }
            },
            "quality_threshold" => {
                if behavior.attention_level > 0.8 {
                    "High viewer attention requires enhanced quality preservation".to_string()
                } else {
                    "Reduced attention allows for quality optimization".to_string()
                }
            },
            "temporal_window" => {
                if content.variance > 0.6 {
                    "High temporal variance requires extended integration window".to_string()
                } else {
                    "Stable temporal content allows for shorter integration window".to_string()
                }
            },
            "spatial_resolution" => {
                if content.spatial_complexity > 0.7 {
                    "High spatial complexity requires enhanced resolution".to_string()
                } else {
                    "Simple spatial content allows for resolution optimization".to_string()
                }
            },
            "attention_weight" => {
                if behavior.saccade_frequency > 0.6 {
                    "High saccade frequency requires increased attention weighting".to_string()
                } else {
                    "Stable gaze allows for attention optimization".to_string()
                }
            },
            _ => "Parameter adjustment based on biological processing requirements".to_string(),
        };

        Ok(rationale)
    }

    fn calculate_adaptation_confidence(&self, content: &ContentComplexity, behavior: &ViewerBehavior, performance: &PerformanceMetrics, strategy: &AdaptationStrategy) -> Result<f64, AfiyahError> {
        let content_confidence = 1.0 - content.variance;
        let behavior_confidence = behavior.consistency_score;
        let performance_confidence = 1.0 - performance.variance;
        let strategy_confidence = 0.8; // Default confidence for strategy

        let overall_confidence = (content_confidence + behavior_confidence + performance_confidence + strategy_confidence) / 4.0;
        Ok(overall_confidence.min(1.0).max(0.0))
    }

    fn update_system_state(&mut self, strategy: &AdaptationStrategy, health: &SystemHealth) -> Result<(), AfiyahError> {
        // Update system state based on adaptation strategy and system health
        self.system_state = match (strategy, health) {
            (AdaptationStrategy::Emergency, _) => AdaptationState::Emergency,
            (_, SystemHealth::Critical) => AdaptationState::Emergency,
            (AdaptationStrategy::Aggressive, SystemHealth::Degraded) => AdaptationState::Adapting,
            (AdaptationStrategy::Conservative, SystemHealth::Good) => AdaptationState::Stable,
            (AdaptationStrategy::Balanced, SystemHealth::Optimal) => AdaptationState::Optimal,
            _ => AdaptationState::Adapting,
        };

        Ok(())
    }

    fn update_adaptation_history(&mut self, output: &AdaptationOutput) -> Result<(), AfiyahError> {
        self.adaptation_history.push_back(output.clone());
        
        // Maintain history size
        if self.adaptation_history.len() > 100 {
            self.adaptation_history.pop_front();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptation_config_default() {
        let config = AdaptationConfig::default();
        assert_eq!(config.adaptation_rate, 0.1);
        assert_eq!(config.stability_threshold, 0.05);
    }

    #[test]
    fn test_adaptation_processor_creation() {
        let processor = RealTimeAdaptationProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_adaptation_processing() {
        let mut processor = RealTimeAdaptationProcessor::new().unwrap();
        let frame = ndarray::Array2::ones((64, 64));
        let parameters = CompressionParameters::default();
        
        let result = processor.process_adaptation(&frame, None, &parameters);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(output.adaptation_confidence >= 0.0 && output.adaptation_confidence <= 1.0);
    }
}