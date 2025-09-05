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

//! Parameter Optimizer for Real-Time Adaptation
//! 
//! Optimizes compression parameters based on content complexity, viewer behavior, and adaptation strategy.
//! 
//! Biological Basis:
//! - Kandel et al. (2013): Principles of Neural Science - Synaptic plasticity and learning
//! - Hebb (1949): The Organization of Behavior - Hebbian learning
//! - Sutton & Barto (2018): Reinforcement Learning - Adaptive parameter optimization
//! - Dayan & Abbott (2001): Theoretical Neuroscience - Computational approaches to learning

use crate::AfiyahError;
use super::content_analyzer::ContentComplexity;
use super::viewer_behavior_tracker::ViewerBehavior;
use super::adaptation_controller::AdaptationStrategy;
use super::performance_monitor::PerformanceMetrics;

/// Compression parameters
#[derive(Debug, Clone)]
pub struct CompressionParameters {
    pub compression_ratio: f64,
    pub quality_threshold: f64,
    pub temporal_window: f64,
    pub spatial_resolution: f64,
    pub attention_weight: f64,
    pub adaptation_rate: f64,
    pub biological_accuracy: f64,
    pub perceptual_quality: f64,
}

impl Default for CompressionParameters {
    fn default() -> Self {
        Self {
            compression_ratio: 0.95,
            quality_threshold: 0.8,
            temporal_window: 0.2,
            spatial_resolution: 1.0,
            attention_weight: 0.5,
            adaptation_rate: 0.1,
            biological_accuracy: 0.95,
            perceptual_quality: 0.98,
        }
    }
}

/// Optimization target
#[derive(Debug, Clone)]
pub struct OptimizationTarget {
    pub compression_efficiency: f64,
    pub perceptual_quality: f64,
    pub biological_accuracy: f64,
    pub processing_speed: f64,
    pub memory_efficiency: f64,
    pub adaptation_speed: f64,
}

impl Default for OptimizationTarget {
    fn default() -> Self {
        Self {
            compression_efficiency: 0.95,
            perceptual_quality: 0.98,
            biological_accuracy: 0.95,
            processing_speed: 0.9,
            memory_efficiency: 0.8,
            adaptation_speed: 0.7,
        }
    }
}

/// Parameter optimizer
pub struct ParameterOptimizer {
    learning_rate: f64,
    optimization_aggressiveness: f64,
    parameter_bounds: ParameterBounds,
    optimization_history: Vec<(CompressionParameters, f64)>, // (parameters, fitness)
    current_target: OptimizationTarget,
    adaptation_memory: Vec<AdaptationMemory>,
}

/// Parameter bounds for optimization
#[derive(Debug, Clone)]
struct ParameterBounds {
    compression_ratio: (f64, f64),
    quality_threshold: (f64, f64),
    temporal_window: (f64, f64),
    spatial_resolution: (f64, f64),
    attention_weight: (f64, f64),
    adaptation_rate: (f64, f64),
    biological_accuracy: (f64, f64),
    perceptual_quality: (f64, f64),
}

impl Default for ParameterBounds {
    fn default() -> Self {
        Self {
            compression_ratio: (0.5, 0.99),
            quality_threshold: (0.1, 1.0),
            temporal_window: (0.05, 1.0),
            spatial_resolution: (0.1, 2.0),
            attention_weight: (0.0, 1.0),
            adaptation_rate: (0.01, 0.5),
            biological_accuracy: (0.8, 1.0),
            perceptual_quality: (0.8, 1.0),
        }
    }
}

/// Adaptation memory for learning
#[derive(Debug, Clone)]
struct AdaptationMemory {
    content_complexity: f64,
    behavior_engagement: f64,
    performance_efficiency: f64,
    parameters: CompressionParameters,
    fitness: f64,
    timestamp: f64,
}

impl ParameterOptimizer {
    /// Creates a new parameter optimizer
    pub fn new() -> Result<Self, AfiyahError> {
        let learning_rate = 0.01;
        let optimization_aggressiveness = 0.7;
        Self::with_config(learning_rate, optimization_aggressiveness)
    }

    /// Creates a new parameter optimizer with custom configuration
    pub fn with_config(learning_rate: f64, optimization_aggressiveness: f64) -> Result<Self, AfiyahError> {
        Ok(Self {
            learning_rate,
            optimization_aggressiveness,
            parameter_bounds: ParameterBounds::default(),
            optimization_history: Vec::new(),
            current_target: OptimizationTarget::default(),
            adaptation_memory: Vec::new(),
        })
    }

    /// Optimizes parameters based on current conditions
    pub fn optimize_parameters(&mut self,
                             current: &CompressionParameters,
                             content: &ContentComplexity,
                             behavior: &ViewerBehavior,
                             strategy: &AdaptationStrategy,
                             performance: &PerformanceMetrics) -> Result<CompressionParameters, AfiyahError> {
        // Calculate current fitness
        let current_fitness = self.calculate_fitness(current, content, behavior, performance)?;

        // Determine optimization approach based on strategy
        let optimization_approach = self.determine_optimization_approach(strategy, content, behavior, performance)?;

        // Generate candidate parameters
        let candidate_parameters = self.generate_candidate_parameters(
            current, content, behavior, strategy, &optimization_approach)?;

        // Evaluate candidates
        let best_candidate = self.evaluate_candidates(
            &candidate_parameters, content, behavior, performance)?;

        // Apply learning from adaptation memory
        let learned_parameters = self.apply_learning(&best_candidate, content, behavior)?;

        // Update adaptation memory
        self.update_adaptation_memory(content, behavior, performance, &learned_parameters, current_fitness)?;

        // Update optimization history
        self.optimization_history.push((learned_parameters.clone(), current_fitness));
        if self.optimization_history.len() > 1000 {
            self.optimization_history.remove(0);
        }

        Ok(learned_parameters)
    }

    /// Updates the optimizer configuration
    pub fn update_config(&mut self, learning_rate: f64, optimization_aggressiveness: f64) -> Result<(), AfiyahError> {
        self.learning_rate = learning_rate;
        self.optimization_aggressiveness = optimization_aggressiveness;
        Ok(())
    }

    /// Resets the optimizer
    pub fn reset(&mut self) -> Result<(), AfiyahError> {
        self.optimization_history.clear();
        self.adaptation_memory.clear();
        Ok(())
    }

    /// Gets current configuration
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Gets optimization aggressiveness
    pub fn get_optimization_aggressiveness(&self) -> f64 {
        self.optimization_aggressiveness
    }

    fn calculate_fitness(&self, 
                        parameters: &CompressionParameters,
                        content: &ContentComplexity,
                        behavior: &ViewerBehavior,
                        performance: &PerformanceMetrics) -> Result<f64, AfiyahError> {
        // Calculate fitness based on multiple objectives
        let compression_fitness = self.calculate_compression_fitness(parameters, content)?;
        let quality_fitness = self.calculate_quality_fitness(parameters, behavior)?;
        let biological_fitness = self.calculate_biological_fitness(parameters, content)?;
        let performance_fitness = self.calculate_performance_fitness(parameters, performance)?;

        // Weighted combination
        let fitness = (
            compression_fitness * 0.3 +
            quality_fitness * 0.3 +
            biological_fitness * 0.25 +
            performance_fitness * 0.15
        );

        Ok(fitness.min(1.0).max(0.0))
    }

    fn calculate_compression_fitness(&self, parameters: &CompressionParameters, content: &ContentComplexity) -> Result<f64, AfiyahError> {
        // Higher fitness for better compression efficiency
        let compression_efficiency = parameters.compression_ratio;
        let content_adaptation = 1.0 - (parameters.compression_ratio - content.compression_difficulty).abs();
        
        let fitness = (compression_efficiency + content_adaptation) / 2.0;
        Ok(fitness.min(1.0).max(0.0))
    }

    fn calculate_quality_fitness(&self, parameters: &CompressionParameters, behavior: &ViewerBehavior) -> Result<f64, AfiyahError> {
        // Higher fitness for better perceptual quality based on viewer behavior
        let quality_score = parameters.perceptual_quality;
        let attention_adaptation = 1.0 - (parameters.attention_weight - behavior.attention_level).abs();
        let engagement_adaptation = 1.0 - (parameters.quality_threshold - behavior.engagement_score).abs();
        
        let fitness = (quality_score + attention_adaptation + engagement_adaptation) / 3.0;
        Ok(fitness.min(1.0).max(0.0))
    }

    fn calculate_biological_fitness(&self, parameters: &CompressionParameters, content: &ContentComplexity) -> Result<f64, AfiyahError> {
        // Higher fitness for better biological accuracy
        let biological_accuracy = parameters.biological_accuracy;
        let processing_adaptation = 1.0 - (parameters.spatial_resolution - content.spatial_complexity).abs();
        let temporal_adaptation = 1.0 - (parameters.temporal_window - content.temporal_complexity).abs();
        
        let fitness = (biological_accuracy + processing_adaptation + temporal_adaptation) / 3.0;
        Ok(fitness.min(1.0).max(0.0))
    }

    fn calculate_performance_fitness(&self, parameters: &CompressionParameters, performance: &PerformanceMetrics) -> Result<f64, AfiyahError> {
        // Higher fitness for better performance
        let efficiency_score = performance.efficiency;
        let latency_score = 1.0 - (performance.latency / 1000.0).min(1.0);
        let memory_score = 1.0 - performance.memory_usage;
        
        let fitness = (efficiency_score + latency_score + memory_score) / 3.0;
        Ok(fitness.min(1.0).max(0.0))
    }

    fn determine_optimization_approach(&self, 
                                     strategy: &AdaptationStrategy,
                                     content: &ContentComplexity,
                                     behavior: &ViewerBehavior,
                                     performance: &PerformanceMetrics) -> Result<OptimizationApproach, AfiyahError> {
        match strategy {
            AdaptationStrategy::Emergency => {
                Ok(OptimizationApproach {
                    exploration_rate: 0.1,
                    exploitation_rate: 0.9,
                    mutation_rate: 0.05,
                    crossover_rate: 0.8,
                    population_size: 10,
                    generations: 5,
                })
            },
            AdaptationStrategy::Aggressive => {
                Ok(OptimizationApproach {
                    exploration_rate: 0.3,
                    exploitation_rate: 0.7,
                    mutation_rate: 0.1,
                    crossover_rate: 0.7,
                    population_size: 20,
                    generations: 10,
                })
            },
            AdaptationStrategy::Balanced => {
                Ok(OptimizationApproach {
                    exploration_rate: 0.5,
                    exploitation_rate: 0.5,
                    mutation_rate: 0.15,
                    crossover_rate: 0.6,
                    population_size: 30,
                    generations: 15,
                })
            },
            AdaptationStrategy::Conservative => {
                Ok(OptimizationApproach {
                    exploration_rate: 0.7,
                    exploitation_rate: 0.3,
                    mutation_rate: 0.2,
                    crossover_rate: 0.5,
                    population_size: 40,
                    generations: 20,
                })
            },
            AdaptationStrategy::Stable => {
                Ok(OptimizationApproach {
                    exploration_rate: 0.9,
                    exploitation_rate: 0.1,
                    mutation_rate: 0.25,
                    crossover_rate: 0.4,
                    population_size: 50,
                    generations: 25,
                })
            },
        }
    }

    fn generate_candidate_parameters(&self,
                                   current: &CompressionParameters,
                                   content: &ContentComplexity,
                                   behavior: &ViewerBehavior,
                                   strategy: &AdaptationStrategy,
                                   approach: &OptimizationApproach) -> Result<Vec<CompressionParameters>, AfiyahError> {
        let mut candidates = Vec::new();
        
        // Add current parameters as baseline
        candidates.push(current.clone());

        // Generate random candidates
        for _ in 0..approach.population_size {
            let candidate = self.generate_random_candidate(current, content, behavior, approach)?;
            candidates.push(candidate);
        }

        // Generate heuristic-based candidates
        let heuristic_candidates = self.generate_heuristic_candidates(current, content, behavior, strategy)?;
        candidates.extend(heuristic_candidates);

        Ok(candidates)
    }

    fn generate_random_candidate(&self,
                               current: &CompressionParameters,
                               content: &ContentComplexity,
                               behavior: &ViewerBehavior,
                               approach: &OptimizationApproach) -> Result<CompressionParameters, AfiyahError> {
        let mut candidate = current.clone();

        // Apply random mutations to each parameter
        candidate.compression_ratio = self.mutate_parameter(
            current.compression_ratio, 
            self.parameter_bounds.compression_ratio, 
            approach.mutation_rate)?;
        
        candidate.quality_threshold = self.mutate_parameter(
            current.quality_threshold, 
            self.parameter_bounds.quality_threshold, 
            approach.mutation_rate)?;
        
        candidate.temporal_window = self.mutate_parameter(
            current.temporal_window, 
            self.parameter_bounds.temporal_window, 
            approach.mutation_rate)?;
        
        candidate.spatial_resolution = self.mutate_parameter(
            current.spatial_resolution, 
            self.parameter_bounds.spatial_resolution, 
            approach.mutation_rate)?;
        
        candidate.attention_weight = self.mutate_parameter(
            current.attention_weight, 
            self.parameter_bounds.attention_weight, 
            approach.mutation_rate)?;
        
        candidate.adaptation_rate = self.mutate_parameter(
            current.adaptation_rate, 
            self.parameter_bounds.adaptation_rate, 
            approach.mutation_rate)?;

        // Calculate derived parameters
        candidate.biological_accuracy = self.calculate_biological_accuracy(&candidate, content)?;
        candidate.perceptual_quality = self.calculate_perceptual_quality(&candidate, behavior)?;

        Ok(candidate)
    }

    fn generate_heuristic_candidates(&self,
                                   current: &CompressionParameters,
                                   content: &ContentComplexity,
                                   behavior: &ViewerBehavior,
                                   strategy: &AdaptationStrategy) -> Result<Vec<CompressionParameters>, AfiyahError> {
        let mut candidates = Vec::new();

        // Content-adaptive candidate
        let mut content_candidate = current.clone();
        content_candidate.compression_ratio = self.adapt_to_content_complexity(content)?;
        content_candidate.spatial_resolution = self.adapt_to_spatial_complexity(content)?;
        content_candidate.temporal_window = self.adapt_to_temporal_complexity(content)?;
        content_candidate.biological_accuracy = self.calculate_biological_accuracy(&content_candidate, content)?;
        content_candidate.perceptual_quality = self.calculate_perceptual_quality(&content_candidate, behavior)?;
        candidates.push(content_candidate);

        // Behavior-adaptive candidate
        let mut behavior_candidate = current.clone();
        behavior_candidate.attention_weight = self.adapt_to_attention_level(behavior)?;
        behavior_candidate.quality_threshold = self.adapt_to_engagement_level(behavior)?;
        behavior_candidate.adaptation_rate = self.adapt_to_consistency_level(behavior)?;
        behavior_candidate.biological_accuracy = self.calculate_biological_accuracy(&behavior_candidate, content)?;
        behavior_candidate.perceptual_quality = self.calculate_perceptual_quality(&behavior_candidate, behavior)?;
        candidates.push(behavior_candidate);

        // Strategy-specific candidate
        let strategy_candidate = self.generate_strategy_candidate(current, strategy, content, behavior)?;
        candidates.push(strategy_candidate);

        Ok(candidates)
    }

    fn generate_strategy_candidate(&self,
                                 current: &CompressionParameters,
                                 strategy: &AdaptationStrategy,
                                 content: &ContentComplexity,
                                 behavior: &ViewerBehavior) -> Result<CompressionParameters, AfiyahError> {
        let mut candidate = current.clone();

        match strategy {
            AdaptationStrategy::Emergency => {
                candidate.compression_ratio = 0.99; // Maximum compression
                candidate.quality_threshold = 0.5; // Lower quality threshold
                candidate.temporal_window = 0.1; // Shorter temporal window
                candidate.spatial_resolution = 0.5; // Lower spatial resolution
            },
            AdaptationStrategy::Aggressive => {
                candidate.compression_ratio = 0.95;
                candidate.quality_threshold = 0.7;
                candidate.temporal_window = 0.15;
                candidate.spatial_resolution = 0.8;
            },
            AdaptationStrategy::Balanced => {
                candidate.compression_ratio = 0.9;
                candidate.quality_threshold = 0.8;
                candidate.temporal_window = 0.2;
                candidate.spatial_resolution = 1.0;
            },
            AdaptationStrategy::Conservative => {
                candidate.compression_ratio = 0.8;
                candidate.quality_threshold = 0.9;
                candidate.temporal_window = 0.3;
                candidate.spatial_resolution = 1.2;
            },
            AdaptationStrategy::Stable => {
                // Keep current parameters with minor adjustments
                candidate.compression_ratio = current.compression_ratio * 1.05;
                candidate.quality_threshold = current.quality_threshold * 1.02;
            },
        }

        candidate.biological_accuracy = self.calculate_biological_accuracy(&candidate, content)?;
        candidate.perceptual_quality = self.calculate_perceptual_quality(&candidate, behavior)?;

        Ok(candidate)
    }

    fn evaluate_candidates(&self,
                         candidates: &[CompressionParameters],
                         content: &ContentComplexity,
                         behavior: &ViewerBehavior,
                         performance: &PerformanceMetrics) -> Result<CompressionParameters, AfiyahError> {
        let mut best_candidate = candidates[0].clone();
        let mut best_fitness = 0.0;

        for candidate in candidates {
            let fitness = self.calculate_fitness(candidate, content, behavior, performance)?;
            if fitness > best_fitness {
                best_fitness = fitness;
                best_candidate = candidate.clone();
            }
        }

        Ok(best_candidate)
    }

    fn apply_learning(&self, 
                     candidate: &CompressionParameters,
                     content: &ContentComplexity,
                     behavior: &ViewerBehavior) -> Result<CompressionParameters, AfiyahError> {
        let mut learned = candidate.clone();

        // Apply learning from adaptation memory
        if !self.adaptation_memory.is_empty() {
            let similar_memories = self.find_similar_memories(content, behavior)?;
            
            for memory in similar_memories {
                // Apply weighted learning
                let weight = self.calculate_memory_weight(&memory, content, behavior)?;
                
                learned.compression_ratio = learned.compression_ratio * (1.0 - weight) + 
                                          memory.parameters.compression_ratio * weight;
                learned.quality_threshold = learned.quality_threshold * (1.0 - weight) + 
                                          memory.parameters.quality_threshold * weight;
                learned.temporal_window = learned.temporal_window * (1.0 - weight) + 
                                        memory.parameters.temporal_window * weight;
                learned.spatial_resolution = learned.spatial_resolution * (1.0 - weight) + 
                                           memory.parameters.spatial_resolution * weight;
                learned.attention_weight = learned.attention_weight * (1.0 - weight) + 
                                         memory.parameters.attention_weight * weight;
            }
        }

        // Ensure parameters are within bounds
        learned = self.clamp_parameters(&learned)?;

        Ok(learned)
    }

    fn update_adaptation_memory(&mut self,
                              content: &ContentComplexity,
                              behavior: &ViewerBehavior,
                              performance: &PerformanceMetrics,
                              parameters: &CompressionParameters,
                              fitness: f64) -> Result<(), AfiyahError> {
        let memory = AdaptationMemory {
            content_complexity: content.overall_complexity,
            behavior_engagement: behavior.engagement_score,
            performance_efficiency: performance.efficiency,
            parameters: parameters.clone(),
            fitness,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.adaptation_memory.push(memory);
        
        // Maintain memory size
        if self.adaptation_memory.len() > 1000 {
            self.adaptation_memory.remove(0);
        }

        Ok(())
    }

    fn mutate_parameter(&self, current: f64, bounds: (f64, f64), mutation_rate: f64) -> Result<f64, AfiyahError> {
        let range = bounds.1 - bounds.0;
        let mutation = (rand::random::<f64>() - 0.5) * range * mutation_rate;
        let new_value = current + mutation;
        Ok(new_value.max(bounds.0).min(bounds.1))
    }

    fn adapt_to_content_complexity(&self, content: &ContentComplexity) -> Result<f64, AfiyahError> {
        // Higher compression for more complex content
        let base_ratio = 0.8;
        let complexity_factor = content.overall_complexity * 0.2;
        Ok((base_ratio + complexity_factor).min(0.99))
    }

    fn adapt_to_spatial_complexity(&self, content: &ContentComplexity) -> Result<f64, AfiyahError> {
        // Higher resolution for more spatially complex content
        let base_resolution = 1.0;
        let complexity_factor = content.spatial_complexity * 0.5;
        Ok((base_resolution + complexity_factor).min(2.0))
    }

    fn adapt_to_temporal_complexity(&self, content: &ContentComplexity) -> Result<f64, AfiyahError> {
        // Longer temporal window for more temporally complex content
        let base_window = 0.2;
        let complexity_factor = content.temporal_complexity * 0.3;
        Ok((base_window + complexity_factor).min(1.0))
    }

    fn adapt_to_attention_level(&self, behavior: &ViewerBehavior) -> Result<f64, AfiyahError> {
        // Higher attention weight for higher attention levels
        Ok(behavior.attention_level)
    }

    fn adapt_to_engagement_level(&self, behavior: &ViewerBehavior) -> Result<f64, AfiyahError> {
        // Higher quality threshold for higher engagement
        Ok(behavior.engagement_score)
    }

    fn adapt_to_consistency_level(&self, behavior: &ViewerBehavior) -> Result<f64, AfiyahError> {
        // Higher adaptation rate for less consistent behavior
        Ok((1.0 - behavior.consistency_score) * 0.5)
    }

    fn calculate_biological_accuracy(&self, parameters: &CompressionParameters, content: &ContentComplexity) -> Result<f64, AfiyahError> {
        // Calculate biological accuracy based on parameter settings
        let spatial_accuracy = 1.0 - (parameters.spatial_resolution - content.spatial_complexity).abs();
        let temporal_accuracy = 1.0 - (parameters.temporal_window - content.temporal_complexity).abs();
        let compression_accuracy = 1.0 - (parameters.compression_ratio - content.compression_difficulty).abs();
        
        let accuracy = (spatial_accuracy + temporal_accuracy + compression_accuracy) / 3.0;
        Ok(accuracy.max(0.8).min(1.0))
    }

    fn calculate_perceptual_quality(&self, parameters: &CompressionParameters, behavior: &ViewerBehavior) -> Result<f64, AfiyahError> {
        // Calculate perceptual quality based on viewer behavior
        let attention_quality = parameters.attention_weight * behavior.attention_level;
        let engagement_quality = parameters.quality_threshold * behavior.engagement_score;
        let consistency_quality = behavior.consistency_score;
        
        let quality = (attention_quality + engagement_quality + consistency_quality) / 3.0;
        Ok(quality.max(0.8).min(1.0))
    }

    fn find_similar_memories(&self, content: &ContentComplexity, behavior: &ViewerBehavior) -> Result<Vec<&AdaptationMemory>, AfiyahError> {
        let mut similar_memories = Vec::new();
        
        for memory in &self.adaptation_memory {
            let content_similarity = 1.0 - (memory.content_complexity - content.overall_complexity).abs();
            let behavior_similarity = 1.0 - (memory.behavior_engagement - behavior.engagement_score).abs();
            
            let similarity = (content_similarity + behavior_similarity) / 2.0;
            
            if similarity > 0.7 { // Threshold for similarity
                similar_memories.push(memory);
            }
        }
        
        // Sort by fitness and take top 5
        similar_memories.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        similar_memories.truncate(5);
        
        Ok(similar_memories)
    }

    fn calculate_memory_weight(&self, memory: &AdaptationMemory, content: &ContentComplexity, behavior: &ViewerBehavior) -> Result<f64, AfiyahError> {
        let content_similarity = 1.0 - (memory.content_complexity - content.overall_complexity).abs();
        let behavior_similarity = 1.0 - (memory.behavior_engagement - behavior.engagement_score).abs();
        let fitness_weight = memory.fitness;
        
        let weight = (content_similarity + behavior_similarity + fitness_weight) / 3.0;
        Ok(weight.min(0.5)) // Cap at 0.5 to avoid over-learning
    }

    fn clamp_parameters(&self, parameters: &CompressionParameters) -> Result<CompressionParameters, AfiyahError> {
        let mut clamped = parameters.clone();
        
        clamped.compression_ratio = clamped.compression_ratio
            .max(self.parameter_bounds.compression_ratio.0)
            .min(self.parameter_bounds.compression_ratio.1);
        
        clamped.quality_threshold = clamped.quality_threshold
            .max(self.parameter_bounds.quality_threshold.0)
            .min(self.parameter_bounds.quality_threshold.1);
        
        clamped.temporal_window = clamped.temporal_window
            .max(self.parameter_bounds.temporal_window.0)
            .min(self.parameter_bounds.temporal_window.1);
        
        clamped.spatial_resolution = clamped.spatial_resolution
            .max(self.parameter_bounds.spatial_resolution.0)
            .min(self.parameter_bounds.spatial_resolution.1);
        
        clamped.attention_weight = clamped.attention_weight
            .max(self.parameter_bounds.attention_weight.0)
            .min(self.parameter_bounds.attention_weight.1);
        
        clamped.adaptation_rate = clamped.adaptation_rate
            .max(self.parameter_bounds.adaptation_rate.0)
            .min(self.parameter_bounds.adaptation_rate.1);
        
        Ok(clamped)
    }
}

/// Optimization approach configuration
#[derive(Debug, Clone)]
struct OptimizationApproach {
    exploration_rate: f64,
    exploitation_rate: f64,
    mutation_rate: f64,
    crossover_rate: f64,
    population_size: usize,
    generations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::content_analyzer::ContentComplexity;
    use super::super::viewer_behavior_tracker::ViewerBehavior;
    use super::super::adaptation_controller::AdaptationStrategy;
    use super::super::performance_monitor::{PerformanceMetrics, SystemHealth};

    #[test]
    fn test_parameter_optimizer_creation() {
        let optimizer = ParameterOptimizer::new();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_parameter_optimization() {
        let mut optimizer = ParameterOptimizer::new().unwrap();
        let current = CompressionParameters::default();
        
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

        let strategy = AdaptationStrategy::Balanced;
        let performance = PerformanceMetrics {
            efficiency: 0.7,
            latency: 50.0,
            memory_usage: 0.6,
            cpu_usage: 0.5,
            variance: 0.2,
            throughput: 100.0,
        };

        let result = optimizer.optimize_parameters(&current, &content, &behavior, &strategy, &performance);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fitness_calculation() {
        let optimizer = ParameterOptimizer::new().unwrap();
        let parameters = CompressionParameters::default();
        
        let content = ContentComplexity {
            overall_complexity: 0.5,
            spatial_complexity: 0.5,
            temporal_complexity: 0.5,
            perceptual_complexity: 0.5,
            compression_difficulty: 0.5,
            biological_processing_load: 0.5,
            variance: 0.1,
            stability: 0.9,
        };

        let behavior = ViewerBehavior {
            gaze_position: (0.5, 0.5),
            saccade_frequency: 0.2,
            fixation_duration: 0.5,
            attention_level: 0.7,
            pupil_dilation: 0.6,
            blink_rate: 0.05,
            head_movement: 0.1,
            engagement_score: 0.8,
            consistency_score: 0.9,
            cognitive_load: 0.3,
        };

        let performance = PerformanceMetrics {
            efficiency: 0.8,
            latency: 30.0,
            memory_usage: 0.5,
            cpu_usage: 0.4,
            variance: 0.1,
            throughput: 120.0,
        };

        let fitness = optimizer.calculate_fitness(&parameters, &content, &behavior, &performance).unwrap();
        assert!(fitness >= 0.0 && fitness <= 1.0);
    }
}