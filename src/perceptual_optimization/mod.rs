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

//! Perceptual Optimization Module
//! 
//! This module implements perceptual optimization mechanisms including masking
//! algorithms, perceptual error models, foveal sampling, quality metrics, and
//! temporal prediction networks based on human visual system limitations.

pub mod masking_algorithms;
pub mod perceptual_error_model;
pub mod foveal_sampling;
pub mod quality_metrics;
pub mod temporal_prediction_networks;

use crate::AfiyahError;
use crate::synaptic_adaptation::AdaptedOutput;

/// Main perceptual optimizer that orchestrates all perceptual optimization mechanisms
pub struct PerceptualOptimizer {
    masking_algorithms: masking_algorithms::MaskingAlgorithms,
    perceptual_error_model: perceptual_error_model::PerceptualErrorModel,
    foveal_sampler: foveal_sampling::FovealSampler,
    quality_metrics: quality_metrics::QualityMetrics,
    temporal_predictor: temporal_prediction_networks::TemporalPredictionNetworks,
    optimization_state: OptimizationState,
}

impl PerceptualOptimizer {
    /// Creates a new perceptual optimizer with biological default parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            masking_algorithms: masking_algorithms::MaskingAlgorithms::new()?,
            perceptual_error_model: perceptual_error_model::PerceptualErrorModel::new()?,
            foveal_sampler: foveal_sampling::FovealSampler::new()?,
            quality_metrics: quality_metrics::QualityMetrics::new()?,
            temporal_predictor: temporal_prediction_networks::TemporalPredictionNetworks::new()?,
            optimization_state: OptimizationState::default(),
        })
    }

    /// Optimizes adapted output using perceptual optimization mechanisms
    pub fn optimize(&mut self, adapted_output: &AdaptedOutput) -> Result<OptimizedOutput, AfiyahError> {
        // Stage 1: Apply perceptual masking
        let masked_output = self.masking_algorithms.apply_masking(adapted_output)?;
        
        // Stage 2: Model perceptual errors
        let error_modeled_output = self.perceptual_error_model.model_errors(&masked_output)?;
        
        // Stage 3: Apply foveal sampling
        let foveal_output = self.foveal_sampler.sample(&error_modeled_output)?;
        
        // Stage 4: Calculate quality metrics
        let quality_metrics = self.quality_metrics.calculate_metrics(&foveal_output)?;
        
        // Stage 5: Apply temporal prediction
        let temporal_output = self.temporal_predictor.predict(&foveal_output)?;
        
        // Stage 6: Generate compressed output
        let compressed_output = self.generate_compressed_output(&temporal_output, &quality_metrics)?;
        
        // Update optimization state
        self.update_optimization_state(&compressed_output)?;
        
        Ok(OptimizedOutput {
            compressed_data: compressed_output.data,
            compression_ratio: compressed_output.compression_ratio,
            perceptual_quality: quality_metrics.overall_quality,
            biological_accuracy: quality_metrics.biological_accuracy,
            masking_efficiency: compressed_output.masking_efficiency,
            foveal_efficiency: compressed_output.foveal_efficiency,
        })
    }

    /// Trains perceptual optimization mechanisms
    pub fn train(&mut self, training_data: &[crate::VisualInput]) -> Result<(), AfiyahError> {
        // Train masking algorithms
        self.masking_algorithms.train(training_data)?;
        
        // Train perceptual error model
        self.perceptual_error_model.train(training_data)?;
        
        // Train foveal sampler
        self.foveal_sampler.train(training_data)?;
        
        // Train quality metrics
        self.quality_metrics.train(training_data)?;
        
        // Train temporal predictor
        self.temporal_predictor.train(training_data)?;
        
        Ok(())
    }

    fn generate_compressed_output(&self, temporal_output: &TemporalOutput, quality_metrics: &QualityMetrics) -> Result<CompressedData, AfiyahError> {
        // Generate compressed data based on temporal predictions and quality metrics
        let mut compressed_data = Vec::new();
        
        // Add temporal prediction data
        for prediction in &temporal_output.predictions {
            compressed_data.extend(prediction.predicted_frame.iter().map(|&x| (x * 255.0) as u8));
        }
        
        // Add quality metadata
        compressed_data.extend(quality_metrics.overall_quality.to_le_bytes());
        compressed_data.extend(quality_metrics.biological_accuracy.to_le_bytes());
        
        // Calculate compression ratio
        let original_size = temporal_output.predictions.len() * temporal_output.predictions[0].predicted_frame.len() * 4; // 4 bytes per float
        let compressed_size = compressed_data.len();
        let compression_ratio = 1.0 - (compressed_size as f64 / original_size as f64);
        
        Ok(CompressedData {
            data: compressed_data,
            compression_ratio,
            masking_efficiency: self.calculate_masking_efficiency(temporal_output),
            foveal_efficiency: self.calculate_foveal_efficiency(temporal_output),
        })
    }

    fn calculate_masking_efficiency(&self, temporal_output: &TemporalOutput) -> f64 {
        // Calculate masking efficiency based on temporal predictions
        let total_predictions = temporal_output.predictions.len() as f64;
        let accurate_predictions = temporal_output.predictions.iter()
            .filter(|p| p.prediction_confidence > 0.8)
            .count() as f64;
        
        accurate_predictions / total_predictions
    }

    fn calculate_foveal_efficiency(&self, temporal_output: &TemporalOutput) -> f64 {
        // Calculate foveal efficiency based on prediction accuracy
        let avg_confidence = temporal_output.predictions.iter()
            .map(|p| p.prediction_confidence)
            .sum::<f64>() / temporal_output.predictions.len() as f64;
        
        avg_confidence
    }

    fn update_optimization_state(&mut self, compressed_output: &CompressedData) -> Result<(), AfiyahError> {
        // Update optimization state based on compressed output
        self.optimization_state.compression_ratio = compressed_output.compression_ratio;
        self.optimization_state.masking_efficiency = compressed_output.masking_efficiency;
        self.optimization_state.foveal_efficiency = compressed_output.foveal_efficiency;
        
        Ok(())
    }
}

/// Output from masking algorithms
#[derive(Debug, Clone)]
pub struct MaskedOutput {
    pub masked_features: Vec<f64>,
    pub masking_strength: f64,
    pub masking_efficiency: f64,
}

/// Output from perceptual error modeling
#[derive(Debug, Clone)]
pub struct ErrorModeledOutput {
    pub error_modeled_features: Vec<f64>,
    pub perceptual_errors: Vec<f64>,
    pub error_threshold: f64,
}

/// Output from foveal sampling
#[derive(Debug, Clone)]
pub struct FovealOutput {
    pub foveal_features: Vec<f64>,
    pub foveal_radius: f64,
    pub resolution_factor: f64,
    pub sampling_efficiency: f64,
}

/// Quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub overall_quality: f64,
    pub biological_accuracy: f64,
    pub perceptual_similarity: f64,
    pub compression_efficiency: f64,
    pub temporal_consistency: f64,
}

/// Output from temporal prediction networks
#[derive(Debug, Clone)]
pub struct TemporalOutput {
    pub predictions: Vec<TemporalPrediction>,
    pub prediction_confidence: f64,
    pub temporal_consistency: f64,
}

/// Temporal prediction
#[derive(Debug, Clone)]
pub struct TemporalPrediction {
    pub predicted_frame: Vec<f64>,
    pub prediction_confidence: f64,
    pub prediction_error: f64,
    pub temporal_offset: f64,
}

/// Compressed data
#[derive(Debug, Clone)]
pub struct CompressedData {
    pub data: Vec<u8>,
    pub compression_ratio: f64,
    pub masking_efficiency: f64,
    pub foveal_efficiency: f64,
}

/// Final optimized output
#[derive(Debug, Clone)]
pub struct OptimizedOutput {
    pub compressed_data: Vec<u8>,
    pub compression_ratio: f64,
    pub perceptual_quality: f64,
    pub biological_accuracy: f64,
    pub masking_efficiency: f64,
    pub foveal_efficiency: f64,
}

/// Optimization state
#[derive(Debug, Clone)]
pub struct OptimizationState {
    pub compression_ratio: f64,
    pub masking_efficiency: f64,
    pub foveal_efficiency: f64,
    pub optimization_level: f64,
}

impl Default for OptimizationState {
    fn default() -> Self {
        Self {
            compression_ratio: 0.0,
            masking_efficiency: 0.0,
            foveal_efficiency: 0.0,
            optimization_level: 0.5,
        }
    }
}

/// Perceptual quality analyzer
pub struct PerceptualQualityAnalyzer {
    pub quality_threshold: f64,
    pub biological_accuracy_threshold: f64,
    pub temporal_consistency_threshold: f64,
}

impl PerceptualQualityAnalyzer {
    /// Creates a new perceptual quality analyzer
    pub fn new() -> Self {
        Self {
            quality_threshold: 0.8,
            biological_accuracy_threshold: 0.9,
            temporal_consistency_threshold: 0.7,
        }
    }

    /// Analyzes perceptual quality
    pub fn analyze_quality(&self, output: &OptimizedOutput) -> Result<QualityAnalysis, AfiyahError> {
        let quality_score = output.perceptual_quality;
        let biological_score = output.biological_accuracy;
        let compression_score = output.compression_ratio;
        
        let is_high_quality = quality_score >= self.quality_threshold;
        let is_biologically_accurate = biological_score >= self.biological_accuracy_threshold;
        let is_efficiently_compressed = compression_score >= 0.9;
        
        let overall_score = (quality_score + biological_score + compression_score) / 3.0;
        
        Ok(QualityAnalysis {
            overall_score,
            quality_score,
            biological_score,
            compression_score,
            is_high_quality,
            is_biologically_accurate,
            is_efficiently_compressed,
            recommendations: self.generate_recommendations(output),
        })
    }

    fn generate_recommendations(&self, output: &OptimizedOutput) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if output.perceptual_quality < self.quality_threshold {
            recommendations.push("Increase perceptual quality by improving masking algorithms".to_string());
        }
        
        if output.biological_accuracy < self.biological_accuracy_threshold {
            recommendations.push("Improve biological accuracy by refining neural models".to_string());
        }
        
        if output.compression_ratio < 0.9 {
            recommendations.push("Enhance compression ratio by optimizing temporal prediction".to_string());
        }
        
        if output.masking_efficiency < 0.8 {
            recommendations.push("Improve masking efficiency by adjusting perceptual thresholds".to_string());
        }
        
        if output.foveal_efficiency < 0.8 {
            recommendations.push("Enhance foveal efficiency by optimizing attention mechanisms".to_string());
        }
        
        recommendations
    }
}

/// Quality analysis result
#[derive(Debug, Clone)]
pub struct QualityAnalysis {
    pub overall_score: f64,
    pub quality_score: f64,
    pub biological_score: f64,
    pub compression_score: f64,
    pub is_high_quality: bool,
    pub is_biologically_accurate: bool,
    pub is_efficiently_compressed: bool,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_optimizer_creation() {
        let optimizer = PerceptualOptimizer::new();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_optimization_state_defaults() {
        let state = OptimizationState::default();
        assert_eq!(state.compression_ratio, 0.0);
        assert_eq!(state.optimization_level, 0.5);
    }

    #[test]
    fn test_perceptual_quality_analyzer() {
        let analyzer = PerceptualQualityAnalyzer::new();
        let output = OptimizedOutput {
            compressed_data: vec![1, 2, 3],
            compression_ratio: 0.95,
            perceptual_quality: 0.9,
            biological_accuracy: 0.95,
            masking_efficiency: 0.85,
            foveal_efficiency: 0.9,
        };
        let analysis = analyzer.analyze_quality(&output);
        assert!(analysis.is_ok());
    }
}